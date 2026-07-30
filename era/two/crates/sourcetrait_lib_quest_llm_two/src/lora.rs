//! Frozen-base additive adapters, placed by construction.
#![allow(dead_code)]
use crate::*;

use burn::tensor::backend::Backend;

/// One adapted projection's low-rank pair, contributing (x.a).b * scale.
pub struct LoraPair<B: Backend> {
    pub a: burn::tensor::Tensor<B, 2>,
    pub b: burn::tensor::Tensor<B, 2>,
    pub scale: f64,
}

/// One standard-normal draw, Box-Muller over SplitMix64.
fn normal_draw(rng: &mut SplitMix64) -> f64 {
    let mut first_unit = rng.next_unit();
    if first_unit <= f64::MIN_POSITIVE {
        first_unit = f64::MIN_POSITIVE;
    }
    let second_unit = rng.next_unit();
    (-2.0 * first_unit.ln()).sqrt() * (2.0 * std::f64::consts::PI * second_unit).cos()
}

impl<B: Backend> LoraPair<B> {
    /// Seeded init: a ~ N(0, 0.02) host-drawn, b zeros.
    pub fn init(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f64,
        rng: &mut SplitMix64,
        device: &B::Device,
    ) -> Self {
        let mut a_values = Vec::with_capacity(in_dim * rank);
        for _ in 0..in_dim * rank {
            a_values.push((normal_draw(rng) * 0.02) as f32);
        }
        let a = burn::tensor::Tensor::from_data(
            burn::tensor::TensorData::new(a_values, [in_dim, rank]),
            device,
        )
        .require_grad();
        let b = burn::tensor::Tensor::zeros([rank, out_dim], device).require_grad();
        Self { a, b, scale: alpha / rank as f64 }
    }

    /// The additive contribution for an [n, in] activation:
    /// (x.a).b * scale.
    pub fn contribution(
        &self,
        x: burn::tensor::Tensor<B, 2>,
    ) -> burn::tensor::Tensor<B, 2> {
        x.matmul(self.a.clone()).matmul(self.b.clone()).mul_scalar(self.scale)
    }

    /// The dense [in, out] delta this pair merges into a base weight.
    #[allow(dead_code)]
    pub fn merged_delta(&self) -> burn::tensor::Tensor<B, 2> {
        self.a.clone().matmul(self.b.clone()).mul_scalar(self.scale)
    }
}

/// The q_conv1d full delta: one zero-init (1, channels) row per tap.
pub struct ConvDelta<B: Backend> {
    pub taps: Vec<burn::tensor::Tensor<B, 2>>,
}

impl<B: Backend> ConvDelta<B> {
    pub fn init(channels: usize, kernel: usize, device: &B::Device) -> Self {
        let taps = (0..kernel)
            .map(|_| burn::tensor::Tensor::zeros([1, channels], device).require_grad())
            .collect();
        Self { taps }
    }

    /// base tap rows + delta tap rows, tap-major.
    pub fn effective_taps(
        &self,
        base: &[burn::tensor::Tensor<B, 2>],
    ) -> Vec<burn::tensor::Tensor<B, 2>> {
        base.iter()
            .zip(self.taps.iter())
            .map(|(base_tap, delta_tap)| base_tap.clone() + delta_tap.clone())
            .collect()
    }
}

/// A GDN layer's sanctioned adapters (readout surface only).
pub struct GdnAdapters<B: Backend> {
    pub q: LoraPair<B>,
    pub q_conv: ConvDelta<B>,
    pub g: LoraPair<B>,
    pub o: LoraPair<B>,
    pub gate: LoraPair<B>,
    pub up: LoraPair<B>,
    pub down: LoraPair<B>,
}

/// An attention layer's sanctioned adapters (q/o + MLP; k/v never).
pub struct AttnAdapters<B: Backend> {
    pub q: LoraPair<B>,
    pub o: LoraPair<B>,
    pub gate: LoraPair<B>,
    pub up: LoraPair<B>,
    pub down: LoraPair<B>,
}

pub enum LayerAdapters<B: Backend> {
    Gdn(GdnAdapters<B>),
    Attn(AttnAdapters<B>),
}

impl<B: Backend> LayerAdapters<B> {
    /// Named refs to this layer's trainable tensors, all rank 2.
    pub fn params(
        &self,
        prefix: &str,
    ) -> Vec<(String, &burn::tensor::Tensor<B, 2>)> {
        let mut params = Vec::new();
        match self {
            LayerAdapters::Gdn(adapters) => {
                pair_refs(&mut params, prefix, "linear_attn.q_proj", &adapters.q);
                for (tap, row) in adapters.q_conv.taps.iter().enumerate() {
                    params.push((
                        format!("{prefix}.linear_attn.q_conv1d.delta_tap{tap}"),
                        row,
                    ));
                }
                pair_refs(&mut params, prefix, "linear_attn.g_proj", &adapters.g);
                pair_refs(&mut params, prefix, "linear_attn.o_proj", &adapters.o);
                pair_refs(&mut params, prefix, "mlp.gate_proj", &adapters.gate);
                pair_refs(&mut params, prefix, "mlp.up_proj", &adapters.up);
                pair_refs(&mut params, prefix, "mlp.down_proj", &adapters.down);
            }
            LayerAdapters::Attn(adapters) => {
                pair_refs(&mut params, prefix, "self_attn.q_proj", &adapters.q);
                pair_refs(&mut params, prefix, "self_attn.o_proj", &adapters.o);
                pair_refs(&mut params, prefix, "mlp.gate_proj", &adapters.gate);
                pair_refs(&mut params, prefix, "mlp.up_proj", &adapters.up);
                pair_refs(&mut params, prefix, "mlp.down_proj", &adapters.down);
            }
        }
        params
    }

    /// The mutable walk (the optimizer step), same names and order as
    /// `params`.
    pub fn params_mut(
        &mut self,
        prefix: &str,
    ) -> Vec<(String, &mut burn::tensor::Tensor<B, 2>)> {
        let mut params = Vec::new();
        match self {
            LayerAdapters::Gdn(adapters) => {
                pair_muts(&mut params, prefix, "linear_attn.q_proj", &mut adapters.q);
                for (tap, row) in adapters.q_conv.taps.iter_mut().enumerate() {
                    params.push((
                        format!("{prefix}.linear_attn.q_conv1d.delta_tap{tap}"),
                        row,
                    ));
                }
                pair_muts(&mut params, prefix, "linear_attn.g_proj", &mut adapters.g);
                pair_muts(&mut params, prefix, "linear_attn.o_proj", &mut adapters.o);
                pair_muts(&mut params, prefix, "mlp.gate_proj", &mut adapters.gate);
                pair_muts(&mut params, prefix, "mlp.up_proj", &mut adapters.up);
                pair_muts(&mut params, prefix, "mlp.down_proj", &mut adapters.down);
            }
            LayerAdapters::Attn(adapters) => {
                pair_muts(&mut params, prefix, "self_attn.q_proj", &mut adapters.q);
                pair_muts(&mut params, prefix, "self_attn.o_proj", &mut adapters.o);
                pair_muts(&mut params, prefix, "mlp.gate_proj", &mut adapters.gate);
                pair_muts(&mut params, prefix, "mlp.up_proj", &mut adapters.up);
                pair_muts(&mut params, prefix, "mlp.down_proj", &mut adapters.down);
            }
        }
        params
    }
}

fn pair_refs<'t, B: Backend>(
    params: &mut Vec<(String, &'t burn::tensor::Tensor<B, 2>)>,
    prefix: &str,
    target: &str,
    pair: &'t LoraPair<B>,
) {
    params.push((format!("{prefix}.{target}.lora_a"), &pair.a));
    params.push((format!("{prefix}.{target}.lora_b"), &pair.b));
}

fn pair_muts<'t, B: Backend>(
    params: &mut Vec<(String, &'t mut burn::tensor::Tensor<B, 2>)>,
    prefix: &str,
    target: &str,
    pair: &'t mut LoraPair<B>,
) {
    params.push((format!("{prefix}.{target}.lora_a"), &mut pair.a));
    params.push((format!("{prefix}.{target}.lora_b"), &mut pair.b));
}

/// The whole model's trainable state: one LayerAdapters per layer.
pub struct ModelAdapters<B: Backend> {
    pub layers: Vec<LayerAdapters<B>>,
    pub rank: usize,
    pub alpha: f64,
    pub seed: u64,
}

impl<B: Backend> ModelAdapters<B> {
    pub fn init(
        config: &HybridCheckpointConfig,
        rank: usize,
        alpha: f64,
        seed: u64,
        device: &B::Device,
    ) -> LibQuestResult<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let gdn_key_width = config.linear_num_key_heads * config.linear_key_head_dim;
        let gdn_value_width = config.linear_num_value_heads * config.linear_value_head_dim;
        let mut rng = SplitMix64::new(seed);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = if config.is_gdn_layer(index)? {
                LayerAdapters::Gdn(GdnAdapters {
                    q: LoraPair::init(hidden, gdn_key_width, rank, alpha, &mut rng, device),
                    q_conv: ConvDelta::init(
                        gdn_key_width,
                        config.linear_conv_kernel_dim,
                        device,
                    ),
                    g: LoraPair::init(hidden, gdn_value_width, rank, alpha, &mut rng, device),
                    o: LoraPair::init(gdn_value_width, hidden, rank, alpha, &mut rng, device),
                    gate: LoraPair::init(hidden, intermediate, rank, alpha, &mut rng, device),
                    up: LoraPair::init(hidden, intermediate, rank, alpha, &mut rng, device),
                    down: LoraPair::init(intermediate, hidden, rank, alpha, &mut rng, device),
                })
            } else {
                LayerAdapters::Attn(AttnAdapters {
                    q: LoraPair::init(hidden, hidden, rank, alpha, &mut rng, device),
                    o: LoraPair::init(hidden, hidden, rank, alpha, &mut rng, device),
                    gate: LoraPair::init(hidden, intermediate, rank, alpha, &mut rng, device),
                    up: LoraPair::init(hidden, intermediate, rank, alpha, &mut rng, device),
                    down: LoraPair::init(intermediate, hidden, rank, alpha, &mut rng, device),
                })
            };
            layers.push(layer);
        }
        Ok(Self { layers, rank, alpha, seed })
    }

    /// Resume from a saved adapter, at the artifact's own geometry.
    pub fn load(
        path: &Path,
        config: &HybridCheckpointConfig,
        expected_model_id: &str,
        device: &B::Device,
    ) -> LibQuestResult<Self> {
        let bytes = fs::read(path)?;
        let (_, header) = match safetensors::SafeTensors::read_metadata(&bytes) {
            Ok(header) => header,
            Err(error) => snafu::whatever!("adapter header parse failed: {error}"),
        };
        let Some(metadata) = header.metadata().as_ref() else {
            snafu::whatever!("adapter {} carries no metadata", path.display());
        };
        let field = |key: &str| -> LibQuestResult<String> {
            match metadata.get(key) {
                Some(value) => Ok(value.clone()),
                None => snafu::whatever!("adapter metadata is missing {key}"),
            }
        };
        snafu::ensure_whatever!(
            field("version")? == "1",
            "adapter version {} is not the supported 1",
            field("version")?
        );
        let model_id = field("model_id")?;
        snafu::ensure_whatever!(
            model_id == expected_model_id,
            "adapter was trained for {model_id}, not {expected_model_id}"
        );
        let Ok(rank) = field("lora_rank")?.parse::<usize>() else {
            snafu::whatever!("adapter lora_rank does not parse");
        };
        let Ok(alpha) = field("lora_alpha")?.parse::<f64>() else {
            snafu::whatever!("adapter lora_alpha does not parse");
        };
        let seed = field("seed")?.parse::<u64>().unwrap_or(0);

        let parsed = match safetensors::SafeTensors::deserialize(&bytes) {
            Ok(parsed) => parsed,
            Err(error) => snafu::whatever!("adapter parse failed: {error}"),
        };
        let mut stored: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (name, view) in parsed.tensors() {
            snafu::ensure_whatever!(
                view.dtype() == safetensors::Dtype::F32,
                "adapter tensor {name}: expected f32, got {:?}",
                view.dtype()
            );
            let values: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
                .collect();
            stored.insert(name, (view.shape().to_vec(), values));
        }

        let mut adapters = Self::init(config, rank, alpha, seed, device)?;
        for (name, tensor) in adapters.params_mut() {
            let replacement = match name.split_once(".delta_tap") {
                Some((prefix, tap_index)) => {
                    let Ok(tap) = tap_index.parse::<usize>() else {
                        snafu::whatever!("adapter tap index in {name} does not parse");
                    };
                    let key = format!("{prefix}.delta");
                    let Some((shape, values)) = stored.get(&key) else {
                        snafu::whatever!("adapter is missing {key}");
                    };
                    snafu::ensure_whatever!(
                        shape.len() == 3 && shape[1] == 1,
                        "{key}: expected (channels, 1, kernel), got {shape:?}"
                    );
                    let (channels, kernel) = (shape[0], shape[2]);
                    snafu::ensure_whatever!(tap < kernel, "{name}: tap beyond kernel {kernel}");
                    let row: Vec<f32> =
                        (0..channels).map(|channel| values[channel * kernel + tap]).collect();
                    burn::tensor::Tensor::from_data(
                        burn::tensor::TensorData::new(row, [1, channels]),
                        device,
                    )
                }
                None => {
                    let Some((shape, values)) = stored.get(&name) else {
                        snafu::whatever!("adapter is missing {name}");
                    };
                    snafu::ensure_whatever!(
                        shape.len() == 2,
                        "{name}: expected rank 2, got {shape:?}"
                    );
                    snafu::ensure_whatever!(
                        *shape == tensor.dims().to_vec(),
                        "{name}: artifact shape {shape:?} does not match the model's {:?}",
                        tensor.dims()
                    );
                    burn::tensor::Tensor::from_data(
                        burn::tensor::TensorData::new(values.clone(), [shape[0], shape[1]]),
                        device,
                    )
                }
            };
            *tensor = replacement.require_grad();
        }
        Ok(adapters)
    }

    /// Named refs to every trainable tensor, layer-major.
    #[allow(dead_code)]
    pub fn params(&self) -> Vec<(String, &burn::tensor::Tensor<B, 2>)> {
        let mut params = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            params.extend(layer.params(&format!("model.layers.{index}")));
        }
        params
    }

    /// The mutable optimizer walk, same names and order.
    pub fn params_mut(&mut self) -> Vec<(String, &mut burn::tensor::Tensor<B, 2>)> {
        let mut params = Vec::new();
        for (index, layer) in self.layers.iter_mut().enumerate() {
            params.extend(layer.params_mut(&format!("model.layers.{index}")));
        }
        params
    }

    /// Persist the adapter as one safetensors file plus its metadata.
    pub fn save(&self, path: &Path, model_id: &str) -> LibQuestResult<()> {
        let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("model.layers.{index}");
            for (name, tensor) in layer.params(&prefix) {
                if name.contains(".delta_tap") {
                    continue;
                }
                let dims = tensor.dims().to_vec();
                buffers.push((name, dims, tensor_f32_bytes(tensor)?));
            }
            if let LayerAdapters::Gdn(adapters) = layer {
                let kernel = adapters.q_conv.taps.len();
                let channels = adapters.q_conv.taps[0].dims()[1];
                let mut values = vec![0f32; channels * kernel];
                for (tap, row) in adapters.q_conv.taps.iter().enumerate() {
                    let row_values = tensor_f32_values(row)?;
                    for (channel, value) in row_values.iter().enumerate() {
                        values[channel * kernel + tap] = *value;
                    }
                }
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                buffers.push((
                    format!("{prefix}.linear_attn.q_conv1d.delta"),
                    vec![channels, 1, kernel],
                    bytes,
                ));
            }
        }
        buffers.sort_by(|left, right| left.0.cmp(&right.0));

        let views: Vec<(String, safetensors::tensor::TensorView)> = buffers
            .iter()
            .map(|(name, dims, bytes)| {
                match safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    dims.clone(),
                    bytes,
                ) {
                    Ok(view) => Ok((name.clone(), view)),
                    Err(error) => snafu::whatever!("adapter view {name} failed: {error}"),
                }
            })
            .collect::<LibQuestResult<Vec<_>>>()?;
        let mut metadata = HashMap::new();
        metadata.insert(String::from("version"), String::from("1"));
        metadata.insert(String::from("model_id"), model_id.to_string());
        metadata.insert(String::from("lora_rank"), self.rank.to_string());
        metadata.insert(String::from("lora_alpha"), self.alpha.to_string());
        metadata.insert(String::from("seed"), self.seed.to_string());
        metadata.insert(
            String::from("targets"),
            String::from(
                "gdn:q_proj,q_conv1d,g_proj,o_proj;attn:q_proj,o_proj;mlp:gate_proj,up_proj,down_proj",
            ),
        );
        metadata.insert(
            String::from("tool_version"),
            String::from(env!("CARGO_PKG_VERSION")),
        );
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        match safetensors::serialize_to_file(views, Some(metadata), path) {
            Ok(()) => Ok(()),
            Err(error) => snafu::whatever!("adapter serialization failed: {error}"),
        }
    }
}

fn tensor_f32_values<B: Backend>(
    tensor: &burn::tensor::Tensor<B, 2>,
) -> LibQuestResult<Vec<f32>> {
    let data = tensor.clone().into_data().convert::<f32>();
    match data.to_vec::<f32>() {
        Ok(values) => Ok(values),
        Err(error) => snafu::whatever!("adapter tensor extraction failed: {error:?}"),
    }
}

fn tensor_f32_bytes<B: Backend>(
    tensor: &burn::tensor::Tensor<B, 2>,
) -> LibQuestResult<Vec<u8>> {
    Ok(tensor_f32_values(tensor)?
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect())
}
