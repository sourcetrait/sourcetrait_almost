//! Engine-side adapter consumption: deltas merged at the loader seam.
use crate::*;

/// The adapter artifact format version (metadata "version").
const ADAPTER_VERSION: &str = "1";

/// The three-tier adapter token resolution, as snapshot tokens use.
pub fn adapter_path(adapters_dir: &Path, token: &str) -> LibQuestResult<PathBuf> {
    if config::is_profile_name(Path::new(token)) {
        return Ok(adapters_dir.join(format!("{token}.safetensors")));
    }
    if !(token.starts_with('/')
        || token.starts_with("./")
        || token.starts_with("../")
        || token.starts_with('~')
        || token.starts_with('$'))
    {
        return Ok(adapters_dir.join(token));
    }
    config::expand_path(token)
}

/// One target weight's delta: a low-rank pair, a dense conv delta, or
/// the channel columns' head rows.
#[derive(Debug)]
pub(crate) enum AdapterDelta {
    LowRank {
        a: candle_core::Tensor,
        b: candle_core::Tensor,
        scale: f64,
    },
    Dense {
        delta: candle_core::Tensor,
    },
    /// Seven unembedding rows (config, then the six-run block), added at
    /// the real channel ids; never materialized vocabulary-wide.
    Rows {
        delta: candle_core::Tensor,
    },
}

impl AdapterDelta {
    /// The dense f32 delta in the checkpoint weight's own layout.
    fn materialize(&self) -> candle_core::Result<candle_core::Tensor> {
        match self {
            AdapterDelta::LowRank { a, b, scale } => {
                (a.matmul(b)? * *scale)?.t()?.contiguous()
            }
            AdapterDelta::Dense { delta } => Ok(delta.clone()),
            AdapterDelta::Rows { .. } => unreachable!("rows apply by slice, never whole"),
        }
    }

    /// Merge this delta into a base weight, in f32.
    fn merge(
        &self,
        base: candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        let base = base.to_dtype(candle_core::DType::F32)?;
        match self {
            AdapterDelta::Rows { delta } => {
                let hidden = delta.dims()[1];
                let config_at = consts::TOKEN_EXTRA_ID_0 as usize;
                let run_at = consts::TOKEN_EXTRA_ID_1 as usize;
                let config_rows = (base.narrow(0, config_at, 1)? + delta.narrow(0, 0, 1)?)?;
                let run_rows = (base.narrow(0, run_at, 6)? + delta.narrow(0, 1, 6)?)?;
                base.slice_assign(&[config_at..config_at + 1, 0..hidden], &config_rows)?
                    .slice_assign(&[run_at..run_at + 6, 0..hidden], &run_rows)
            }
            _ => base + self.materialize()?,
        }
    }
}

/// Map an adapter tensor name to its checkpoint weight, or reject it.
fn target_weight_name(delta_name: &str) -> LibQuestResult<(String, &'static str)> {
    if delta_name == consts::HEAD_DELTA_TENSOR {
        return Ok((String::from("lm_head.weight"), "rows"));
    }
    let Some(rest) = delta_name.strip_prefix("model.layers.") else {
        snafu::whatever!("adapter tensor {delta_name} is outside model.layers");
    };
    let mut segments = rest.split('.');
    let Some(index) = segments.next().filter(|s| s.parse::<usize>().is_ok()) else {
        snafu::whatever!("adapter tensor {delta_name} carries no layer index");
    };
    let module = segments.next().unwrap_or_default();
    let target = segments.next().unwrap_or_default();
    let kind = segments.next().unwrap_or_default();
    snafu::ensure_whatever!(
        segments.next().is_none(),
        "adapter tensor {delta_name} has trailing segments"
    );
    let sanctioned = matches!(
        (module, target),
        ("linear_attn", "q_proj" | "g_proj" | "o_proj")
            | ("self_attn", "q_proj" | "o_proj")
            | ("mlp", "gate_proj" | "up_proj" | "down_proj")
    );
    let conv = (module, target) == ("linear_attn", "q_conv1d");
    match kind {
        "lora_a" | "lora_b" if sanctioned => Ok((
            format!("model.layers.{index}.{module}.{target}.weight"),
            kind_str(kind),
        )),
        "delta" if conv => Ok((
            format!("model.layers.{index}.{module}.{target}.weight"),
            "delta",
        )),
        _ => snafu::whatever!(
            "adapter tensor {delta_name} targets outside the direction-3 surface \
             (state-carrying weights never adapt)"
        ),
    }
}

fn kind_str(kind: &str) -> &'static str {
    match kind {
        "lora_a" => "lora_a",
        _ => "lora_b",
    }
}

/// A validated adapter file, keyed by the weight names it merges into.
#[derive(Debug)]
pub(crate) struct AdapterFile {
    pub(crate) deltas: HashMap<String, AdapterDelta>,
}

/// Read a rank-checked f32 tensor from the adapter file onto the
/// target device.
fn adapter_tensor(
    view: &safetensors::tensor::TensorView,
    name: &str,
    device: &candle_core::Device,
) -> LibQuestResult<candle_core::Tensor> {
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
    Ok(candle_core::Tensor::from_vec(
        values,
        view.shape().to_vec(),
        device,
    )?)
}

impl AdapterFile {
    /// Parse and validate an adapter artifact onto the target device.
    pub(crate) fn load(
        path: &Path,
        expected_model_id: &str,
        device: &candle_core::Device,
    ) -> LibQuestResult<Self> {
        let bytes = fs::read(path)?;
        let (_, header) = match safetensors::SafeTensors::read_metadata(&bytes) {
            Ok(header) => header,
            Err(e) => snafu::whatever!("adapter {} header parse failed: {e}", path.display()),
        };
        let Some(metadata) = header.metadata().as_ref() else {
            snafu::whatever!("adapter {} carries no metadata", path.display());
        };
        snafu::ensure_whatever!(
            metadata.get("version").map(String::as_str) == Some(ADAPTER_VERSION),
            "adapter {} version {:?} is not {ADAPTER_VERSION}",
            path.display(),
            metadata.get("version")
        );
        let model_id = metadata.get("model_id").cloned().unwrap_or_default();
        snafu::ensure_whatever!(
            model_id == expected_model_id,
            "adapter {} was trained for {model_id}, not {expected_model_id}",
            path.display()
        );
        let rank: usize = match metadata.get("lora_rank").and_then(|r| r.parse().ok()) {
            Some(rank) => rank,
            None => snafu::whatever!("adapter {} carries no lora_rank", path.display()),
        };
        let alpha: f64 = match metadata.get("lora_alpha").and_then(|a| a.parse().ok()) {
            Some(alpha) => alpha,
            None => snafu::whatever!("adapter {} carries no lora_alpha", path.display()),
        };
        let scale = alpha / rank as f64;

        let parsed = match safetensors::SafeTensors::deserialize(&bytes) {
            Ok(parsed) => parsed,
            Err(e) => snafu::whatever!("adapter {} parse failed: {e}", path.display()),
        };
        struct Pending {
            a: Option<candle_core::Tensor>,
            b: Option<candle_core::Tensor>,
            conv: Option<candle_core::Tensor>,
            rows: Option<candle_core::Tensor>,
        }
        let mut pending: HashMap<String, Pending> = HashMap::new();
        for (name, view) in parsed.tensors() {
            let (weight_name, kind) = target_weight_name(&name)?;
            let tensor = adapter_tensor(&view, &name, device)?;
            let entry = pending.entry(weight_name).or_insert(Pending {
                a: None,
                b: None,
                conv: None,
                rows: None,
            });
            match kind {
                "lora_a" => {
                    snafu::ensure_whatever!(
                        tensor.dims().len() == 2 && tensor.dims()[1] == rank,
                        "adapter tensor {name}: lora_a must be [in, rank {rank}]"
                    );
                    entry.a = Some(tensor);
                }
                "lora_b" => {
                    snafu::ensure_whatever!(
                        tensor.dims().len() == 2 && tensor.dims()[0] == rank,
                        "adapter tensor {name}: lora_b must be [rank {rank}, out]"
                    );
                    entry.b = Some(tensor);
                }
                "rows" => {
                    snafu::ensure_whatever!(
                        tensor.dims().len() == 2 && tensor.dims()[0] == 7,
                        "adapter tensor {name}: channel rows must be [7, hidden]"
                    );
                    entry.rows = Some(tensor);
                }
                _ => {
                    snafu::ensure_whatever!(
                        tensor.dims().len() == 3,
                        "adapter tensor {name}: conv delta must be rank 3"
                    );
                    entry.conv = Some(tensor);
                }
            }
        }

        let mut deltas = HashMap::new();
        for (weight_name, entry) in pending {
            let delta = match (entry.a, entry.b, entry.conv, entry.rows) {
                (Some(a), Some(b), None, None) => AdapterDelta::LowRank { a, b, scale },
                (None, None, Some(delta), None) => AdapterDelta::Dense { delta },
                (None, None, None, Some(delta)) => AdapterDelta::Rows { delta },
                _ => snafu::whatever!(
                    "adapter target {weight_name} is incomplete (needs lora_a + lora_b, \
                     one conv delta, or one channel-rows delta)"
                ),
            };
            deltas.insert(weight_name, delta);
        }
        snafu::ensure_whatever!(!deltas.is_empty(), "adapter {} is empty", path.display());
        Ok(Self { deltas })
    }

    /// The wrong-model geometry guard, against the checkpoint header.
    pub(crate) fn validate_geometry(
        &self,
        inventory: &[TensorInfo],
    ) -> LibQuestResult<()> {
        let shapes: HashMap<&str, &Vec<usize>> = inventory
            .iter()
            .map(|info| (info.name.as_str(), &info.shape))
            .collect();
        for (weight_name, delta) in &self.deltas {
            let Some(shape) = shapes.get(weight_name.as_str()) else {
                snafu::whatever!("adapter target {weight_name} is not in the checkpoint");
            };
            match delta {
                AdapterDelta::LowRank { a, b, .. } => {
                    let expected = vec![b.dims()[1], a.dims()[0]];
                    snafu::ensure_whatever!(
                        **shape == expected,
                        "adapter target {weight_name}: checkpoint shape {shape:?} vs delta {expected:?}"
                    );
                }
                AdapterDelta::Dense { delta } => {
                    let expected = delta.dims().to_vec();
                    snafu::ensure_whatever!(
                        **shape == expected,
                        "adapter target {weight_name}: checkpoint shape {shape:?} vs delta {expected:?}"
                    );
                }
                AdapterDelta::Rows { delta } => {
                    let hidden = delta.dims()[1];
                    let past_run = consts::TOKEN_EXTRA_ID_1 as usize + 6;
                    snafu::ensure_whatever!(
                        shape.len() == 2 && shape[1] == hidden && shape[0] >= past_run,
                        "adapter target {weight_name}: checkpoint shape {shape:?} cannot \
                         take [7, {hidden}] channel rows"
                    );
                }
            }
        }
        Ok(())
    }
}

/// The merging VarBuilder backend, wrapping the mmaped checkpoint.
struct DeltaBackend {
    inner: candle_core::safetensors::MmapedSafetensors,
    deltas: HashMap<String, AdapterDelta>,
}

impl DeltaBackend {
    fn apply(
        &self,
        base: candle_core::Tensor,
        name: &str,
        dtype: candle_core::DType,
    ) -> candle_core::Result<candle_core::Tensor> {
        let Some(delta) = self.deltas.get(name) else {
            return Ok(base);
        };
        delta.merge(base)?.to_dtype(dtype)
    }
}

impl candle_nn::var_builder::SimpleBackend for DeltaBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: candle_core::DType,
        dev: &candle_core::Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        let base =
            candle_nn::var_builder::SimpleBackend::get(&self.inner, s, name, h, dtype, dev)?;
        self.apply(base, name, dtype)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: candle_core::DType,
        dev: &candle_core::Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        let base = candle_nn::var_builder::SimpleBackend::get_unchecked(
            &self.inner,
            name,
            dtype,
            dev,
        )?;
        self.apply(base, name, dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name)
    }
}

/// The adapter-aware weight loader; no token means the plain load.
pub fn load_weights(
    config: &LibConfig,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> LibQuestResult<candle_nn::VarBuilder<'static>> {
    let model_dir = config.model_dir();
    let Some(token) = &config.adapter else {
        return mmap_weights(&model_dir, dtype, device);
    };
    let path = adapter_path(&config.adapters_dir, token)?;
    let file = AdapterFile::load(&path, &config.model, device)?;
    file.validate_geometry(&tensor_inventory(&model_dir)?)?;
    let shard = model_dir.join("model.safetensors");
    let inner = unsafe { candle_core::safetensors::MmapedSafetensors::new(&shard)? };
    let backend = DeltaBackend {
        inner,
        deltas: file.deltas,
    };
    Ok(candle_nn::VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}
