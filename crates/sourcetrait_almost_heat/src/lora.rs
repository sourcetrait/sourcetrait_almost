use crate::*;

use burn::tensor::{
    Distribution,
    Tensor,
    backend::Backend,
};

/// One low-rank adapter over a frozen linear: contribution =
/// (x . a) . b * (alpha / rank). a is gaussian-init, b zero-init, so
/// training starts exactly at the base function.
pub(crate) struct LoraPair<B: Backend> {
    /// (input, rank), N(0, 0.02) init.
    pub(crate) a: Tensor<B, 2>,
    /// (rank, output), zero init.
    pub(crate) b: Tensor<B, 2>,
}

impl<B: Backend> LoraPair<B> {
    pub(crate) fn new(input: usize, output: usize, rank: usize, device: &B::Device) -> Self {
        let a = Tensor::random([input, rank], Distribution::Normal(0.0, 0.02), device)
            .require_grad();
        let b = Tensor::zeros([rank, output], device).require_grad();
        Self { a, b }
    }

    /// The adapter's additive term for input x (n, input).
    pub(crate) fn contribution(&self, x: &Tensor<B, 2>, scale: f64) -> Tensor<B, 2> {
        x.clone()
            .matmul(self.a.clone())
            .matmul(self.b.clone())
            .mul_scalar(scale)
    }
}

/// Adapter pairs for one decoder layer's five sanctioned targets. k and
/// v stay frozen by the program constraint: KV caches and prefix
/// snapshots must remain adapter-invariant.
pub(crate) struct LayerLora<B: Backend> {
    pub(crate) q: LoraPair<B>,
    pub(crate) o: LoraPair<B>,
    pub(crate) gate: LoraPair<B>,
    pub(crate) up: LoraPair<B>,
    pub(crate) down: LoraPair<B>,
}

/// The whole model's trainable state: one LayerLora per decoder layer,
/// plus the rank/alpha identity that scales every contribution.
pub(crate) struct ModelLora<B: Backend> {
    pub(crate) layers: Vec<LayerLora<B>>,
    pub(crate) rank: usize,
    pub(crate) alpha: f64,
}

impl<B: Backend> ModelLora<B> {
    pub(crate) fn new(config: &HeatConfig, rank: usize, alpha: f64, device: &B::Device) -> Self {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let layers = (0..config.num_hidden_layers)
            .map(|_| LayerLora {
                q: LoraPair::new(hidden, hidden, rank, device),
                o: LoraPair::new(hidden, hidden, rank, device),
                gate: LoraPair::new(hidden, intermediate, rank, device),
                up: LoraPair::new(hidden, intermediate, rank, device),
                down: LoraPair::new(intermediate, hidden, rank, device),
            })
            .collect();
        Self { layers, rank, alpha }
    }

    pub(crate) fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    /// Named mutable access to every trainable tensor - the optimizer
    /// walk. Names are stable artifact keys (layers.<i>.<target>.<a|b>).
    pub(crate) fn params_mut(&mut self) -> Vec<(String, &mut Tensor<B, 2>)> {
        let mut params = Vec::with_capacity(self.layers.len() * 10);
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let targets = [
                ("q", &mut layer.q),
                ("o", &mut layer.o),
                ("gate", &mut layer.gate),
                ("up", &mut layer.up),
                ("down", &mut layer.down),
            ];
            for (target, pair) in targets {
                let LoraPair { a, b } = pair;
                params.push((format!("layers.{index}.{target}.a"), a));
                params.push((format!("layers.{index}.{target}.b"), b));
            }
        }
        params
    }

    /// Immutable named walk (the adapter export).
    pub(crate) fn params(&self) -> Vec<(String, &Tensor<B, 2>)> {
        let mut params = Vec::with_capacity(self.layers.len() * 10);
        for (index, layer) in self.layers.iter().enumerate() {
            let targets = [
                ("q", &layer.q),
                ("o", &layer.o),
                ("gate", &layer.gate),
                ("up", &layer.up),
                ("down", &layer.down),
            ];
            for (target, pair) in targets {
                params.push((format!("layers.{index}.{target}.a"), &pair.a));
                params.push((format!("layers.{index}.{target}.b"), &pair.b));
            }
        }
        params
    }

    /// Persist the adapter as one safetensors file (f32 payloads), with
    /// rank/alpha riding the metadata so a merge/load can validate.
    pub(crate) fn save(&self, path: &Path) -> HeatResult<()> {
        let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for (name, tensor) in self.params() {
            let dims = tensor.dims().to_vec();
            let data = tensor.clone().into_data().convert::<f32>();
            let values = match data.to_vec::<f32>() {
                Ok(values) => values,
                Err(error) => snafu::whatever!("adapter tensor {name} extraction failed: {error:?}"),
            };
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            buffers.push((name, dims, bytes));
        }
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
            .collect::<HeatResult<Vec<_>>>()?;
        let mut metadata = HashMap::new();
        metadata.insert(String::from("lora_rank"), self.rank.to_string());
        metadata.insert(String::from("lora_alpha"), self.alpha.to_string());
        metadata.insert(
            String::from("lora_targets"),
            String::from("q,o,gate,up,down"),
        );
        match safetensors::serialize_to_file(views, Some(metadata), path) {
            Ok(()) => Ok(()),
            Err(error) => snafu::whatever!("adapter serialization failed: {error}"),
        }
    }
}
