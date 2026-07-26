//! The burn oracle's own config serde, weight loader and dump readers.
#![allow(dead_code)]
use crate::*;

/// The oracle's own read of config.json, independent of lib's.
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct HybridCheckpointConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) layer_types: Vec<String>,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
}

impl HybridCheckpointConfig {
    pub(crate) fn load(model_dir: &Path) -> BquestResult<Self> {
        let text = fs::read(model_dir.join("config.json"))?;
        let config: Self = serde_json::from_slice(&text)?;
        snafu::ensure_whatever!(
            config.layer_types.len() == config.num_hidden_layers,
            "layer_types carries {} entries for {} layers",
            config.layer_types.len(),
            config.num_hidden_layers
        );
        snafu::ensure_whatever!(
            config.linear_num_key_heads == config.linear_num_value_heads,
            "unequal GDN head counts ({} key vs {} value)",
            config.linear_num_key_heads,
            config.linear_num_value_heads
        );
        Ok(config)
    }

    pub(crate) fn is_gdn_layer(&self, index: usize) -> BquestResult<bool> {
        match self.layer_types[index].as_str() {
            "linear_attention" => Ok(true),
            "full_attention" => Ok(false),
            other => snafu::whatever!("unknown layer type {other} at layer {index}"),
        }
    }
}

/// Checkpoint tensors as host f32, consumed by safetensors name.
pub(crate) struct HybridWeights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl HybridWeights {
    /// Assemble from pre-built host tensors: the toy builders' entry.
    pub(crate) fn from_tensors(tensors: HashMap<String, (Vec<usize>, Vec<f32>)>) -> Self {
        Self { tensors }
    }

    /// Read the single shard, converting bf16 payloads to f32.
    pub(crate) fn load(model_dir: &Path) -> BquestResult<Self> {
        let bytes = fs::read(model_dir.join("model.safetensors"))?;
        let parsed = match safetensors::SafeTensors::deserialize(&bytes) {
            Ok(parsed) => parsed,
            Err(e) => snafu::whatever!("shard parse failed: {e}"),
        };
        let mut tensors = HashMap::new();
        for (name, view) in parsed.tensors() {
            let shape = view.shape().to_vec();
            let data = view.data();
            let values: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|pair| half::bf16::from_le_bytes([pair[0], pair[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
                    .collect(),
                other => snafu::whatever!("unsupported weight dtype {other:?} for {name}"),
            };
            tensors.insert(name, (shape, values));
        }
        Ok(Self { tensors })
    }

    pub(crate) fn take(&mut self, name: &str) -> BquestResult<(Vec<usize>, Vec<f32>)> {
        let Some(entry) = self.tensors.remove(name) else {
            snafu::whatever!("checkpoint tensor {name} is missing");
        };
        Ok(entry)
    }

    /// A pytorch Linear weight (out, in), transposed to (in, out).
    pub(crate) fn take_linear_transposed<B: burn::tensor::backend::Backend>(
        &mut self,
        name: &str,
        device: &B::Device,
    ) -> BquestResult<burn::tensor::Tensor<B, 2>> {
        let (shape, values) = self.take(name)?;
        snafu::ensure_whatever!(shape.len() == 2, "{name}: expected rank 2, got {shape:?}");
        let (rows, cols) = (shape[0], shape[1]);
        let mut transposed = vec![0f32; values.len()];
        for row in 0..rows {
            for col in 0..cols {
                transposed[col * rows + row] = values[row * cols + col];
            }
        }
        let data = burn::tensor::TensorData::new(transposed, [cols, rows]);
        Ok(burn::tensor::Tensor::from_data(data, device))
    }

    pub(crate) fn take_vector<B: burn::tensor::backend::Backend>(
        &mut self,
        name: &str,
        device: &B::Device,
    ) -> BquestResult<burn::tensor::Tensor<B, 1>> {
        let (shape, values) = self.take(name)?;
        snafu::ensure_whatever!(shape.len() == 1, "{name}: expected rank 1, got {shape:?}");
        let length = shape[0];
        let data = burn::tensor::TensorData::new(values, [length]);
        Ok(burn::tensor::Tensor::from_data(data, device))
    }

    /// A rank-1 tensor as a broadcastable (1, n) row.
    pub(crate) fn take_row<B: burn::tensor::backend::Backend>(
        &mut self,
        name: &str,
        device: &B::Device,
    ) -> BquestResult<burn::tensor::Tensor<B, 2>> {
        let (shape, values) = self.take(name)?;
        snafu::ensure_whatever!(shape.len() == 1, "{name}: expected rank 1, got {shape:?}");
        let length = shape[0];
        let data = burn::tensor::TensorData::new(values, [1, length]);
        Ok(burn::tensor::Tensor::from_data(data, device))
    }

    /// A depthwise conv weight as broadcastable tap rows, tap-major.
    pub(crate) fn take_conv_taps<B: burn::tensor::backend::Backend>(
        &mut self,
        name: &str,
        kernel: usize,
        device: &B::Device,
    ) -> BquestResult<Vec<burn::tensor::Tensor<B, 2>>> {
        let (shape, values) = self.take(name)?;
        snafu::ensure_whatever!(
            shape.len() == 3 && shape[1] == 1 && shape[2] == kernel,
            "{name}: expected (channels, 1, {kernel}), got {shape:?}"
        );
        let channels = shape[0];
        let mut taps = Vec::with_capacity(kernel);
        for tap in 0..kernel {
            let mut row = vec![0f32; channels];
            for (channel, slot) in row.iter_mut().enumerate() {
                *slot = values[channel * kernel + tap];
            }
            let data = burn::tensor::TensorData::new(row, [1, channels]);
            taps.push(burn::tensor::Tensor::from_data(data, device));
        }
        Ok(taps)
    }

    /// Raw host values of a rank-2 tensor, as (rows, cols, values).
    pub(crate) fn take_host_matrix(
        &mut self,
        name: &str,
    ) -> BquestResult<(usize, usize, Vec<f32>)> {
        let (shape, values) = self.take(name)?;
        snafu::ensure_whatever!(shape.len() == 2, "{name}: expected rank 2, got {shape:?}");
        Ok((shape[0], shape[1], values))
    }
}

/// Read a rank-1 u32 tensor from a reference dump.
pub(crate) fn dump_read_u32(
    file: &safetensors::SafeTensors,
    name: &str,
) -> BquestResult<Vec<u32>> {
    let Ok(view) = file.tensor(name) else {
        snafu::whatever!("dump is missing tensor {name}");
    };
    snafu::ensure_whatever!(
        view.dtype() == safetensors::Dtype::U32,
        "{name}: expected u32, got {:?}",
        view.dtype()
    );
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|quad| u32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
        .collect())
}

/// Read a rank-2 f32 dump tensor as (rows, cols, values).
pub(crate) fn dump_read_f32_matrix(
    file: &safetensors::SafeTensors,
    name: &str,
) -> BquestResult<(usize, usize, Vec<f32>)> {
    let Ok(view) = file.tensor(name) else {
        snafu::whatever!("dump is missing tensor {name}");
    };
    snafu::ensure_whatever!(
        view.dtype() == safetensors::Dtype::F32,
        "{name}: expected f32, got {:?}",
        view.dtype()
    );
    let shape = view.shape();
    snafu::ensure_whatever!(shape.len() == 2, "{name}: expected rank 2, got {shape:?}");
    let values = view
        .data()
        .chunks_exact(4)
        .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
        .collect();
    Ok((shape[0], shape[1], values))
}
