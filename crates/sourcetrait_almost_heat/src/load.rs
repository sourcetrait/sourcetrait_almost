use crate::*;

/// All checkpoint tensors as host f32 buffers, keyed by their safetensors
/// names; consumed (removed) as the model takes them.
pub(crate) struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    /// Read every shard, converting bf16 payloads to f32.
    pub(crate) fn load(shards: &[PathBuf]) -> HeatResult<Self> {
        let mut tensors = HashMap::new();
        for shard in shards {
            let bytes = std::fs::read(shard)?;
            let parsed = safetensors::SafeTensors::deserialize(&bytes)?;
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
        }
        Ok(Self { tensors })
    }

    /// Remove and return a named tensor's shape + values.
    pub(crate) fn take(&mut self, name: &str) -> HeatResult<(Vec<usize>, Vec<f32>)> {
        let Some(entry) = self.tensors.remove(name) else {
            snafu::whatever!("checkpoint tensor {name} is missing");
        };
        Ok(entry)
    }

    /// A pytorch Linear weight (out, in), pre-transposed to (in, out) so the
    /// forward pass is a plain x.matmul(w).
    pub(crate) fn take_linear_transposed<B: burn::tensor::backend::Backend>(
        &mut self,
        name: &str,
        device: &B::Device,
    ) -> HeatResult<burn::tensor::Tensor<B, 2>> {
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

    /// A rank-1 weight vector (norm scales).
    pub(crate) fn take_vector<B: burn::tensor::backend::Backend>(
        &mut self,
        name: &str,
        device: &B::Device,
    ) -> HeatResult<burn::tensor::Tensor<B, 1>> {
        let (shape, values) = self.take(name)?;
        snafu::ensure_whatever!(shape.len() == 1, "{name}: expected rank 1, got {shape:?}");
        let length = shape[0];
        let data = burn::tensor::TensorData::new(values, [length]);
        Ok(burn::tensor::Tensor::from_data(data, device))
    }

    /// Raw host values of a rank-2 tensor (the embedding table stays on the
    /// host for direct row gathering).
    pub(crate) fn take_host_matrix(&mut self, name: &str) -> HeatResult<(usize, usize, Vec<f32>)> {
        let (shape, values) = self.take(name)?;
        snafu::ensure_whatever!(shape.len() == 2, "{name}: expected rank 2, got {shape:?}");
        Ok((shape[0], shape[1], values))
    }
}
