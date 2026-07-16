use crate::*;

/// Read a rank-1 u32 tensor from a safetensors file.
pub(crate) fn read_u32(file: &safetensors::SafeTensors, name: &str) -> HeatResult<Vec<u32>> {
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

/// Read a rank-2 f32 tensor: (rows, cols, values).
pub(crate) fn read_f32_matrix(
    file: &safetensors::SafeTensors,
    name: &str,
) -> HeatResult<(usize, usize, Vec<f32>)> {
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

/// Owned little-endian buffers backing the views handed to the safetensors
/// serializer (the serializer borrows).
pub(crate) struct DumpPayload {
    pub(crate) logits_rows: usize,
    pub(crate) vocab: usize,
    pub(crate) logits_bytes: Vec<u8>,
    pub(crate) prompt_bytes: Vec<u8>,
    pub(crate) prompt_len: usize,
    pub(crate) fed_bytes: Vec<u8>,
    pub(crate) fed_len: usize,
}

impl DumpPayload {
    pub(crate) fn new(rows: usize, vocab: usize, logits: &[f32], prompt_ids: &[u32], fed_ids: &[u32]) -> Self {
        Self {
            logits_rows: rows,
            vocab,
            logits_bytes: logits.iter().flat_map(|v| v.to_le_bytes()).collect(),
            prompt_bytes: prompt_ids.iter().flat_map(|v| v.to_le_bytes()).collect(),
            prompt_len: prompt_ids.len(),
            fed_bytes: fed_ids.iter().flat_map(|v| v.to_le_bytes()).collect(),
            fed_len: fed_ids.len(),
        }
    }

    pub(crate) fn write(&self, path: &Path) -> HeatResult<()> {
        let logits_view = match safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![self.logits_rows, self.vocab],
            &self.logits_bytes,
        ) {
            Ok(view) => view,
            Err(error) => snafu::whatever!("logits view failed: {error}"),
        };
        let prompt_view = match safetensors::tensor::TensorView::new(
            safetensors::Dtype::U32,
            vec![self.prompt_len],
            &self.prompt_bytes,
        ) {
            Ok(view) => view,
            Err(error) => snafu::whatever!("prompt view failed: {error}"),
        };
        let fed_view = match safetensors::tensor::TensorView::new(
            safetensors::Dtype::U32,
            vec![self.fed_len],
            &self.fed_bytes,
        ) {
            Ok(view) => view,
            Err(error) => snafu::whatever!("fed view failed: {error}"),
        };
        let tensors = vec![
            (String::from("logits"), logits_view),
            (String::from("prompt_ids"), prompt_view),
            (String::from("fed_ids"), fed_view),
        ];
        match safetensors::serialize_to_file(tensors, None, path) {
            Ok(()) => Ok(()),
            Err(error) => snafu::whatever!("dump serialization failed: {error}"),
        }
    }
}
