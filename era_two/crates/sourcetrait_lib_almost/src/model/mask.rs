//! Additive causal masks for the prefill paths (decode is mask-free).
use crate::*;

/// Additive causal mask [T, T]: 0 on and below the diagonal, -inf
/// above (exp saturates to exactly 0 either way upstream renders it).
pub(crate) fn causal_mask(
    seq_len: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> LibAlmostResult<candle_core::Tensor> {
    offset_causal_mask(seq_len, 0, dtype, device)
}

/// Additive causal mask for a chunk at an offset: [t, past + t], row
/// r sees every column through past + r, -inf beyond.
pub(crate) fn offset_causal_mask(
    seq_len: usize,
    past: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> LibAlmostResult<candle_core::Tensor> {
    let width = past + seq_len;
    let mut values = vec![0f32; seq_len * width];
    for (row, chunk) in values.chunks_exact_mut(width).enumerate() {
        for value in chunk.iter_mut().skip(past + row + 1) {
            *value = f32::NEG_INFINITY;
        }
    }
    Ok(
        candle_core::Tensor::from_vec(values, (seq_len, width), device)?
            .to_dtype(dtype)?,
    )
}
