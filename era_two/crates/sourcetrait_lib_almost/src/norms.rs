//! RMSNorm variants with the pinned upstream dtype choreography.
//!
//! The order of casts is semantics, not style: variance and normalize
//! run in f32, the weight multiplies AFTER the downcast to the input
//! dtype, and the gated form applies silu(gate) in f32 after the
//! weight (matching torch's promotion in the upstream modeling).
use crate::*;

/// Plain RMSNorm (rms_norm_eps everywhere it appears in this model).
pub(crate) fn rms_norm(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    eps: f64,
) -> LibAlmostResult<candle_core::Tensor> {
    let dtype = x.dtype();
    let last = x.dims().len() - 1;
    let xf = x.to_dtype(candle_core::DType::F32)?;
    let variance = xf.sqr()?.mean_keepdim(last)?;
    let normed = xf.broadcast_div(&(variance + eps)?.sqrt()?)?;
    Ok(weight.broadcast_mul(&normed.to_dtype(dtype)?)?)
}

/// Gated RMSNorm over the GDN value-head dim: norm-before-gate, weight
/// in input dtype, gate = silu(gate) in f32, result back in input
/// dtype. Eps here is 1e-5 (the fla FusedRMSNormGated default), NOT
/// rms_norm_eps - the one eps exception in the model.
pub(crate) fn rms_norm_gated(
    x: &candle_core::Tensor,
    gate: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    eps: f64,
) -> LibAlmostResult<candle_core::Tensor> {
    let dtype = x.dtype();
    let last = x.dims().len() - 1;
    let xf = x.to_dtype(candle_core::DType::F32)?;
    let variance = xf.sqr()?.mean_keepdim(last)?;
    let normed = xf.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let weighted = weight.broadcast_mul(&normed.to_dtype(dtype)?)?;
    let gate_f32 = gate.to_dtype(candle_core::DType::F32)?;
    let silu_gate = (&gate_f32 * candle_nn::ops::sigmoid(&gate_f32)?)?;
    Ok(weighted
        .to_dtype(candle_core::DType::F32)?
        .mul(&silu_gate)?
        .to_dtype(dtype)?)
}
