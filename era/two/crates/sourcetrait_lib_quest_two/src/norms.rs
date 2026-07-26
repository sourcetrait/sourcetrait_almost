//! RMSNorm variants with the pinned upstream dtype choreography.
use crate::*;

/// Plain RMSNorm (rms_norm_eps everywhere it appears in this model).
pub(crate) fn rms_norm(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    eps: f64,
) -> LibQuestResult<candle_core::Tensor> {
    let dtype = x.dtype();
    let last = x.dims().len() - 1;
    let xf = x.to_dtype(candle_core::DType::F32)?;
    let variance = xf.sqr()?.mean_keepdim(last)?;
    let normed = xf.broadcast_div(&(variance + eps)?.sqrt()?)?;
    Ok(weight.broadcast_mul(&normed.to_dtype(dtype)?)?)
}

/// rms_norm, dispatching to the fused kernel where it is eligible.
pub(crate) fn rms_norm_auto(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    eps: f64,
    fused: bool,
) -> LibQuestResult<candle_core::Tensor> {
    #[cfg(feature = "cuda")]
    if fused
        && x.dims().len() == 2
        && x.is_contiguous()
        && x.device().is_cuda()
        && x.dtype() == candle_core::DType::BF16
    {
        let out = candle_core::Tensor::zeros(
            x.dims(),
            candle_core::DType::BF16,
            x.device(),
        )?;
        out.inplace_op3(x, weight, &fused::RmsNormFused { eps: eps as f32 })?;
        return Ok(out);
    }
    #[cfg(not(feature = "cuda"))]
    let _ = fused;
    rms_norm(x, weight, eps)
}

/// Gated RMSNorm over the GDN value-head dim: norm before gate.
pub(crate) fn rms_norm_gated(
    x: &candle_core::Tensor,
    gate: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    eps: f64,
) -> LibQuestResult<candle_core::Tensor> {
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
