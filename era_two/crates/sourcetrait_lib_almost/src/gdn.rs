//! Gated-DeltaNet building blocks. D1 carries the gating scalars; the
//! recurrence and conv machinery land with D2.
use crate::*;

/// The per-token gating scalars, computed in f32 exactly as the pinned
/// upstream does: g = -exp(A_log) * softplus(a + dt_bias) (log-space
/// decay), beta = sigmoid(b), doubled under the negative-eigenvalue
/// variant (this checkpoint's posture).
#[allow(dead_code)]
pub(crate) fn gdn_gates(
    a: &candle_core::Tensor,
    b: &candle_core::Tensor,
    a_log: &candle_core::Tensor,
    dt_bias: &candle_core::Tensor,
    allow_neg_eigval: bool,
) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
    let af = a.to_dtype(candle_core::DType::F32)?;
    let a_logf = a_log.to_dtype(candle_core::DType::F32)?;
    let biasf = dt_bias.to_dtype(candle_core::DType::F32)?;
    let x = af.broadcast_add(&biasf)?;
    let g = softplus(&x)?
        .broadcast_mul(&a_logf.exp()?)?
        .neg()?;
    let bf = b.to_dtype(candle_core::DType::F32)?;
    let mut beta = candle_nn::ops::sigmoid(&bf)?;
    if allow_neg_eigval {
        beta = (beta * 2.0)?;
    }
    Ok((g, beta))
}

/// softplus with the overflow guard: x when x > 20, ln(1 + e^x) below
/// (the same threshold the upstream fused gating kernel uses).
#[allow(dead_code)]
pub(crate) fn softplus(x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
    let smooth = ((x.exp()? + 1.0)?).log()?;
    let guard = x.ge(20.0f64)?;
    Ok(guard.where_cond(x, &smooth)?)
}
