//! Gated-DeltaNet building blocks: gating scalars, the fla-style l2
//! norm, the stateless causal conv, and the per-token recurrence step.
use crate::*;

/// The gated output norm's eps: 1e-5, the fla FusedRMSNormGated
/// default the pinned upstream hardcodes - NOT rms_norm_eps.
pub(crate) const O_NORM_EPS: f64 = 1e-5;

/// The per-token gating scalars, computed in f32 exactly as the pinned
/// upstream does: g = -exp(A_log) * softplus(a + dt_bias) (log-space
/// decay), beta = sigmoid(b), doubled under the negative-eigenvalue
/// variant (this checkpoint's posture).
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
pub(crate) fn softplus(x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
    let smooth = ((x.exp()? + 1.0)?).log()?;
    let guard = x.ge(20.0f64)?;
    Ok(guard.where_cond(x, &smooth)?)
}

/// The fla-style l2 normalization over the last dim:
/// x / sqrt(sum(x^2) + eps), eps 1e-6. A SUM under the root (not
/// RMSNorm's mean); applied to the q and k head vectors before the
/// recurrence.
pub(crate) fn l2_norm(x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
    let last = x.dims().len() - 1;
    let norm = ((x.sqr()?.sum_keepdim(last)?) + 1e-6)?.sqrt()?;
    Ok(x.broadcast_div(&norm)?)
}

/// Depthwise causal conv + silu in the stateless prefill form the
/// pinned upstream uses: pad kernel-1 both sides, keep the first T
/// outputs (equal to a left-only causal pad - the LAST weight column
/// multiplies the current token), then silu. x [T, C]; weight
/// [C, 1, kernel].
pub(crate) fn causal_conv_silu(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
) -> LibAlmostResult<candle_core::Tensor> {
    let (seq_len, channels) = x.dims2()?;
    let kernel = weight.dim(2)?;
    let conv_in = x.t()?.unsqueeze(0)?.contiguous()?;
    let out = conv_in.conv1d(weight, kernel - 1, 1, 1, channels)?;
    let out = out.narrow(2, 0, seq_len)?.squeeze(0)?.t()?.contiguous()?;
    Ok(out.silu()?)
}

/// One gated-delta recurrence step in the pinned upstream order: decay
/// the state BEFORE the delta correction. All f32. state [heads, dk,
/// dv]; q_t/k_t [heads, dk] (l2-normed, q pre-scaled by dk^-0.5); v_t
/// [heads, dv]; decay_t = exp(g_t) and beta_t [heads]. Returns the
/// advanced state and the step's output row [heads, dv].
pub(crate) fn recurrent_step(
    state: &candle_core::Tensor,
    q_t: &candle_core::Tensor,
    k_t: &candle_core::Tensor,
    v_t: &candle_core::Tensor,
    decay_t: &candle_core::Tensor,
    beta_t: &candle_core::Tensor,
) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
    let decayed = state.broadcast_mul(&decay_t.unsqueeze(1)?.unsqueeze(2)?)?;
    let kv_mem = decayed.broadcast_mul(&k_t.unsqueeze(2)?)?.sum(1)?;
    let delta = v_t.sub(&kv_mem)?.broadcast_mul(&beta_t.unsqueeze(1)?)?;
    let state = (decayed + k_t.unsqueeze(2)?.broadcast_mul(&delta.unsqueeze(1)?)?)?;
    let y_t = state.broadcast_mul(&q_t.unsqueeze(2)?)?.sum(1)?;
    Ok((state, y_t))
}
