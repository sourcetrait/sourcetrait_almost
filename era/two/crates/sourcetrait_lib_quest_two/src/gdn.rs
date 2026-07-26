//! Gated-DeltaNet building blocks: gating scalars, the fla-style l2
//! norm, the stateless causal conv, and the per-token recurrence step.
use crate::*;

/// The gated output norm's eps: 1e-5, and NOT rms_norm_eps.
pub(crate) const O_NORM_EPS: f64 = 1e-5;

/// The per-token gating scalars in f32; decay stays in log space.
pub(crate) fn gdn_gates(
    a: &candle_core::Tensor,
    b: &candle_core::Tensor,
    a_log: &candle_core::Tensor,
    dt_bias: &candle_core::Tensor,
    allow_neg_eigval: bool,
) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor)> {
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

/// softplus with the overflow guard: x when x > 20, ln(1 + e^x) below.
pub(crate) fn softplus(x: &candle_core::Tensor) -> LibQuestResult<candle_core::Tensor> {
    let smooth = ((x.exp()? + 1.0)?).log()?;
    let guard = x.ge(20.0f64)?;
    Ok(guard.where_cond(x, &smooth)?)
}

/// l2 normalization over the last dim: a SUM under the root, not a mean.
pub(crate) fn l2_norm(x: &candle_core::Tensor) -> LibQuestResult<candle_core::Tensor> {
    let last = x.dims().len() - 1;
    let norm = ((x.sqr()?.sum_keepdim(last)?) + 1e-6)?.sqrt()?;
    Ok(x.broadcast_div(&norm)?)
}

/// The depthwise causal conv core, as kernel-many shifted multiply-adds.
fn depthwise_causal(
    history: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    seq_len: usize,
) -> LibQuestResult<candle_core::Tensor> {
    let (_, channels) = history.dims2()?;
    let kernel = weight.dim(2)?;
    let column = |tap: usize| -> LibQuestResult<candle_core::Tensor> {
        Ok(weight.narrow(2, tap, 1)?.reshape((1, channels))?)
    };
    let mut acc = history.narrow(0, 0, seq_len)?.broadcast_mul(&column(0)?)?;
    for tap in 1..kernel {
        let shifted = history.narrow(0, tap, seq_len)?;
        acc = (acc + shifted.broadcast_mul(&column(tap)?)?)?;
    }
    Ok(acc)
}

/// Depthwise causal conv + silu, stateless: kernel-1 zeros of history.
pub(crate) fn causal_conv_silu(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
) -> LibQuestResult<candle_core::Tensor> {
    let (seq_len, _) = x.dims2()?;
    let kernel = weight.dim(2)?;
    let history = x.pad_with_zeros(0, kernel - 1, 0)?;
    Ok(depthwise_causal(&history, weight, seq_len)?.silu()?)
}

/// The rule's internal chunk length, as the pinned upstream sets it.
pub(crate) const CHUNK: usize = 64;

/// The same conv over a chunk with a CARRIED tail, returning both.
pub(crate) fn conv_with_tail(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    tail: &candle_core::Tensor,
) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor)> {
    let (seq_len, _) = x.dims2()?;
    let kernel = weight.dim(2)?;
    let tail_cast = tail.to_dtype(x.dtype())?;
    let history = candle_core::Tensor::cat(&[&tail_cast, x], 0)?;
    let total = history.dim(0)?;
    let new_tail = history
        .narrow(0, total - (kernel - 1), kernel - 1)?
        .to_dtype(candle_core::DType::F32)?;
    let out = depthwise_causal(&history, weight, seq_len)?.silu()?;
    Ok((out, new_tail))
}

/// One gated-delta step: decay the state BEFORE the delta correction.
pub(crate) fn recurrent_step(
    state: &candle_core::Tensor,
    q_t: &candle_core::Tensor,
    k_t: &candle_core::Tensor,
    v_t: &candle_core::Tensor,
    decay_t: &candle_core::Tensor,
    beta_t: &candle_core::Tensor,
) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor)> {
    let decayed = state.broadcast_mul(&decay_t.unsqueeze(1)?.unsqueeze(2)?)?;
    let kv_mem = decayed.broadcast_mul(&k_t.unsqueeze(2)?)?.sum(1)?;
    let delta = v_t.sub(&kv_mem)?.broadcast_mul(&beta_t.unsqueeze(1)?)?;
    let state = (decayed + k_t.unsqueeze(2)?.broadcast_mul(&delta.unsqueeze(1)?)?)?;
    let y_t = state.broadcast_mul(&q_t.unsqueeze(2)?)?.sum(1)?;
    Ok((state, y_t))
}

/// A [size, size] f32 0/1 lower-triangular mask, strict or inclusive.
pub(crate) fn lower_triangle(
    size: usize,
    strict: bool,
    device: &candle_core::Device,
) -> LibQuestResult<candle_core::Tensor> {
    let mut values = vec![0f32; size * size];
    for (row, chunk) in values.chunks_exact_mut(size).enumerate() {
        let keep = if strict { row } else { row + 1 };
        for value in chunk.iter_mut().take(keep) {
            *value = 1.0;
        }
    }
    Ok(candle_core::Tensor::from_vec(values, (size, size), device)?)
}

/// The chunked gated-delta rule over one span, with initial-state carry.
pub(crate) fn chunk_rule(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    g: &candle_core::Tensor,
    beta: &candle_core::Tensor,
    state: &candle_core::Tensor,
) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor)> {
    let (t, heads, head_k) = q.dims3()?;
    let head_v = v.dim(2)?;
    let device = q.device();
    let pad = (CHUNK - t % CHUNK) % CHUNK;
    let chunks = (t + pad) / CHUNK;

    let q = q.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;
    let k = k.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;
    let v = v.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;
    let g = g.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;
    let beta = beta.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;

    let v_beta = v.broadcast_mul(&beta.unsqueeze(2)?)?;
    let k_beta = k.broadcast_mul(&beta.unsqueeze(2)?)?;

    let q = q.reshape((heads, chunks, CHUNK, head_k))?;
    let k = k.reshape((heads, chunks, CHUNK, head_k))?;
    let v_beta = v_beta.reshape((heads, chunks, CHUNK, head_v))?;
    let k_beta = k_beta.reshape((heads, chunks, CHUNK, head_k))?;
    let g = g.reshape((heads, chunks, CHUNK))?.cumsum(2)?;

    let lower_incl = lower_triangle(CHUNK, false, device)?;
    let lower_strict = lower_triangle(CHUNK, true, device)?;
    let eye = candle_core::Tensor::eye(CHUNK, candle_core::DType::F32, device)?;

    let diff = g.unsqueeze(3)?.broadcast_sub(&g.unsqueeze(2)?)?;
    let decay_mask = diff
        .broadcast_mul(&lower_incl)?
        .exp()?
        .broadcast_mul(&lower_incl)?;

    let attn = k_beta
        .matmul(&k.transpose(2, 3)?.contiguous()?)?
        .mul(&decay_mask)?
        .broadcast_mul(&lower_strict)?
        .neg()?;
    for row_idx in 1..CHUNK {
        let row = attn.narrow(2, row_idx, 1)?;
        let row_prefix = row.narrow(3, 0, row_idx)?.contiguous()?;
        let sub = attn
            .narrow(2, 0, row_idx)?
            .narrow(3, 0, row_idx)?
            .contiguous()?;
        let updated = (&row_prefix + row_prefix.matmul(&sub)?)?;
        let tail = row.narrow(3, row_idx, CHUNK - row_idx)?;
        let full_row = candle_core::Tensor::cat(&[&updated, &tail], 3)?.contiguous()?;
        attn.slice_set(&full_row, 2, row_idx)?;
    }
    let attn = attn.broadcast_add(&eye)?;

    let value = attn.matmul(&v_beta)?;
    let g_exp = g.exp()?.unsqueeze(3)?;
    let k_cumdecay = attn.matmul(&k_beta.broadcast_mul(&g_exp)?.contiguous()?)?;

    let mut carried = state.clone();
    let mut outs = Vec::with_capacity(chunks);
    for chunk_idx in 0..chunks {
        let q_i = q.narrow(1, chunk_idx, 1)?.squeeze(1)?.contiguous()?;
        let k_i = k.narrow(1, chunk_idx, 1)?.squeeze(1)?.contiguous()?;
        let v_i = value.narrow(1, chunk_idx, 1)?.squeeze(1)?;
        let g_i = g.narrow(1, chunk_idx, 1)?.squeeze(1)?;
        let mask_i = decay_mask.narrow(1, chunk_idx, 1)?.squeeze(1)?;
        let attn_local = q_i
            .matmul(&k_i.transpose(1, 2)?.contiguous()?)?
            .mul(&mask_i)?
            .broadcast_mul(&lower_incl)?;
        let v_prime = k_cumdecay
            .narrow(1, chunk_idx, 1)?
            .squeeze(1)?
            .contiguous()?
            .matmul(&carried)?;
        let v_new = v_i.sub(&v_prime)?;
        let g_exp_i = g_i.exp()?.unsqueeze(2)?;
        let attn_inter = q_i.broadcast_mul(&g_exp_i)?.matmul(&carried)?;
        outs.push((attn_inter + attn_local.matmul(&v_new)?)?);
        let g_last = g_i.narrow(1, CHUNK - 1, 1)?;
        let state_decay = g_last.exp()?.unsqueeze(2)?;
        let k_scale = g_last.broadcast_sub(&g_i)?.exp()?.unsqueeze(2)?;
        carried = (carried.broadcast_mul(&state_decay)?
            + k_i
                .broadcast_mul(&k_scale)?
                .transpose(1, 2)?
                .contiguous()?
                .matmul(&v_new)?)?;
    }
    let out = candle_core::Tensor::cat(&outs, 1)?
        .narrow(1, 0, t)?
        .transpose(0, 1)?
        .contiguous()?;
    Ok((out, carried))
}
