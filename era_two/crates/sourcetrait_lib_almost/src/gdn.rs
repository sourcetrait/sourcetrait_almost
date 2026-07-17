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

/// The depthwise causal conv core over an already-prepended history
/// ([seq_len + kernel - 1, C]): the kernel unrolls into kernel-many
/// shifted broadcast multiply-adds - the same values in the same
/// per-element accumulation order as a grouped conv1d, in O(kernel)
/// tensor ops. (candle 0.11's cpu grouped conv chunks the input into
/// `groups` single convs and maps them SERIALLY - at 2880-5760
/// depthwise groups that is ~11.5k convs per GDN layer, a fixed
/// ~60 s per forward; the unroll is milliseconds.)
fn depthwise_causal(
    history: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    seq_len: usize,
) -> LibAlmostResult<candle_core::Tensor> {
    let (_, channels) = history.dims2()?;
    let kernel = weight.dim(2)?;
    let column = |tap: usize| -> LibAlmostResult<candle_core::Tensor> {
        Ok(weight.narrow(2, tap, 1)?.reshape((1, channels))?)
    };
    let mut acc = history.narrow(0, 0, seq_len)?.broadcast_mul(&column(0)?)?;
    for tap in 1..kernel {
        let shifted = history.narrow(0, tap, seq_len)?;
        acc = (acc + shifted.broadcast_mul(&column(tap)?)?)?;
    }
    Ok(acc)
}

/// Depthwise causal conv + silu in the stateless prefill form the
/// pinned upstream uses: kernel-1 zeros of history (equal to the
/// upstream's pad-both-sides-truncate; the LAST weight column
/// multiplies the current token), then silu. x [T, C]; weight
/// [C, 1, kernel].
pub(crate) fn causal_conv_silu(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
) -> LibAlmostResult<candle_core::Tensor> {
    let (seq_len, _) = x.dims2()?;
    let kernel = weight.dim(2)?;
    let history = x.pad_with_zeros(0, kernel - 1, 0)?;
    Ok(depthwise_causal(&history, weight, seq_len)?.silu()?)
}

/// The rule's internal chunk length (the pinned upstream's
/// torch_chunk_gated_delta_rule default; fla's kernels use the same).
pub(crate) const CHUNK: usize = 64;

/// Depthwise causal conv + silu over a chunk with a CARRIED tail: the
/// kernel-1 raw pre-conv rows from the previous chunk prepend x, the
/// conv runs valid (no padding), and the new tail is the last kernel-1
/// raw rows of the concatenation. A zero tail equals the stateless
/// form, so cold starts and chunk chains agree by construction. x
/// [t, C] model dtype; weight [C, 1, kernel]; tail [kernel-1, C]
/// stored f32 (exact for bf16 models). Returns (out [t, C], new tail).
pub(crate) fn conv_with_tail(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    tail: &candle_core::Tensor,
) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
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

/// A [size, size] f32 0/1 mask keeping column <= row (with the
/// diagonal) or column < row (strict).
pub(crate) fn lower_triangle(
    size: usize,
    strict: bool,
    device: &candle_core::Device,
) -> LibAlmostResult<candle_core::Tensor> {
    let mut values = vec![0f32; size * size];
    for (row, chunk) in values.chunks_exact_mut(size).enumerate() {
        let keep = if strict { row } else { row + 1 };
        for value in chunk.iter_mut().take(keep) {
            *value = 1.0;
        }
    }
    Ok(candle_core::Tensor::from_vec(values, (size, size), device)?)
}

/// The chunked gated-delta rule over one contiguous span - the pinned
/// upstream torch_chunk_gated_delta_rule re-expressed in candle ops
/// (chunk 64, pad-to-chunk, in-chunk decay cumsum, the triangular
/// inversion loop, k_cumdecay, per-chunk state advance) with
/// initial-state carry. All f32. q/k/v [t, heads, dk|dv] with q/k
/// l2-normed and q pre-scaled by dk^-0.5 (the same contract as
/// recurrent_step); g/beta [t, heads]; state [heads, dk, dv]. The pad
/// rows are inert (zero k) and dropped from the output. Returns
/// (out [t, heads, dv], final state). Matches the sequential
/// recurrence to f32 rounding (the upstream's own pin: 1.2e-13 nmse).
pub(crate) fn chunk_rule(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    g: &candle_core::Tensor,
    beta: &candle_core::Tensor,
    state: &candle_core::Tensor,
) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
    let (t, heads, head_k) = q.dims3()?;
    let head_v = v.dim(2)?;
    let device = q.device();
    let pad = (CHUNK - t % CHUNK) % CHUNK;
    let chunks = (t + pad) / CHUNK;

    // [t, h, d] -> [h, t, d], zero-padded to the chunk multiple.
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
    // In-chunk cumulative log decay.
    let g = g.reshape((heads, chunks, CHUNK))?.cumsum(2)?;

    let lower_incl = lower_triangle(CHUNK, false, device)?;
    let lower_strict = lower_triangle(CHUNK, true, device)?;
    let eye = candle_core::Tensor::eye(CHUNK, candle_core::DType::F32, device)?;

    // decay_mask[i][j] = exp(g_i - g_j) on the lower triangle, 0 above.
    let diff = g.unsqueeze(3)?.broadcast_sub(&g.unsqueeze(2)?)?;
    let decay_mask = diff
        .broadcast_mul(&lower_incl)?
        .exp()?
        .broadcast_mul(&lower_incl)?;

    // The in-chunk inversion: attn0 = -(k_beta k^T . decay), strict
    // lower; forward substitution row_i += row_i @ rows[..i] runs IN
    // PLACE on one working matrix via slice_set. (The prior Vec+cat
    // shape re-copied every prior row per iteration - ~2k copy
    // kernels per call, measured at 39% of ALL 32K GPU time; this
    // shape is ~5 kernels per row with bit-identical values. Reads
    // materialize via contiguous()/cat before the row write, so no
    // view observes its own mutation.)
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
        // cat on a non-zero dim returns a TRANSPOSED VIEW (the
        // era-one contiguity lesson); slice_set demands contiguity.
        let full_row = candle_core::Tensor::cat(&[&updated, &tail], 3)?.contiguous()?;
        attn.slice_set(&full_row, 2, row_idx)?;
    }
    let attn = attn.broadcast_add(&eye)?;

    let value = attn.matmul(&v_beta)?;
    let g_exp = g.exp()?.unsqueeze(3)?;
    let k_cumdecay = attn.matmul(&k_beta.broadcast_mul(&g_exp)?.contiguous()?)?;

    // Per-chunk sequential state advance.
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
