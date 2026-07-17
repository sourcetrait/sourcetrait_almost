//! A3 stage-2-only KV eviction primitives: the SnapKV-style last-pass
//! re-score (chunk-tail queries against the whole store, row-sliced),
//! the order-preserving keep-set, and the per-head row gather.
//!
//! Stage-2-only: scores are the LAST scoring pass alone - each prefill
//! chunk overwrites them from its tail queries (by the final chunk
//! that is the question window), and each decode row overwrites them
//! again (response-conditioned). Compaction keeps a protected sink
//! prefix + recent suffix and the top-ranked middle, order-preserving
//! per head. NoPE makes eviction position-free: nothing but the causal
//! structure of PREFILL depends on store order, and compaction only
//! ever runs post-prefill or between decode steps.
use crate::*;

/// Tail queries per prefill-chunk scoring pass (the observation
/// window) - the DEFAULT; the live value rides
/// EvictionSettings.score_tail (the RescoreTuning prototyping knob).
pub(crate) const SCORE_TAIL: usize = 64;
/// Query rows per scoring matmul (the peak-transient lever - a
/// full-width f32 chain beside prefill KV breached era-one's peak
/// contract) - the DEFAULT; the live value rides
/// EvictionSettings.score_slice.
pub(crate) const SCORE_SLICE: usize = 16;
/// Decode steps past the cap before an overflow re-compaction epoch.
pub(crate) const OVERFLOW_SLACK: usize = 64;

/// The order-preserving keep-set for one head: protected sink prefix
/// [0, sink) + protected recent suffix [len-recent, len) + the
/// top-scored middle rows up to `cap` total. Ties break toward the
/// OLDER row (lower index) so the set is deterministic. Indices come
/// back ascending (order-preserving compaction).
pub(crate) fn keep_indices(
    scores: &[f32],
    cap: usize,
    recent: usize,
    sink: usize,
) -> Vec<u32> {
    let len = scores.len();
    if len <= cap {
        return (0..len as u32).collect();
    }
    let sink_end = sink.min(len);
    let recent_start = len.saturating_sub(recent).max(sink_end);
    let budget = cap.saturating_sub(sink_end + (len - recent_start));
    let mut middle: Vec<u32> = (sink_end as u32..recent_start as u32).collect();
    middle.sort_by(|a, b| {
        scores[*b as usize]
            .total_cmp(&scores[*a as usize])
            .then(a.cmp(b))
    });
    middle.truncate(budget);
    let mut keep: Vec<u32> = (0..sink_end as u32)
        .chain(middle)
        .chain(recent_start as u32..len as u32)
        .collect();
    keep.sort_unstable();
    keep
}

/// Gather `keep`-indexed rows per head from a [heads, capacity, dim]
/// buffer's first `len` rows; one keep-set per head, all `cap` long.
/// Returns the packed [heads, cap, dim] gather (fresh storage - safe
/// to slice_set back into the source buffer).
pub(crate) fn gather_rows(
    buffer: &candle_core::Tensor,
    len: usize,
    keep: &[Vec<u32>],
) -> LibAlmostResult<candle_core::Tensor> {
    let heads = buffer.dim(0)?;
    snafu::ensure_whatever!(
        keep.len() == heads,
        "gather_rows wants one keep-set per head ({heads}), got {}",
        keep.len()
    );
    let device = buffer.device();
    let mut gathered = Vec::with_capacity(heads);
    for (head, keep_set) in keep.iter().enumerate() {
        let indices =
            candle_core::Tensor::from_vec(keep_set.clone(), keep_set.len(), device)?;
        gathered.push(
            buffer
                .narrow(0, head, 1)?
                .narrow(1, 0, len)?
                .index_select(&indices, 1)?,
        );
    }
    Ok(candle_core::Tensor::cat(&gathered, 0)?)
}

/// The last-pass scores: mean attention mass each store row receives
/// from `tail` query rows (the last rows of a prefill chunk, causally
/// masked to their own positions), computed f32 in `slice_rows`-row
/// slices. q_tail [heads, tail, dim] whose row j sits at absolute
/// position len - tail + j; k_valid [heads, len, dim]. Returns
/// [heads, len, 1] f32, slice_set-ready against a score buffer.
pub(crate) fn last_pass_scores(
    q_tail: &candle_core::Tensor,
    k_valid: &candle_core::Tensor,
    scale: f64,
    slice_rows: usize,
) -> LibAlmostResult<candle_core::Tensor> {
    let (heads, tail, _) = q_tail.dims3()?;
    let len = k_valid.dim(1)?;
    let device = q_tail.device();
    // The matmul rides the model dtype over a transpose VIEW (the
    // eager-attention shape - no store-sized copy or upcast); only
    // the sliced softmax runs f32.
    let k_t = k_valid.transpose(1, 2)?;
    let columns = candle_core::Tensor::arange(0f32, len as f32, device)?
        .reshape((1, len))?;
    let mut total: Option<candle_core::Tensor> = None;
    let mut row = 0usize;
    while row < tail {
        let rows = slice_rows.max(1).min(tail - row);
        let scores = ((q_tail.narrow(1, row, rows)?.matmul(&k_t)? * scale)?)
            .to_dtype(candle_core::DType::F32)?;
        // Causal tail mask, arithmetic form: rows at absolute position
        // len - tail + row + j see columns <= that position; blocked
        // columns ride a -1e30 additive term (underflows to exactly
        // zero mass through softmax).
        let positions: Vec<f32> = (0..rows)
            .map(|j| (len - tail + row + j) as f32)
            .collect();
        let positions = candle_core::Tensor::from_vec(positions, (rows, 1), device)?;
        let allowed = (positions.broadcast_sub(&columns)? + 1.0)?.clamp(0f64, 1f64)?;
        let blocked = ((allowed.clone() - 1.0)? * 1e30)?;
        let masked = scores
            .broadcast_mul(&allowed)?
            .broadcast_add(&blocked)?;
        let probs = candle_nn::ops::softmax_last_dim(&masked)?;
        let summed = probs.sum(1)?;
        total = Some(match total {
            Some(total) => (total + summed)?,
            None => summed,
        });
        row += rows;
    }
    let total = total.expect("tail is non-empty");
    Ok(((total / tail as f64)?).reshape((heads, len, 1))?)
}
