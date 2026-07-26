//! Stage-2-only KV eviction: the last-pass re-score, the keep-set, the
//! gather.
use crate::*;

/// Default tail queries per prefill-chunk scoring pass.
pub(crate) const SCORE_TAIL: usize = 16;
/// Default query rows per scoring matmul (the peak-transient lever).
pub(crate) const SCORE_SLICE: usize = 16;
/// Decode steps past the cap before an overflow re-compaction epoch.
pub(crate) const OVERFLOW_SLACK: usize = 64;

/// One head's keep-set: sink prefix, recent suffix, top-scored middle.
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

/// Gather each head's keep-set rows into fresh packed storage.
pub(crate) fn gather_rows(
    buffer: &candle_core::Tensor,
    len: usize,
    keep: &[Vec<u32>],
) -> LibQuestResult<candle_core::Tensor> {
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

/// Mean attention mass each store row receives from the tail queries.
pub(crate) fn last_pass_scores(
    q_tail: &candle_core::Tensor,
    k_valid: &candle_core::Tensor,
    scale: f64,
    slice_rows: usize,
) -> LibQuestResult<candle_core::Tensor> {
    let (heads, tail, _) = q_tail.dims3()?;
    let len = k_valid.dim(1)?;
    let device = q_tail.device();
    let k_t = k_valid.transpose(1, 2)?;
    let columns = candle_core::Tensor::arange(0f32, len as f32, device)?
        .reshape((1, len))?;
    let mut total: Option<candle_core::Tensor> = None;
    let mut row = 0usize;
    while row < tail {
        let rows = slice_rows.max(1).min(tail - row);
        let scores = ((q_tail.narrow(1, row, rows)?.matmul(&k_t)? * scale)?)
            .to_dtype(candle_core::DType::F32)?;
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
