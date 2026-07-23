use crate::*;

/// A3 two-stage eviction over full-attention layers; sliding layers are
/// untouched. Stage 1: a running per-head cap during chunked prefill (the
/// peak bound). Stage 2: post-prefill compaction to a tighter decode cap
/// (question-informed - scoring has seen the final chunk by then).
/// Eviction preserves store order, so the sink prefix stays at the first
/// slots and the recent suffix at the last; positions live in the rope'd
/// keys, so attention over a compacted store needs no bookkeeping.
#[derive(Debug, Clone, Copy)]
pub struct EvictionSettings {
    /// Running cap during prefill, entries per head per full layer.
    pub prefill_cap: usize,
    /// Post-prefill compaction target; None keeps the prefill cap.
    pub decode_cap: Option<usize>,
    /// Always-keep: the first `sink_keep` stored entries.
    pub sink_keep: usize,
    /// Always-keep: the last `recent_keep` stored entries.
    pub recent_keep: usize,
}

impl EvictionSettings {
    /// The cap decode-phase overflow maintains.
    pub(crate) fn decode_phase_cap(&self) -> usize {
        self.decode_cap.unwrap_or(self.prefill_cap)
    }

    /// Bounds that make every keep-set well-formed.
    pub(crate) fn validate(&self) -> QuestResult<()> {
        let floor = self.sink_keep + self.recent_keep;
        snafu::ensure_whatever!(
            self.prefill_cap > floor,
            "eviction prefill cap {} leaves no middle budget beyond sink {} + recent {}",
            self.prefill_cap,
            self.sink_keep,
            self.recent_keep
        );
        if let Some(decode_cap) = self.decode_cap {
            snafu::ensure_whatever!(
                decode_cap > floor && decode_cap <= self.prefill_cap,
                "eviction decode cap {} must sit in ({floor}, {}]",
                decode_cap,
                self.prefill_cap
            );
        }
        Ok(())
    }
}

/// Decode-phase overflow slack: eviction fires once the store exceeds the
/// cap by this much, amortizing the compaction over steps.
pub(crate) const DECODE_EVICT_SLACK: usize = 64;

/// Which signal ranks the middle at a compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvictRanking {
    /// Stage 1 (question-blind running cap): cumulative received mass
    /// normalized by each entry's observation count - mean attention per
    /// pass, killing the age bias that starves young entries.
    NormalizedCumulative,
    /// Stage 2 (question-informed compaction) and decode overflow: the
    /// most recent scoring pass alone - the SnapKV observation window
    /// (the final prefill pass covers the question region; decode passes
    /// are the response's own attention).
    LastPass,
}

/// Queries sampled from each prefill chunk's tail for the flash-path
/// re-score (SnapKV's observation window, applied per chunk).
pub(crate) const SCORE_TAIL: usize = 64;

/// Query rows per re-score matmul: bounds the (slice, store) f32
/// softmax transient, which peaks beside full prefill KV at 32K.
pub(crate) const SCORE_TAIL_SLICE: usize = 16;

/// Order-preserving keep-set for one head: the sink prefix and recent
/// suffix are protected; the middle keeps its top-scored entries up to
/// the cap. Returns ascending store indices (len = min(cap, len)).
/// Ties break toward the older (lower) index, deterministically.
pub(crate) fn keep_set(
    scores: &[f32],
    cap: usize,
    sink_keep: usize,
    recent_keep: usize,
) -> Vec<u32> {
    let len = scores.len();
    if len <= cap {
        return (0..len as u32).collect();
    }
    let sink_end = sink_keep.min(len);
    let recent_start = len.saturating_sub(recent_keep).max(sink_end);
    let middle_budget = cap - sink_end - (len - recent_start);

    let mut middle: Vec<u32> = (sink_end as u32..recent_start as u32).collect();
    if middle_budget < middle.len() {
        middle.select_nth_unstable_by(middle_budget.max(1) - 1, |a, b| {
            scores[*b as usize]
                .total_cmp(&scores[*a as usize])
                .then(a.cmp(b))
        });
        middle.truncate(middle_budget);
        middle.sort_unstable();
    }

    let mut keep = Vec::with_capacity(cap.min(len));
    keep.extend(0..sink_end as u32);
    keep.extend(middle);
    keep.extend(recent_start as u32..len as u32);
    keep
}

/// Eager mask values for a chunk attending an evicted store: the store
/// columns are all past (always visible); the trailing q x q block is
/// causal. Row-major 0.0 / -inf, shape (q, store_len + q).
pub(crate) fn evicted_mask_values(q_len: usize, store_len: usize) -> Vec<f32> {
    let total = store_len + q_len;
    let mut values = Vec::with_capacity(q_len * total);
    for row in 0..q_len {
        for column in 0..total {
            let visible = column < store_len || column - store_len <= row;
            values.push(if visible { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    values
}
