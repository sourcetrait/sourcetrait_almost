//! The A2 observation instrument (attn-profile feature): per-head
//! attention-mass accumulation on the profile-armed attention path.
//!
//! Masses per query row at absolute position p, eligible when
//! p >= window + 1 (so a middle exists): sink = column 0, recent = the
//! trailing `window` columns [p-window+1, p], middle = everything
//! between - a 3-way partition of each eligible row's unit mass
//! (columns past p carry exactly zero softmax mass and fold into
//! recent harmlessly). The armed path computes attention in row slices
//! so the f32 softmax transient stays bounded at 32K; slicing is
//! value-identical (row-independent ops).
use crate::*;

/// Rows per profiled attention slice (the sliced-transient lever).
pub(crate) const PROFILE_ROW_SLICE: usize = 64;

/// One head's mean masses over the accumulated rows.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HeadMasses {
    pub middle: f64,
    pub sink: f64,
    pub recent: f64,
}

/// One attention layer's profile: per-head prefill and decode means
/// plus the eligible-row counts behind them.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerProfile {
    pub layer_index: usize,
    pub window: usize,
    pub prefill_rows: u64,
    pub decode_rows: u64,
    pub prefill: Vec<HeadMasses>,
    pub decode: Vec<HeadMasses>,
}

/// Per-layer accumulator; lives on the layer while armed (interior
/// mutability - attention runs under &self).
#[derive(Debug)]
pub(crate) struct ProfileAccum {
    window: usize,
    heads: usize,
    prefill_middle: Vec<f64>,
    prefill_sink: Vec<f64>,
    prefill_recent: Vec<f64>,
    prefill_rows: u64,
    decode_middle: Vec<f64>,
    decode_sink: Vec<f64>,
    decode_recent: Vec<f64>,
    decode_rows: u64,
}

impl ProfileAccum {
    pub(crate) fn new(heads: usize, window: usize) -> Self {
        Self {
            window,
            heads,
            prefill_middle: vec![0.0; heads],
            prefill_sink: vec![0.0; heads],
            prefill_recent: vec![0.0; heads],
            prefill_rows: 0,
            decode_middle: vec![0.0; heads],
            decode_sink: vec![0.0; heads],
            decode_recent: vec![0.0; heads],
            decode_rows: 0,
        }
    }

    /// Fold one probability slice in: probs [heads, rows, width] f32,
    /// whose first row sits at absolute position `first_position`.
    /// Decode rows (a 1-row slice at the live tail) accumulate into
    /// the decode set; everything else is prefill.
    pub(crate) fn accumulate(
        &mut self,
        probs: &candle_core::Tensor,
        first_position: usize,
        decode: bool,
    ) -> LibAlmostResult<()> {
        let (heads, rows, width) = probs.dims3()?;
        snafu::ensure_whatever!(
            heads == self.heads,
            "profile accumulator built for {} heads, saw {heads}",
            self.heads
        );
        let device = probs.device();
        // Row eligibility + the recent-band threshold, host-built
        // (rows-sized only; the column masks derive on device).
        let mut thresholds = Vec::with_capacity(rows);
        let mut eligible = Vec::with_capacity(rows);
        let mut eligible_rows = 0u64;
        for row in 0..rows {
            let position = first_position + row;
            if position > self.window {
                thresholds.push((position + 1 - self.window) as f32);
                eligible.push(1f32);
                eligible_rows += 1;
            } else {
                thresholds.push(f32::MAX);
                eligible.push(0f32);
            }
        }
        if eligible_rows == 0 {
            return Ok(());
        }
        let thresholds =
            candle_core::Tensor::from_vec(thresholds, (rows, 1), device)?;
        let eligible_col =
            candle_core::Tensor::from_vec(eligible.clone(), (rows, 1), device)?;
        let eligible_row =
            candle_core::Tensor::from_vec(eligible, (1, rows), device)?;
        let columns = candle_core::Tensor::arange(0f32, width as f32, device)?
            .reshape((1, width))?;

        // recent: column >= threshold; middle: 1 <= column < threshold.
        // Integer-valued f32 arithmetic + clamp keeps the masks exact.
        let recent_mask = (columns.broadcast_sub(&thresholds)? + 1.0)?
            .clamp(0f64, 1f64)?
            .broadcast_mul(&eligible_col)?;
        let column_past_sink = columns.clamp(0f64, 1f64)?;
        let middle_mask = thresholds
            .broadcast_sub(&columns)?
            .clamp(0f64, 1f64)?
            .broadcast_mul(&column_past_sink)?
            .broadcast_mul(&eligible_col)?;

        let middle = probs
            .broadcast_mul(&middle_mask)?
            .sum(2)?
            .sum(1)?
            .to_vec1::<f32>()?;
        let recent = probs
            .broadcast_mul(&recent_mask)?
            .sum(2)?
            .sum(1)?
            .to_vec1::<f32>()?;
        let sink = probs
            .narrow(2, 0, 1)?
            .squeeze(2)?
            .broadcast_mul(&eligible_row)?
            .sum(1)?
            .to_vec1::<f32>()?;

        let (middle_acc, sink_acc, recent_acc, rows_acc) = if decode {
            (
                &mut self.decode_middle,
                &mut self.decode_sink,
                &mut self.decode_recent,
                &mut self.decode_rows,
            )
        } else {
            (
                &mut self.prefill_middle,
                &mut self.prefill_sink,
                &mut self.prefill_recent,
                &mut self.prefill_rows,
            )
        };
        for head in 0..heads {
            middle_acc[head] += middle[head] as f64;
            sink_acc[head] += sink[head] as f64;
            recent_acc[head] += recent[head] as f64;
        }
        *rows_acc += eligible_rows;
        Ok(())
    }

    pub(crate) fn report(self, layer_index: usize) -> LayerProfile {
        let means = |sums: &[f64], rows: u64| -> Vec<f64> {
            sums.iter()
                .map(|sum| if rows == 0 { 0.0 } else { sum / rows as f64 })
                .collect()
        };
        let pack = |middle: Vec<f64>, sink: Vec<f64>, recent: Vec<f64>| {
            middle
                .into_iter()
                .zip(sink)
                .zip(recent)
                .map(|((middle, sink), recent)| HeadMasses { middle, sink, recent })
                .collect()
        };
        LayerProfile {
            layer_index,
            window: self.window,
            prefill_rows: self.prefill_rows,
            decode_rows: self.decode_rows,
            prefill: pack(
                means(&self.prefill_middle, self.prefill_rows),
                means(&self.prefill_sink, self.prefill_rows),
                means(&self.prefill_recent, self.prefill_rows),
            ),
            decode: pack(
                means(&self.decode_middle, self.decode_rows),
                means(&self.decode_sink, self.decode_rows),
                means(&self.decode_recent, self.decode_rows),
            ),
        }
    }
}
