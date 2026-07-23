use crate::*;

/// Sink columns the observation pass tracks individually (a superset of
/// any S the phase-3 mechanism would pin; the per-position curve picks S).
pub(crate) const S_PROBE: usize = 16;

/// Per-full-layer observation accumulator (A2 phase 2): partitions each
/// query row's attention mass into sink (first S_PROBE columns), window
/// (the trailing w positions), and middle (everything between - the
/// retrieval evidence), from the eager path's f32 softmax weights.
/// Prefill and decode rows accumulate separately: decode-row middle mass
/// is the response-conditioned importance signal (LookaheadKV's GT
/// definition restricted to this head).
#[derive(Debug, Clone)]
pub(crate) struct ProfileAccum {
    pub(crate) window: usize,
    heads: usize,
    prefill_before: Vec<f64>,
    prefill_sink_before: Vec<f64>,
    sink_by_position: Vec<f64>,
    prefill_rows: u64,
    prefill_rows_past_window: u64,
    decode_before: Vec<f64>,
    decode_sink_before: Vec<f64>,
    decode_rows: u64,
    decode_rows_past_window: u64,
}

/// One head's drained totals (see ProfileAccum; middle = before -
/// sink_before, the mass that is neither sink nor window).
pub(crate) struct HeadReport {
    pub(crate) prefill_middle: f64,
    pub(crate) prefill_sink_before: f64,
    pub(crate) decode_middle: f64,
    pub(crate) decode_sink_before: f64,
    pub(crate) sink_by_position: Vec<f64>,
}

/// A full layer's drained observation report.
pub(crate) struct LayerReport {
    pub(crate) window: usize,
    pub(crate) s_probe: usize,
    pub(crate) prefill_rows: u64,
    pub(crate) prefill_rows_past_window: u64,
    pub(crate) decode_rows: u64,
    pub(crate) decode_rows_past_window: u64,
    pub(crate) heads: Vec<HeadReport>,
}

impl ProfileAccum {
    pub(crate) fn new(heads: usize, window: usize) -> Self {
        Self {
            window,
            heads,
            prefill_before: vec![0.0; heads],
            prefill_sink_before: vec![0.0; heads],
            sink_by_position: vec![0.0; heads * S_PROBE],
            prefill_rows: 0,
            prefill_rows_past_window: 0,
            decode_before: vec![0.0; heads],
            decode_sink_before: vec![0.0; heads],
            decode_rows: 0,
            decode_rows_past_window: 0,
        }
    }

    /// Accumulate one eager forward's post-softmax weights
    /// (b=1, h, q, kv) at absolute offset. Full layers only: kv columns
    /// are absolute positions 0..offset+q.
    pub(crate) fn observe(
        &mut self,
        weights_f32: &candle_core::Tensor,
        offset: usize,
    ) -> QuestResult<()> {
        let (_b, heads, rows, kv) = weights_f32.dims4()?;
        snafu::ensure_whatever!(
            heads == self.heads,
            "profile accumulator holds {} heads but the forward carries {heads}",
            self.heads
        );
        let device = weights_f32.device();
        let window = self.window;

        // Row thresholds t(i) = (offset + i) - window + 1: columns strictly
        // below t sit before the window. Integer-valued f32 arithmetic, so
        // clamp(diff, 0, 1) is an exact 0/1 mask.
        let row_positions = candle_core::Tensor::arange(0u32, rows as u32, device)?
            .to_dtype(candle_core::DType::F32)?
            .reshape((rows, 1))?;
        let thresholds =
            ((row_positions + offset as f64)? - (window as f64 - 1.0))?;
        let columns = candle_core::Tensor::arange(0u32, kv as u32, device)?
            .to_dtype(candle_core::DType::F32)?
            .reshape((1, kv))?;
        let before_mask = thresholds
            .broadcast_sub(&columns)?
            .clamp(0f32, 1f32)?;
        let before = weights_f32
            .broadcast_mul(&before_mask.unsqueeze(0)?.unsqueeze(0)?)?
            .sum(3)?
            .sum(2)?
            .squeeze(0)?
            .to_vec1::<f32>()?;

        let sink_columns = S_PROBE.min(kv);
        let weights_sink = weights_f32.narrow(3, 0, sink_columns)?;
        // Sink columns that are ALSO before the window: threshold clamped
        // into [0, S_PROBE] per row.
        let sink_thresholds = thresholds.clamp(0f32, sink_columns as f32)?;
        let sink_before_mask = sink_thresholds
            .broadcast_sub(&columns.narrow(1, 0, sink_columns)?)?
            .clamp(0f32, 1f32)?;
        let sink_before = weights_sink
            .broadcast_mul(&sink_before_mask.unsqueeze(0)?.unsqueeze(0)?)?
            .sum(3)?
            .sum(2)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        let sink_positions = weights_sink
            .sum(2)?
            .squeeze(0)?
            .to_vec2::<f32>()?;

        let rows_past_window = (offset + rows).saturating_sub(offset.max(window)) as u64;
        let (mass_before, mass_sink_before, row_count, row_count_past) = if rows == 1 {
            (
                &mut self.decode_before,
                &mut self.decode_sink_before,
                &mut self.decode_rows,
                &mut self.decode_rows_past_window,
            )
        } else {
            (
                &mut self.prefill_before,
                &mut self.prefill_sink_before,
                &mut self.prefill_rows,
                &mut self.prefill_rows_past_window,
            )
        };
        for head in 0..heads {
            mass_before[head] += before[head] as f64;
            mass_sink_before[head] += sink_before[head] as f64;
        }
        *row_count += rows as u64;
        *row_count_past += rows_past_window;
        for (head, row) in sink_positions.iter().enumerate() {
            for (position, value) in row.iter().enumerate() {
                self.sink_by_position[head * S_PROBE + position] += *value as f64;
            }
        }
        Ok(())
    }

    /// Drain into a report (the accumulator resets).
    pub(crate) fn drain(&mut self) -> LayerReport {
        let heads = (0..self.heads)
            .map(|head| HeadReport {
                prefill_middle: self.prefill_before[head] - self.prefill_sink_before[head],
                prefill_sink_before: self.prefill_sink_before[head],
                decode_middle: self.decode_before[head] - self.decode_sink_before[head],
                decode_sink_before: self.decode_sink_before[head],
                sink_by_position: self.sink_by_position
                    [head * S_PROBE..(head + 1) * S_PROBE]
                    .to_vec(),
            })
            .collect();
        let report = LayerReport {
            window: self.window,
            s_probe: S_PROBE,
            prefill_rows: self.prefill_rows,
            prefill_rows_past_window: self.prefill_rows_past_window,
            decode_rows: self.decode_rows,
            decode_rows_past_window: self.decode_rows_past_window,
            heads,
        };
        *self = Self::new(self.heads, self.window);
        report
    }
}
