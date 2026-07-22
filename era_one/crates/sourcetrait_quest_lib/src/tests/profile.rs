use crate::profile::ProfileAccum;

/// Uniform weights over kv=20 at offset 18, q=2, window 2: rows sit at
/// p=18,19 with before-window thresholds 17,18; sink (S_PROBE=16) covers
/// columns 0..16, so middle = (17-16)+(18-16) columns = 3/20 of the mass.
#[test]
fn partition_sums_match_hand_math() {
    let device = candle_core::Device::Cpu;
    let heads = 2usize;
    let q = 2usize;
    let kv = 20usize;
    let weights = candle_core::Tensor::full(1.0f32 / kv as f32, (1, heads, q, kv), &device).unwrap();
    let mut accum = ProfileAccum::new(heads, 2);
    accum.observe(&weights, 18).unwrap();
    let report = accum.drain();
    assert_eq!(report.prefill_rows, 2);
    assert_eq!(report.prefill_rows_past_window, 2);
    assert_eq!(report.decode_rows, 0);
    for head in &report.heads {
        assert!((head.prefill_middle - 3.0 / 20.0).abs() < 1e-6);
        assert!((head.prefill_sink_before - 32.0 / 20.0).abs() < 1e-6);
        for position_mass in &head.sink_by_position {
            assert!((position_mass - 2.0 / 20.0).abs() < 1e-6);
        }
    }
}

/// A q=1 forward lands in the decode accumulators.
#[test]
fn decode_rows_accumulate_separately() {
    let device = candle_core::Device::Cpu;
    let kv = 20usize;
    let weights = candle_core::Tensor::full(1.0f32 / kv as f32, (1, 1, 1, kv), &device).unwrap();
    let mut accum = ProfileAccum::new(1, 2);
    accum.observe(&weights, 19).unwrap();
    let report = accum.drain();
    assert_eq!(report.prefill_rows, 0);
    assert_eq!(report.decode_rows, 1);
    assert_eq!(report.decode_rows_past_window, 1);
    // p=19, t=18: before = 18/20, sink_before = 16/20, middle = 2/20.
    assert!((report.heads[0].decode_middle - 2.0 / 20.0).abs() < 1e-6);
}
