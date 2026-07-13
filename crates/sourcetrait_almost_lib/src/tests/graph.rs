use crate::graph::SlotWrite;

/// The E4 building-block contract: a raw-launched slot_write at a
/// device-resident index lands exactly where a host-offset slice_set
/// would. Skips quietly off-gpu (the cuda feature compiles on any box).
#[test]
fn slot_write_matches_slice_set() {
    if !crate::r::candle::cuda_is_available() {
        return;
    }
    let device = candle_core::Device::new_cuda(0).unwrap();
    for dtype in [candle_core::DType::F32, candle_core::DType::BF16] {
        let dst = candle_core::Tensor::zeros((1, 4, 8, 16), dtype, &device).unwrap();
        let reference = candle_core::Tensor::zeros((1, 4, 8, 16), dtype, &device).unwrap();
        let src = candle_core::Tensor::arange(1f32, (4 * 16 + 1) as f32, &device)
            .unwrap()
            .reshape((1, 4, 1, 16))
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let slot = candle_core::Tensor::new(&[5u32], &device).unwrap();

        dst.inplace_op3(&src, &slot, &SlotWrite { dtype }).unwrap();
        reference.slice_set(&src, 2, 5).unwrap();

        let written = dst
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = reference
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(written, expected, "dtype {dtype:?}");
    }
}

/// Re-staging the slot buffer moves later writes without touching the
/// kernel or its arguments - the staged-buffer indirection the decode
/// graph replays against.
#[test]
fn slot_write_follows_restaged_index() {
    if !crate::r::candle::cuda_is_available() {
        return;
    }
    let device = candle_core::Device::new_cuda(0).unwrap();
    let dst = candle_core::Tensor::zeros((1, 2, 4, 8), candle_core::DType::F32, &device).unwrap();
    let src_one = candle_core::Tensor::full(1f32, (1, 2, 1, 8), &device).unwrap();
    let src_two = candle_core::Tensor::full(2f32, (1, 2, 1, 8), &device).unwrap();
    let slot = candle_core::Tensor::new(&[0u32], &device).unwrap();

    let op = SlotWrite {
        dtype: candle_core::DType::F32,
    };
    dst.inplace_op3(&src_one, &slot, &op).unwrap();
    // Re-stage the index in place (what the host does between replays).
    slot.slice_set(&candle_core::Tensor::new(&[3u32], &device).unwrap(), 0, 0)
        .unwrap();
    dst.inplace_op3(&src_two, &slot, &op).unwrap();

    let sums = dst
        .sum(3)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // Per head: slot 0 sums to 8 (ones), slot 3 sums to 16 (twos).
    assert_eq!(sums, vec![8.0, 0.0, 0.0, 16.0, 8.0, 0.0, 0.0, 16.0]);
}
