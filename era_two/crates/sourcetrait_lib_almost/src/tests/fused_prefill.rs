//! PrefillDispatch locks: the TriSolve-backed chunk path against the
//! classic candle chain on the real recurrence rig (cuda;
//! checkpoint-free).
use crate::fused_prefill::chunk_rule_fused;
use crate::gdn::{chunk_rule, l2_norm};

fn splitmix_f32(seed: &mut u64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| {
            *seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = *seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn synth(
    seed: &mut u64,
    shape: (usize, usize, usize),
    device: &candle_core::Device,
) -> candle_core::Tensor {
    let (a, b, c) = shape;
    candle_core::Tensor::from_vec(splitmix_f32(seed, a * b * c), shape, device).expect("synth")
}

fn nmse(actual: &candle_core::Tensor, reference: &candle_core::Tensor) -> f64 {
    let a = actual
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host");
    let r = reference
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host");
    assert_eq!(a.len(), r.len());
    let mut numerator = 0f64;
    let mut denominator = 0f64;
    for (x, y) in a.iter().zip(&r) {
        let difference = (*x as f64) - (*y as f64);
        numerator += difference * difference;
        denominator += (*y as f64) * (*y as f64);
    }
    numerator / denominator.max(f64::MIN_POSITIVE)
}

/// The fused chunk path against the classic chain: 150 tokens (two
/// full 64-chunks + a ragged 22), the real geometry (30 heads, dk 96,
/// dv 192), a NONZERO initial state - whole-pass AND split-with-carry
/// must agree to the f32 reassociation class (the kernel runs serial
/// in-thread dots where cublas tiles).
#[test]
#[ignore = "needs a cuda card"]
fn fused_chunk_rule_matches_the_classic_chain() {
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let mut seed = 0x0102_0304_0506_0708u64;
    let t = 150;
    let (heads, dk, dv) = (30, 96, 192);
    let q = (l2_norm(&synth(&mut seed, (t, heads, dk), &device)).expect("q norm")
        * (dk as f64).powf(-0.5))
    .expect("q scale");
    let k = l2_norm(&synth(&mut seed, (t, heads, dk), &device)).expect("k norm");
    let v = synth(&mut seed, (t, heads, dv), &device);
    let g = (synth(&mut seed, (t, heads, 1), &device).abs().expect("abs") * -0.5)
        .expect("g")
        .squeeze(2)
        .expect("g shape");
    let beta = (synth(&mut seed, (t, heads, 1), &device).abs().expect("abs") * 1.5)
        .expect("beta")
        .squeeze(2)
        .expect("beta shape");
    let state_0 = synth(&mut seed, (heads, dk, dv), &device);

    let (classic_out, classic_state) =
        chunk_rule(&q, &k, &v, &g, &beta, &state_0).expect("classic");
    let (fused_out, fused_state) =
        chunk_rule_fused(&q, &k, &v, &g, &beta, &state_0).expect("fused");
    let out_nmse = nmse(&fused_out, &classic_out);
    let state_nmse = nmse(&fused_state, &classic_state);
    println!(
        "fused-prefill whole-pass vs classic: out nmse {out_nmse:.3e}, state nmse {state_nmse:.3e}"
    );
    assert!(out_nmse <= 1e-9, "out {out_nmse:.3e} beyond the f32 reassociation class");
    assert!(
        state_nmse <= 1e-9,
        "state {state_nmse:.3e} beyond the f32 reassociation class"
    );

    // Split 70 + 80 with carried state through the FUSED path against
    // the classic whole pass.
    let split = 70;
    let head = |x: &candle_core::Tensor| x.narrow(0, 0, split).unwrap();
    let tail = |x: &candle_core::Tensor| x.narrow(0, split, t - split).unwrap().contiguous().unwrap();
    let (out_a, state_a) = chunk_rule_fused(
        &head(&q), &head(&k), &head(&v), &head(&g), &head(&beta), &state_0,
    )
    .expect("fused split a");
    let (out_b, state_b) = chunk_rule_fused(
        &tail(&q), &tail(&k), &tail(&v), &tail(&g), &tail(&beta), &state_a,
    )
    .expect("fused split b");
    let split_out = candle_core::Tensor::cat(&[&out_a, &out_b], 0).expect("cat");
    let split_nmse = nmse(&split_out, &classic_out);
    let split_state_nmse = nmse(&state_b, &classic_state);
    println!(
        "fused-prefill split-carry vs classic: out nmse {split_nmse:.3e}, state nmse {split_state_nmse:.3e}"
    );
    assert!(
        split_nmse <= 1e-9,
        "split out {split_nmse:.3e} beyond the f32 reassociation class"
    );
    assert!(
        split_state_nmse <= 1e-9,
        "split state {split_state_nmse:.3e} beyond the f32 reassociation class"
    );
}
