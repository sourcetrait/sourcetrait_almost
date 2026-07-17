//! GdnChainFusion locks: the fused decode step against the classic
//! candle chain on the real geometry (cuda; checkpoint-free).
use crate::fused::GdnFusedStep;
use crate::gdn::recurrent_step;

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
    shape: (usize, usize),
    device: &candle_core::Device,
) -> candle_core::Tensor {
    let (a, b) = shape;
    candle_core::Tensor::from_vec(splitmix_f32(seed, a * b), shape, device).expect("synth")
}

fn relative_spread(actual: &candle_core::Tensor, reference: &candle_core::Tensor) -> f64 {
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
    let mut numerator = 0f64;
    let mut denominator = 0f64;
    for (x, y) in a.iter().zip(&r) {
        let difference = (*x as f64) - (*y as f64);
        numerator += difference * difference;
        denominator += (*y as f64) * (*y as f64);
    }
    numerator / denominator.max(f64::MIN_POSITIVE)
}

/// The fused kernel against the classic chain, single step AND a
/// 64-step carried chain, on the real (30, 96, 192) geometry. The two
/// deliberate reassociations (the decay fold, serial in-thread sums)
/// bound the single-step spread at the f32 rounding class; the chain
/// leg pins the compounding drift.
#[test]
#[ignore = "needs a cuda card"]
fn fused_step_matches_the_classic_chain() {
    const STEPS: usize = 64;
    let (heads, dk, dv) = (30usize, 96usize, 192usize);
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let mut seed = 0x00AA_BB01_5EED_F00Du64;

    let state_start = candle_core::Tensor::from_vec(
        splitmix_f32(&mut seed, heads * dk * dv),
        (heads, dk, dv),
        &device,
    )
    .expect("state");
    let classic_state = state_start.copy().expect("classic state");
    let fused_state = state_start.copy().expect("fused state");

    let mut worst_step_y = 0f64;
    let mut worst_step_state = 0f64;
    for _ in 0..STEPS {
        let q = synth(&mut seed, (heads, dk), &device);
        let k = synth(&mut seed, (heads, dk), &device);
        let v = synth(&mut seed, (heads, dv), &device);
        // Decay in (0, 1), beta in (0, 2) - the live gate ranges.
        let decay = ((synth(&mut seed, (heads, 1), &device).abs().expect("abs") * 0.9)
            .expect("scale")
            + 0.05)
            .expect("shift")
            .squeeze(1)
            .expect("decay");
        let beta = (synth(&mut seed, (heads, 1), &device).abs().expect("abs") * 1.5)
            .expect("beta")
            .squeeze(1)
            .expect("beta shape");

        let (next_classic, y_classic) =
            recurrent_step(&classic_state, &q, &k, &v, &decay, &beta).expect("classic step");
        classic_state
            .slice_set(&next_classic, 0, 0)
            .expect("classic carry");

        let packed = candle_core::Tensor::cat(
            &[
                &q,
                &k,
                &v,
                &decay.reshape((heads, 1)).expect("decay col"),
                &beta.reshape((heads, 1)).expect("beta col"),
            ],
            1,
        )
        .expect("pack")
        .contiguous()
        .expect("packed rows");
        let y_fused = candle_core::Tensor::zeros((heads, dv), candle_core::DType::F32, &device)
            .expect("y");
        fused_state
            .inplace_op3(&packed, &y_fused, &GdnFusedStep)
            .expect("fused step");

        worst_step_y = worst_step_y.max(relative_spread(&y_fused, &y_classic));
        worst_step_state = worst_step_state.max(relative_spread(&fused_state, &classic_state));
    }
    println!(
        "fused-vs-classic over {STEPS} carried steps: worst y nmse {worst_step_y:.3e}, \
         worst state nmse {worst_step_state:.3e}"
    );
    assert!(
        worst_step_y <= 1e-9,
        "y spread {worst_step_y:.3e} beyond the f32 reassociation class"
    );
    assert!(
        worst_step_state <= 1e-9,
        "state spread {worst_step_state:.3e} beyond the f32 reassociation class"
    );
}
