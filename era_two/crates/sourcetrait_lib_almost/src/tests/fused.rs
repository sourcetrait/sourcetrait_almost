//! GdnChainFusion locks: the fused decode step and the fused norms
//! against the classic candle chains on the real geometry (cuda;
//! checkpoint-free).
use crate::fused::{ConvStepFused, GdnFusedStep, RmsNormFused, RmsNormGatedFused};
use crate::gdn::{conv_with_tail, recurrent_step};
use crate::norms::{rms_norm, rms_norm_gated};

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

/// The fused conv decode tap against the classic conv_with_tail
/// chain over carried steps at the model width. The fused form keeps
/// the tail in raw f32 where the classic chain rounds it through
/// bf16 per step - a bf16 quantization-class spread.
#[test]
#[ignore = "needs a cuda card"]
fn fused_conv_step_matches_the_classic_chain() {
    const STEPS: usize = 6;
    let channels = 11520usize;
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let mut seed = 0xC0FF_EE00_DEAD_BEEFu64;
    let weight_rows = synth(&mut seed, (4, channels), &device)
        .to_dtype(candle_core::DType::BF16)
        .expect("weight rows");
    // The classic layout: [channels, 1, kernel].
    let weight = weight_rows
        .transpose(0, 1)
        .expect("transpose")
        .reshape((channels, 1, 4))
        .expect("weight")
        .contiguous()
        .expect("packed weight");
    let classic_tail =
        candle_core::Tensor::zeros((3, channels), candle_core::DType::F32, &device)
            .expect("classic tail");
    let fused_tail = classic_tail.copy().expect("fused tail");

    let mut worst_out = 0f64;
    for _ in 0..STEPS {
        let x = synth(&mut seed, (1, channels), &device)
            .to_dtype(candle_core::DType::BF16)
            .expect("x bf16");
        let (classic_out, new_tail) =
            conv_with_tail(&x, &weight, &classic_tail).expect("classic conv");
        classic_tail.slice_set(&new_tail, 0, 0).expect("classic carry");

        let packed = candle_core::Tensor::cat(&[&x, &weight_rows], 0).expect("pack");
        let fused_out =
            candle_core::Tensor::zeros((1, channels), candle_core::DType::BF16, &device)
                .expect("out");
        fused_out
            .inplace_op3(&fused_tail, &packed, &ConvStepFused)
            .expect("fused conv step");
        worst_out = worst_out.max(relative_spread(
            &fused_out.to_dtype(candle_core::DType::F32).expect("f32"),
            &classic_out.to_dtype(candle_core::DType::F32).expect("f32"),
        ));
    }
    let tail_spread = relative_spread(&fused_tail, &classic_tail);
    println!(
        "fused conv vs classic over {STEPS} carried steps: worst out nmse {worst_out:.3e}, \
         final tail nmse {tail_spread:.3e}"
    );
    assert!(worst_out <= 5e-5, "out spread {worst_out:.3e} beyond the bf16 class");
    assert!(tail_spread <= 5e-5, "tail spread {tail_spread:.3e} beyond the bf16 class");
}

/// The fused plain norm against norms::rms_norm on bf16 cuda at the
/// model width. The fused form rounds ONCE at the store where the
/// classic chain rounds normed before the weight mul - a bf16
/// quantization-class spread, not bit equality.
#[test]
#[ignore = "needs a cuda card"]
fn fused_rms_norm_matches_the_classic_chain() {
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let mut seed = 0x0BAD_5EED_0123_4567u64;
    let x = synth(&mut seed, (3, 3840), &device)
        .to_dtype(candle_core::DType::BF16)
        .expect("x bf16");
    let weight = synth(&mut seed, (1, 3840), &device)
        .squeeze(0)
        .expect("weight shape")
        .to_dtype(candle_core::DType::BF16)
        .expect("weight bf16");
    let classic = rms_norm(&x, &weight, 1e-6).expect("classic");
    let fused = candle_core::Tensor::zeros((3, 3840), candle_core::DType::BF16, &device)
        .expect("out");
    fused
        .inplace_op3(&x, &weight, &RmsNormFused { eps: 1e-6 })
        .expect("fused norm");
    let spread = relative_spread(
        &fused.to_dtype(candle_core::DType::F32).expect("f32"),
        &classic.to_dtype(candle_core::DType::F32).expect("f32"),
    );
    println!("fused rms_norm vs classic: nmse {spread:.3e}");
    assert!(spread <= 5e-5, "spread {spread:.3e} beyond the bf16 rounding class");
}

/// The fused gated norm against the classic finish_mixer semantics
/// (y cast to bf16 first there; the fused form consumes raw f32 y).
#[test]
#[ignore = "needs a cuda card"]
fn fused_rms_norm_gated_matches_the_classic_chain() {
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let mut seed = 0xFEED_FACE_0000_0001u64;
    let y = synth(&mut seed, (30, 192), &device);
    let gate = synth(&mut seed, (30, 192), &device)
        .to_dtype(candle_core::DType::BF16)
        .expect("gate bf16");
    let weight = synth(&mut seed, (1, 192), &device)
        .squeeze(0)
        .expect("weight shape")
        .to_dtype(candle_core::DType::BF16)
        .expect("weight bf16");
    let classic = rms_norm_gated(
        &y.to_dtype(candle_core::DType::BF16).expect("y bf16"),
        &gate,
        &weight,
        1e-5,
    )
    .expect("classic");
    let packed = candle_core::Tensor::cat(
        &[&y, &gate.to_dtype(candle_core::DType::F32).expect("gate f32")],
        1,
    )
    .expect("pack")
    .contiguous()
    .expect("packed rows");
    let fused = candle_core::Tensor::zeros((30, 192), candle_core::DType::BF16, &device)
        .expect("out");
    fused
        .inplace_op3(&packed, &weight, &RmsNormGatedFused { eps: 1e-5 })
        .expect("fused gated norm");
    let spread = relative_spread(
        &fused.to_dtype(candle_core::DType::F32).expect("f32"),
        &classic.to_dtype(candle_core::DType::F32).expect("f32"),
    );
    println!("fused rms_norm_gated vs classic: nmse {spread:.3e}");
    assert!(spread <= 5e-5, "spread {spread:.3e} beyond the bf16 rounding class");
}
