//! Hand-pinned numeric locks over the norm primitives (cpu f32).
use crate::gdn::{gdn_gates, softplus};
use crate::norms::{rms_norm, rms_norm_gated};

fn tensor(values: &[f32]) -> candle_core::Tensor {
    candle_core::Tensor::new(values, &candle_core::Device::Cpu).expect("tensor")
}

fn values(tensor: &candle_core::Tensor) -> Vec<f32> {
    tensor.to_vec1::<f32>().expect("f32 vec")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "index {index}: {a} vs {e} (tolerance {tolerance})"
        );
    }
}

#[test]
fn rms_norm_matches_hand_computation() {
    // var([1,2,3]) = 14/3; rsqrt(var + 1e-6) = 0.46291 (to 5 places).
    let out = rms_norm(&tensor(&[1.0, 2.0, 3.0]), &tensor(&[1.0, 1.0, 1.0]), 1e-6)
        .expect("norms");
    assert_close(&values(&out), &[0.462_91, 0.925_82, 1.388_73], 1e-5);
}

#[test]
fn rms_norm_applies_weight_after_normalize() {
    let out = rms_norm(&tensor(&[1.0, 2.0, 3.0]), &tensor(&[2.0, 0.5, -1.0]), 1e-6)
        .expect("norms");
    assert_close(&values(&out), &[0.925_82, 0.462_91, -1.388_73], 1e-5);
}

#[test]
fn rms_norm_gated_is_norm_before_gate() {
    // gate 0 -> silu(0) = 0 everywhere; a gate-before-norm ordering
    // would divide by a zero variance instead.
    let out = rms_norm_gated(
        &tensor(&[1.0, 2.0, 3.0]),
        &tensor(&[0.0, 0.0, 0.0]),
        &tensor(&[1.0, 1.0, 1.0]),
        1e-5,
    )
    .expect("gated");
    assert_close(&values(&out), &[0.0, 0.0, 0.0], 1e-7);

    // gate 1 -> silu(1) = 0.731059; scales the normed values uniformly.
    let out = rms_norm_gated(
        &tensor(&[1.0, 2.0, 3.0]),
        &tensor(&[1.0, 1.0, 1.0]),
        &tensor(&[1.0, 1.0, 1.0]),
        1e-5,
    )
    .expect("gated");
    assert_close(&values(&out), &[0.338_411, 0.676_822, 1.015_233], 1e-5);
}

#[test]
fn softplus_guard_passes_large_inputs_through() {
    let out = softplus(&tensor(&[0.0, 25.0, -5.0])).expect("softplus");
    let got = values(&out);
    assert_close(&[got[0]], &[std::f32::consts::LN_2], 1e-6);
    assert_eq!(got[1], 25.0, "guarded region is exact passthrough");
    assert_close(&[got[2]], &[0.006_715], 1e-6);
}

#[test]
fn gdn_gates_pin_the_scalar_math() {
    // a = 0, dt_bias = 0 -> softplus(0) = ln 2; A_log = 0 -> exp = 1;
    // g = -ln 2. b = 0 -> sigmoid = 0.5 -> beta = 1.0 under
    // neg-eigval, 0.5 without.
    let (g, beta) = gdn_gates(
        &tensor(&[0.0]),
        &tensor(&[0.0]),
        &tensor(&[0.0]),
        &tensor(&[0.0]),
        true,
    )
    .expect("gates");
    assert_close(&values(&g), &[-std::f32::consts::LN_2], 1e-6);
    assert_close(&values(&beta), &[1.0], 1e-7);

    let (_, beta) = gdn_gates(
        &tensor(&[0.0]),
        &tensor(&[0.0]),
        &tensor(&[0.0]),
        &tensor(&[0.0]),
        false,
    )
    .expect("gates");
    assert_close(&values(&beta), &[0.5], 1e-7);
}
