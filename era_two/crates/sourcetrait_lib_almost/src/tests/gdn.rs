//! Hand-pinned numeric locks over the GDN primitives (cpu f32):
//! gating scalars, softplus guard, l2 norm, the stateless causal conv,
//! and the recurrence step.
use crate::gdn::{causal_conv_silu, gdn_gates, l2_norm, recurrent_step, softplus};

fn tensor(values: &[f32]) -> candle_core::Tensor {
    candle_core::Tensor::new(values, &candle_core::Device::Cpu).expect("tensor")
}

fn shaped(values: &[f32], shape: (usize, usize)) -> candle_core::Tensor {
    candle_core::Tensor::from_vec(values.to_vec(), shape, &candle_core::Device::Cpu)
        .expect("shaped tensor")
}

fn values(tensor: &candle_core::Tensor) -> Vec<f32> {
    tensor.flatten_all().expect("flatten").to_vec1::<f32>().expect("f32 vec")
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

#[test]
fn l2_norm_uses_sum_not_mean() {
    // [3, 4]: sum of squares 25 -> unit vector [0.6, 0.8]. RMSNorm's
    // mean would divide by sqrt(12.5) instead - the pin separates them.
    let out = l2_norm(&shaped(&[3.0, 4.0], (1, 2))).expect("l2 norm");
    assert_close(&values(&out), &[0.6, 0.8], 1e-6);
}

#[test]
fn causal_conv_orients_last_weight_at_current_token() {
    // One channel, kernel [10, 1] over [1, 2, 3]: out[t] = 10*x[t-1]
    // + 1*x[t] (zero history) = [1, 12, 23] pre-silu. A flipped or
    // right-padded conv breaks every pinned value.
    let x = shaped(&[1.0, 2.0, 3.0], (3, 1));
    let weight =
        candle_core::Tensor::from_vec(vec![10.0f32, 1.0], (1, 1, 2), &candle_core::Device::Cpu)
            .expect("weight");
    let out = causal_conv_silu(&x, &weight).expect("conv");
    assert_close(&values(&out), &[0.731_058_6, 11.999_926, 23.0], 1e-4);
}

#[test]
fn recurrent_step_pins_the_delta_order() {
    // One head, dk 2, dv 1. Step 1 from zero state: decay is moot,
    // k = e0, v = 2, beta = 0.5 -> state row0 = 1, y = q . state = 1.
    // Step 2: decay 0.5 halves row0 BEFORE the delta (the order the
    // pin guards), k = e1, v = 3, beta = 1 -> rows [0.5, 3], y = 3.5.
    let state = candle_core::Tensor::zeros((1, 2, 1), candle_core::DType::F32, &candle_core::Device::Cpu)
        .expect("state");
    let (state, y_1) = recurrent_step(
        &state,
        &shaped(&[1.0, 1.0], (1, 2)),
        &shaped(&[1.0, 0.0], (1, 2)),
        &shaped(&[2.0], (1, 1)),
        &tensor(&[0.5]),
        &tensor(&[0.5]),
    )
    .expect("step 1");
    assert_close(&values(&y_1), &[1.0], 1e-7);
    assert_close(&values(&state), &[1.0, 0.0], 1e-7);

    let (state, y_2) = recurrent_step(
        &state,
        &shaped(&[1.0, 1.0], (1, 2)),
        &shaped(&[0.0, 1.0], (1, 2)),
        &shaped(&[3.0], (1, 1)),
        &tensor(&[0.5]),
        &tensor(&[1.0]),
    )
    .expect("step 2");
    assert_close(&values(&y_2), &[3.5], 1e-7);
    assert_close(&values(&state), &[0.5, 3.0], 1e-7);
}
