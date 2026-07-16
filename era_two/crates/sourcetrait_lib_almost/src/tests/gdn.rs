//! Hand-pinned numeric locks over the GDN primitives (cpu f32):
//! gating scalars, softplus guard, l2 norm, the causal convs
//! (stateless + carried tail), the recurrence step, and the chunked
//! rule's equivalence to the sequential recurrence.
use crate::gdn::{
    causal_conv_silu,
    chunk_rule,
    conv_with_tail,
    gdn_gates,
    l2_norm,
    recurrent_step,
    softplus,
};

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
fn conv_with_tail_zero_tail_matches_stateless() {
    let x = shaped(&[1.0, 2.0, 3.0, 4.0, 5.0], (5, 1));
    let weight =
        candle_core::Tensor::from_vec(vec![4.0f32, 3.0, 2.0, 1.0], (1, 1, 4), &candle_core::Device::Cpu)
            .expect("weight");
    let tail = candle_core::Tensor::zeros((3, 1), candle_core::DType::F32, &candle_core::Device::Cpu)
        .expect("tail");
    let stateless = causal_conv_silu(&x, &weight).expect("stateless");
    let (carried, new_tail) = conv_with_tail(&x, &weight, &tail).expect("carried");
    assert_close(&values(&carried), &values(&stateless), 1e-7);
    // The new tail holds the last kernel-1 RAW rows.
    assert_close(&values(&new_tail), &[3.0, 4.0, 5.0], 1e-7);
}

#[test]
fn conv_with_tail_chains_across_chunks() {
    let x = shaped(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0], (7, 1));
    let weight =
        candle_core::Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 1.5], (1, 1, 4), &candle_core::Device::Cpu)
            .expect("weight");
    let zero_tail =
        candle_core::Tensor::zeros((3, 1), candle_core::DType::F32, &candle_core::Device::Cpu)
            .expect("tail");
    let (whole, _) = conv_with_tail(&x, &weight, &zero_tail).expect("whole");

    let first = x.narrow(0, 0, 4).expect("first");
    let second = x.narrow(0, 4, 3).expect("second");
    let (out_1, tail_1) = conv_with_tail(&first, &weight, &zero_tail).expect("chunk 1");
    let (out_2, _) = conv_with_tail(&second, &weight, &tail_1).expect("chunk 2");
    let chained: Vec<f32> = values(&out_1).into_iter().chain(values(&out_2)).collect();
    assert_close(&chained, &values(&whole), 1e-7);
}

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

fn synth(seed: &mut u64, shape: (usize, usize, usize)) -> candle_core::Tensor {
    let (a, b, c) = shape;
    candle_core::Tensor::from_vec(splitmix_f32(seed, a * b * c), shape, &candle_core::Device::Cpu)
        .expect("synth tensor")
}

fn nmse(actual: &candle_core::Tensor, reference: &candle_core::Tensor) -> f64 {
    let a = values(actual);
    let r = values(reference);
    assert_eq!(a.len(), r.len());
    let mut numerator = 0f64;
    let mut denominator = 0f64;
    for (x, y) in a.iter().zip(&r) {
        let difference = (*x as f64) - (*y as f64);
        numerator += difference * difference;
        denominator += (*y as f64) * (*y as f64);
    }
    numerator / denominator
}

#[test]
fn chunk_rule_matches_the_sequential_recurrence() {
    // 150 tokens (two full 64-chunks + a ragged 22), 2 heads, dk 3,
    // dv 4, a NONZERO initial state - the chunked rule, the same rule
    // split 70+80 with carried state, and the per-token loop must
    // agree to f32 rounding on outputs AND final states.
    let mut seed = 0x0102_0304_0506_0708u64;
    let t = 150;
    let (heads, dk, dv) = (2, 3, 4);
    let q = (l2_norm(&synth(&mut seed, (t, heads, dk))).expect("q norm")
        * (dk as f64).powf(-0.5))
    .expect("q scale");
    let k = l2_norm(&synth(&mut seed, (t, heads, dk))).expect("k norm");
    let v = synth(&mut seed, (t, heads, dv));
    let g = (synth(&mut seed, (t, heads, 1)).abs().expect("abs") * -0.5)
        .expect("g")
        .squeeze(2)
        .expect("g shape");
    let beta = (synth(&mut seed, (t, heads, 1)).abs().expect("abs") * 1.5)
        .expect("beta")
        .squeeze(2)
        .expect("beta shape");
    let state_0 = synth(&mut seed, (heads, dk, dv))
        .reshape((heads, dk, dv))
        .expect("state");

    // The sequential reference.
    let decay = g.exp().expect("decay");
    let mut state = state_0.clone();
    let mut rows = Vec::with_capacity(t);
    for position in 0..t {
        let narrow = |x: &candle_core::Tensor| x.narrow(0, position, 1).unwrap().squeeze(0).unwrap();
        let (next, y) = recurrent_step(
            &state,
            &narrow(&q),
            &narrow(&k),
            &narrow(&v),
            &narrow(&decay),
            &narrow(&beta),
        )
        .expect("step");
        state = next;
        rows.push(y);
    }
    let sequential_out = candle_core::Tensor::stack(&rows, 0).expect("stack");
    let sequential_state = state;

    // One chunked pass.
    let (chunk_out, chunk_state) = chunk_rule(&q, &k, &v, &g, &beta, &state_0).expect("chunked");
    assert!(nmse(&chunk_out, &sequential_out) <= 1e-10, "whole-pass out");
    assert!(nmse(&chunk_state, &sequential_state) <= 1e-10, "whole-pass state");

    // Split 70 + 80 with carried state.
    let split = 70;
    let head = |x: &candle_core::Tensor| x.narrow(0, 0, split).unwrap();
    let tail = |x: &candle_core::Tensor| x.narrow(0, split, t - split).unwrap();
    let (out_a, state_a) = chunk_rule(
        &head(&q), &head(&k), &head(&v), &head(&g), &head(&beta), &state_0,
    )
    .expect("split a");
    let (out_b, state_b) = chunk_rule(
        &tail(&q), &tail(&k), &tail(&v), &tail(&g), &tail(&beta), &state_a,
    )
    .expect("split b");
    let split_out = candle_core::Tensor::cat(&[&out_a, &out_b], 0).expect("cat");
    assert!(nmse(&split_out, &sequential_out) <= 1e-10, "split out");
    assert!(nmse(&state_b, &sequential_state) <= 1e-10, "split state");
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
