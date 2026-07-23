//! PrefillDispatch locks: the rule kernels (TriSolve + StateAdvance
//! over the PackTrim back half) against the classic candle chain on
//! the real recurrence rig, and the PrepChunk end-to-end pipeline
//! against the classic conv/l2/gates chain (cuda; checkpoint-free).
//! SmallT: the t = 32 and t = 17 legs drive the tile-32 module (the
//! speculation verify/readvance span class); the t = 150 legs keep
//! the tile-64 module and their standing digits.
use crate::fused_prefill::{PrepInputs, prep_chunk_rule, rule_from_parts};
use crate::gdn::{chunk_rule, conv_with_tail, gdn_gates, l2_norm};

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

fn synth3(
    seed: &mut u64,
    shape: (usize, usize, usize),
    device: &candle_core::Device,
) -> candle_core::Tensor {
    let (a, b, c) = shape;
    candle_core::Tensor::from_vec(splitmix_f32(seed, a * b * c), shape, device).expect("synth")
}

fn synth2(
    seed: &mut u64,
    shape: (usize, usize),
    device: &candle_core::Device,
) -> candle_core::Tensor {
    let (a, b) = shape;
    candle_core::Tensor::from_vec(splitmix_f32(seed, a * b), shape, device).expect("synth")
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

/// One rule-kernel leg at a given t: whole-pass AND split-with-carry
/// vs the classic chain, at the BundleBf16 quantization class (the
/// bar RECALIBRATED from the f32-reassociation 1e-9 when the
/// bundle's q/k/w/u columns went bf16-pair; g/beta and the state
/// stay f32-exact, and the per-element accumulation orders are
/// unchanged - the spread is pure operand quantization).
fn chunk_rule_leg(t: usize, split: usize, seed_base: u64) {
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let mut seed = seed_base;
    let (heads, dk, dv) = (30, 96, 192);
    let q = (l2_norm(&synth3(&mut seed, (t, heads, dk), &device)).expect("q norm")
        * (dk as f64).powf(-0.5))
    .expect("q scale");
    let k = l2_norm(&synth3(&mut seed, (t, heads, dk), &device)).expect("k norm");
    let v = synth3(&mut seed, (t, heads, dv), &device);
    let g = (synth3(&mut seed, (t, heads, 1), &device).abs().expect("abs") * -0.5)
        .expect("g")
        .squeeze(2)
        .expect("g shape");
    let beta = (synth3(&mut seed, (t, heads, 1), &device).abs().expect("abs") * 1.5)
        .expect("beta")
        .squeeze(2)
        .expect("beta shape");
    let state_0 = synth3(&mut seed, (heads, dk, dv), &device);

    let (classic_out, classic_state) =
        chunk_rule(&q, &k, &v, &g, &beta, &state_0).expect("classic");
    let (fused_out, fused_state) =
        rule_from_parts(&q, &k, &v, &g, &beta, &state_0).expect("fused");
    let out_nmse = nmse(&fused_out, &classic_out);
    let state_nmse = nmse(&fused_state, &classic_state);
    println!(
        "fused-prefill t {t} whole-pass vs classic: out nmse {out_nmse:.3e}, \
         state nmse {state_nmse:.3e}"
    );
    assert!(
        out_nmse <= 1e-4,
        "t {t}: out {out_nmse:.3e} beyond the BundleBf16 quantization class"
    );
    assert!(
        state_nmse <= 1e-4,
        "t {t}: state {state_nmse:.3e} beyond the BundleBf16 quantization class"
    );

    // Split with carried state through the FUSED path against the
    // classic whole pass (at t <= 32 both halves ride tile-32 - the
    // readvance shape: a carried state entering a small span).
    let head = |x: &candle_core::Tensor| x.narrow(0, 0, split).unwrap();
    let tail = |x: &candle_core::Tensor| {
        x.narrow(0, split, t - split).unwrap().contiguous().unwrap()
    };
    let (out_a, state_a) = rule_from_parts(
        &head(&q), &head(&k), &head(&v), &head(&g), &head(&beta), &state_0,
    )
    .expect("fused split a");
    let (out_b, state_b) = rule_from_parts(
        &tail(&q), &tail(&k), &tail(&v), &tail(&g), &tail(&beta), &state_a,
    )
    .expect("fused split b");
    let split_out = candle_core::Tensor::cat(&[&out_a, &out_b], 0).expect("cat");
    let split_nmse = nmse(&split_out, &classic_out);
    let split_state_nmse = nmse(&state_b, &classic_state);
    println!(
        "fused-prefill t {t} split-carry ({split}+{}) vs classic: out nmse \
         {split_nmse:.3e}, state nmse {split_state_nmse:.3e}",
        t - split
    );
    assert!(
        split_nmse <= 1e-4,
        "t {t}: split out {split_nmse:.3e} beyond the BundleBf16 quantization class"
    );
    assert!(
        split_state_nmse <= 1e-4,
        "t {t}: split state {split_state_nmse:.3e} beyond the BundleBf16 quantization class"
    );
}

/// The rule kernels against the classic chain. The t = 150 leg (two
/// full 64-chunks + a ragged 22, the standing seed) keeps tile-64;
/// the t = 32 and t = 17 legs ride the tile-32 module (SmallT), the
/// speculation verify/readvance span class, split legs included.
#[test]
#[ignore = "needs a cuda card"]
fn fused_chunk_rule_matches_the_classic_chain() {
    chunk_rule_leg(150, 70, 0x0102_0304_0506_0708);
    chunk_rule_leg(32, 15, 0x0102_0304_0506_0709);
    chunk_rule_leg(17, 8, 0x0102_0304_0506_070A);
}

/// One PrepChunk end-to-end leg at a given t: the fused pipeline
/// (prep kernel + rule back half) against the classic
/// conv_with_tail / l2 / gates / chunk_rule chain, with a
/// carried-tail split leg. The prep kernel runs the conv/silu chain
/// in f32 where the classic chain rounds through bf16 per op, so the
/// out/state bars ride the bf16-upgrade class (the FusedDecodeConv
/// precedent); the tail is classic-exact (the same bf16 round-trip).
fn prep_chunk_leg(t: usize, split: usize, seed_base: u64) {
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let mut seed = seed_base;
    let (heads, dk, dv) = (30usize, 96usize, 192usize);
    let key_width = heads * dk;
    let value_width = heads * dv;
    let conv_width = 2 * key_width + value_width;

    let conv_in = synth2(&mut seed, (t, conv_width), &device)
        .to_dtype(candle_core::DType::BF16)
        .expect("conv_in bf16");
    let a_rows = synth2(&mut seed, (t, heads), &device)
        .to_dtype(candle_core::DType::BF16)
        .expect("a bf16");
    let b_rows = synth2(&mut seed, (t, heads), &device)
        .to_dtype(candle_core::DType::BF16)
        .expect("b bf16");
    let weight = synth3(&mut seed, (conv_width, 1, 4), &device)
        .to_dtype(candle_core::DType::BF16)
        .expect("weight bf16");
    let a_log = synth2(&mut seed, (1, heads), &device)
        .squeeze(0)
        .expect("a_log shape")
        .to_dtype(candle_core::DType::BF16)
        .expect("a_log bf16");
    let dt_bias = synth2(&mut seed, (1, heads), &device)
        .squeeze(0)
        .expect("dt shape")
        .to_dtype(candle_core::DType::BF16)
        .expect("dt bf16");
    let conv_tail = synth2(&mut seed, (3, conv_width), &device);
    let state_0 = synth3(&mut seed, (heads, dk, dv), &device);

    // The layer's static-rows assembly.
    let weight_rows = weight
        .squeeze(1)
        .expect("squeeze")
        .transpose(0, 1)
        .expect("transpose")
        .contiguous()
        .expect("weight rows");
    let head_pad = candle_core::Tensor::cat(
        &[
            &a_log.reshape((1, heads)).expect("a_log row"),
            &dt_bias.reshape((1, heads)).expect("dt row"),
        ],
        1,
    )
    .expect("head pad");
    let zero_pad = candle_core::Tensor::zeros(
        (3, 2 * heads),
        candle_core::DType::BF16,
        &device,
    )
    .expect("zero pad");
    let pad_column = candle_core::Tensor::cat(&[&head_pad, &zero_pad], 0).expect("pad col");
    let static_rows = candle_core::Tensor::cat(&[&weight_rows, &pad_column], 1)
        .expect("static cat")
        .contiguous()
        .expect("static rows");

    // The classic reference chain (the layer's prep path verbatim).
    let (conv_out, tail_ref) =
        conv_with_tail(&conv_in, &weight, &conv_tail).expect("classic conv");
    let heads_of = |x: &candle_core::Tensor, offset: usize, width: usize, dim: usize| {
        x.narrow(1, offset, width)
            .expect("narrow")
            .contiguous()
            .expect("contiguous")
            .reshape((t, heads, dim))
            .expect("reshape")
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
    };
    let q = l2_norm(&heads_of(&conv_out, 0, key_width, dk)).expect("q norm");
    let q = (q * (dk as f64).powf(-0.5)).expect("q scale");
    let k = l2_norm(&heads_of(&conv_out, key_width, key_width, dk)).expect("k norm");
    let v = heads_of(&conv_out, 2 * key_width, value_width, dv);
    let (g, beta) = gdn_gates(&a_rows, &b_rows, &a_log, &dt_bias, true).expect("gates");
    let (out_ref, state_ref) = chunk_rule(&q, &k, &v, &g, &beta, &state_0).expect("classic");

    // Fused whole-pass.
    let (out, state_out, tail_out) = prep_chunk_rule(
        PrepInputs {
            conv_in: &conv_in,
            a_rows: &a_rows,
            b_rows: &b_rows,
            static_rows: &static_rows,
            conv_tail: &conv_tail,
            state: &state_0,
        },
        (dk as f64).powf(-0.5),
        true,
    )
    .expect("fused");
    let out_nmse = nmse(&out, &out_ref);
    let state_nmse = nmse(&state_out, &state_ref);
    let tail_nmse = nmse(&tail_out, &tail_ref);
    println!(
        "prep t {t} whole-pass vs classic: out nmse {out_nmse:.3e}, state nmse \
         {state_nmse:.3e}, tail nmse {tail_nmse:.3e}"
    );
    assert!(
        out_nmse <= 1e-4,
        "t {t}: out {out_nmse:.3e} beyond the bf16-upgrade class"
    );
    assert!(
        state_nmse <= 1e-4,
        "t {t}: state {state_nmse:.3e} beyond the bf16-upgrade class"
    );
    assert!(
        tail_nmse <= 1e-12,
        "t {t}: tail {tail_nmse:.3e} must be classic-exact"
    );

    // Split with tail AND state carried through the fused path
    // against the classic whole pass.
    let front = |x: &candle_core::Tensor| x.narrow(0, 0, split).unwrap();
    let back = |x: &candle_core::Tensor| x.narrow(0, split, t - split).unwrap();
    let front_conv = front(&conv_in);
    let front_a = front(&a_rows);
    let front_b = front(&b_rows);
    let (out_a, state_a, tail_a) = prep_chunk_rule(
        PrepInputs {
            conv_in: &front_conv,
            a_rows: &front_a,
            b_rows: &front_b,
            static_rows: &static_rows,
            conv_tail: &conv_tail,
            state: &state_0,
        },
        (dk as f64).powf(-0.5),
        true,
    )
    .expect("fused split a");
    let back_conv = back(&conv_in);
    let back_a = back(&a_rows);
    let back_b = back(&b_rows);
    let (out_b, state_b, _) = prep_chunk_rule(
        PrepInputs {
            conv_in: &back_conv,
            a_rows: &back_a,
            b_rows: &back_b,
            static_rows: &static_rows,
            conv_tail: &tail_a,
            state: &state_a,
        },
        (dk as f64).powf(-0.5),
        true,
    )
    .expect("fused split b");
    let split_out = candle_core::Tensor::cat(&[&out_a, &out_b], 0).expect("cat");
    let split_nmse = nmse(&split_out, &out_ref);
    let split_state_nmse = nmse(&state_b, &state_ref);
    println!(
        "prep t {t} split-carry ({split}+{}) vs classic: out nmse {split_nmse:.3e}, \
         state nmse {split_state_nmse:.3e}",
        t - split
    );
    assert!(
        split_nmse <= 1e-4,
        "t {t}: split out {split_nmse:.3e} beyond the bf16-upgrade class"
    );
    assert!(
        split_state_nmse <= 1e-4,
        "t {t}: split state {split_state_nmse:.3e} beyond the bf16-upgrade class"
    );
}

/// The PrepChunk end-to-end lock. The t = 150 leg keeps the standing
/// seed and tile-64; the t = 32 and t = 17 legs ride the tile-32
/// module (SmallT), split-carry included.
#[test]
#[ignore = "needs a cuda card"]
fn prep_chunk_rule_matches_the_classic_chain() {
    prep_chunk_leg(150, 70, 0x5EED_00AA_BB01_F00D);
    prep_chunk_leg(32, 15, 0x5EED_00AA_BB01_F00E);
    prep_chunk_leg(17, 8, 0x5EED_00AA_BB01_F00F);
}
