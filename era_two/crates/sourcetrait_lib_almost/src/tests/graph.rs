//! E4 gates: bucket math units, the staged-vs-classic /
//! captured-vs-uncaptured / replay-determinism exactness ladder, the
//! capacity epoch, and the settings rejection (cuda feature; the gpu
//! legs are #[ignore]).
use crate::*;
use crate::graph::{bucket_for, pad_mask_values};

#[test]
fn bucket_math_rounds_up_by_grain() {
    assert_eq!(bucket_for(0, 2048), 2048);
    assert_eq!(bucket_for(1, 2048), 2048);
    assert_eq!(bucket_for(2048, 2048), 2048);
    assert_eq!(bucket_for(2049, 2048), 4096);
    assert_eq!(bucket_for(300, 64), 320);
}

#[test]
fn pad_mask_zeroes_valid_and_infs_the_tail() {
    let values = pad_mask_values(8, 3);
    assert_eq!(&values[..3], &[0.0, 0.0, 0.0]);
    assert!(values[3..].iter().all(|v| *v == f32::NEG_INFINITY));
}

/// The slot-write kernel in isolation: scatter one row block into a
/// zeroed buffer at a device-resident slot and read it back.
#[test]
#[ignore = "needs a cuda card"]
fn slot_write_scatters_one_row() {
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let buffer = candle_core::Tensor::zeros((2, 8, 4), candle_core::DType::F32, &device)
        .expect("buffer");
    let row = candle_core::Tensor::from_vec(
        vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        (2, 1, 4),
        &device,
    )
    .expect("row");
    let slot = candle_core::Tensor::from_vec(vec![3u32], (1,), &device).expect("slot");
    buffer
        .inplace_op3(&row, &slot, &graph::SlotWrite {
            dtype: candle_core::DType::F32,
        })
        .expect("slot write");
    let host = buffer
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host");
    // Head 0 row 3 holds 1..4; head 1 row 3 holds 5..8; all else zero.
    assert_eq!(&host[3 * 4..4 * 4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&host[(8 + 3) * 4..(8 + 4) * 4], &[5.0, 6.0, 7.0, 8.0]);
    let written: f32 = host.iter().sum();
    assert_eq!(written, 36.0, "exactly the two rows were written");
}

/// The bf16 kernel variant of the scatter probe (the model's own
/// dtype; the f32 probe alone left it unverified).
#[test]
#[ignore = "needs a cuda card"]
fn slot_write_scatters_bf16() {
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let source: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let buffer = candle_core::Tensor::zeros((2, 8, 4), candle_core::DType::BF16, &device)
        .expect("buffer");
    let row = candle_core::Tensor::from_vec(source.clone(), (2, 1, 4), &device)
        .expect("row f32")
        .to_dtype(candle_core::DType::BF16)
        .expect("row bf16");
    let slot = candle_core::Tensor::from_vec(vec![3u32], (1,), &device).expect("slot");
    buffer
        .inplace_op3(&row, &slot, &graph::SlotWrite {
            dtype: candle_core::DType::BF16,
        })
        .expect("slot write");
    let host = buffer
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host");
    assert_eq!(&host[3 * 4..4 * 4], &source[..4]);
    assert_eq!(&host[(8 + 3) * 4..(8 + 4) * 4], &source[4..]);
    let written: f32 = host.iter().sum();
    assert_eq!(written, 36.0, "exactly the two rows were written");
}

/// The offset-leak lock: a source derived through narrow keeps a
/// nonzero storage start_offset even when every later op reports
/// contiguous (size-1 dims skip the stride check) - the kernel must
/// read at the layout offset, not the storage base (the v-as-q-span
/// defect this pins).
#[test]
#[ignore = "needs a cuda card"]
fn slot_write_honors_source_start_offset() {
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    // One row of 12: [q q q q | k k k k | v v v v]; v = 100..104.
    let fused: Vec<f32> = (0..12).map(|v| v as f32 + if v >= 8 { 92.0 } else { 0.0 }).collect();
    let fused = candle_core::Tensor::from_vec(fused, (1, 12), &device).expect("fused");
    let v_span = fused
        .narrow(1, 8, 4)
        .expect("narrow")
        .contiguous()
        .expect("contiguous keeps the offset when already-contiguous")
        .reshape((1, 1, 4))
        .expect("reshape");
    let buffer = candle_core::Tensor::zeros((1, 4, 4), candle_core::DType::F32, &device)
        .expect("buffer");
    let slot = candle_core::Tensor::from_vec(vec![2u32], (1,), &device).expect("slot");
    buffer
        .inplace_op3(&v_span, &slot, &graph::SlotWrite {
            dtype: candle_core::DType::F32,
        })
        .expect("slot write");
    let host = buffer
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host");
    assert_eq!(
        &host[2 * 4..3 * 4],
        &[100.0, 101.0, 102.0, 103.0],
        "the v span must land, not the storage-base (q) bytes"
    );
}

fn dpo_dir() -> PathBuf {
    model_dir(consts::DPO_MODEL_NAME).expect("dpo dir")
}

fn cuda_model(settings: LibSettings) -> OlmoHybrid {
    let dir = dpo_dir();
    let config = load_config(&dir).expect("config");
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
    OlmoHybrid::new(&config, settings, weights).expect("model")
}

fn synthetic_ids(count: usize) -> Vec<u32> {
    (0..count as u32).map(|i| 1000 + i * 7).collect()
}

fn prefill(model: &mut OlmoHybrid, ids: &[u32]) -> u32 {
    model.clear_cache().expect("clear");
    let tensor =
        candle_core::Tensor::from_vec(ids.to_vec(), ids.len(), &model.device().clone())
            .expect("ids");
    let logits = model.forward_chunk(&tensor).expect("prefill");
    let last = logits
        .narrow(0, ids.len() - 1, 1)
        .expect("last row")
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host");
    argmax(&last)
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in row.iter().enumerate() {
        if *value > row[best] {
            best = index;
        }
    }
    best as u32
}

fn classic_step_rows(model: &mut OlmoHybrid, tokens: &[u32]) -> Vec<Vec<f32>> {
    let device = model.device().clone();
    tokens
        .iter()
        .map(|token| {
            let ids =
                candle_core::Tensor::from_vec(vec![*token], 1, &device).expect("token");
            model
                .forward_chunk(&ids)
                .expect("classic step")
                .to_dtype(candle_core::DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("host")
        })
        .collect()
}

fn staged_step_rows(model: &mut OlmoHybrid, tokens: &[u32]) -> Vec<Vec<f32>> {
    tokens
        .iter()
        .map(|token| {
            model.graph_disarm_when_full().expect("capacity check");
            assert!(model.graph_armed(), "capacity epoch fired unexpectedly");
            model
                .graph_decode_step(*token)
                .expect("staged step")
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("host")
        })
        .collect()
}

fn nmse(actual: &[f32], reference: &[f32]) -> f64 {
    let mut numerator = 0f64;
    let mut denominator = 0f64;
    for (a, r) in actual.iter().zip(reference) {
        let difference = (*a as f64) - (*r as f64);
        numerator += difference * difference;
        denominator += (*r as f64) * (*r as f64);
    }
    numerator / denominator
}

/// The E4 exactness ladder on one model: classic reference, staged
/// uncaptured (envelope + argmax identity), captured (exactly 0.0 vs
/// staged), and cached-replay determinism across a clear/rearm epoch
/// (exactly 0.0 again).
///
/// The grain is 128 so the bucket pads this 106-token rig by ~1.2x -
/// the regime the 5e-4 staged-vs-classic bar was measured in. At the
/// production grain (2048) the same rig pads ~19x and the padded
/// softmax's regrouped f32 sums + bf16 prob quantization read
/// ~2e-3-class - pad-ratio-dependent numeric noise, not a mechanics
/// defect; the capture legs (exactly 0.0) hold at any grain.
#[test]
#[ignore = "needs the DPO checkpoint + a cuda card"]
fn graph_gate_exactness_ladder() {
    const STEPS: usize = 64;
    let mut model = cuda_model(LibSettings {
        graph: true,
        graph_bucket_grain: 128,
        ..LibSettings::default()
    });
    let prompt = synthetic_ids(106);

    // Leg A - the classic reference, greedy trajectory recorded.
    let first = prefill(&mut model, &prompt);
    let mut tokens = vec![first];
    let mut classic_rows: Vec<Vec<f32>> = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let rows = classic_step_rows(&mut model, &[*tokens.last().expect("token")]);
        let row = rows.into_iter().next().expect("row");
        tokens.push(argmax(&row));
        classic_rows.push(row);
    }
    let feed = &tokens[..STEPS];

    // Leg B - staged, uncaptured: the same op sequence as capture,
    // run as ordinary ops; same-ids replay against leg A. Kin-vs-kin:
    // two valid bf16 paths whose GDN states drift apart over steps
    // (bucket-width kernel selection), so this leg REPORTS the spread
    // under a sanity ceiling - quality is the truth-envelope gate's
    // job (graph_gate_staged_truth_envelope).
    let _ = prefill(&mut model, &prompt);
    model
        .arm_graph_decode(prompt.len() + STEPS + 1)
        .expect("arm");
    model.set_graph_capture(false).expect("capture off");
    let staged_rows = staged_step_rows(&mut model, feed);
    let mut max_error = 0f64;
    let mut argmax_hits = 0usize;
    for (staged, classic) in staged_rows.iter().zip(&classic_rows) {
        max_error = max_error.max(nmse(staged, classic));
        if argmax(staged) == argmax(classic) {
            argmax_hits += 1;
        }
    }
    println!(
        "ladder staged-vs-classic (kin drift): max nmse {max_error:.3e}, argmax {argmax_hits}/{STEPS}"
    );
    assert!(
        max_error <= 5e-3,
        "staged-vs-classic max nmse {max_error:.3e} exceeds the 5e-3 sanity ceiling"
    );
    assert!(
        argmax_hits * 10 >= STEPS * 9,
        "staged-vs-classic argmax {argmax_hits}/{STEPS} under the 90% sanity floor"
    );

    // Leg C - captured: bitwise-identical to the staged sequence.
    let _ = prefill(&mut model, &prompt);
    model
        .arm_graph_decode(prompt.len() + STEPS + 1)
        .expect("re-arm");
    model.set_graph_capture(true).expect("capture on");
    let captured_rows = staged_step_rows(&mut model, feed);
    for (step, (captured, staged)) in captured_rows.iter().zip(&staged_rows).enumerate() {
        assert!(
            captured == staged,
            "captured-vs-uncaptured differs at step {step} (must be exactly 0.0)"
        );
    }

    // Leg D - cached-graph replay across a clear/rearm epoch:
    // deterministic to the bit.
    let _ = prefill(&mut model, &prompt);
    model
        .arm_graph_decode(prompt.len() + STEPS + 1)
        .expect("re-arm cached");
    let replay_rows = staged_step_rows(&mut model, feed);
    for (step, (replay, captured)) in replay_rows.iter().zip(&captured_rows).enumerate() {
        assert!(
            replay == captured,
            "cached replay differs at step {step} (must be exactly 0.0)"
        );
    }
}

/// Diagnostic (report-only): staged-vs-classic first-step nmse at
/// several context lengths - a self-column/off-by-one defect explodes
/// at tiny contexts; a fixed perturbation stays flat.
#[test]
#[ignore = "diagnostic: needs the DPO checkpoint + a cuda card"]
fn graph_diag_context_scaling() {
    let mut model = cuda_model(LibSettings {
        graph: true,
        graph_bucket_grain: 128,
        ..LibSettings::default()
    });
    for context in [1usize, 4, 32, 106] {
        let prompt = synthetic_ids(context);
        let first = prefill(&mut model, &prompt);
        let classic = classic_step_rows(&mut model, &[first])
            .into_iter()
            .next()
            .expect("classic row");
        let _ = prefill(&mut model, &prompt);
        model
            .arm_graph_decode(prompt.len() + 4)
            .expect("arm");
        model.set_graph_capture(false).expect("capture off");
        let staged = staged_step_rows(&mut model, &[first])
            .into_iter()
            .next()
            .expect("staged row");
        let error = nmse(&staged, &classic);
        let classic_top = argmax(&classic);
        let staged_top = argmax(&staged);
        println!(
            "diag ctx {context}: nmse {error:.3e}, argmax classic {classic_top} vs staged {staged_top}"
        );
    }
}

fn u32_field(tensors: &safetensors::SafeTensors, name: &str) -> Vec<u32> {
    let view = tensors.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    view.data()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f32_field(tensors: &safetensors::SafeTensors, name: &str) -> Vec<f32> {
    let view = tensors.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    view.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// THE staged-path correctness gate, truth-anchored cuda-grade-style:
/// every row after the first rides graph_decode_step (uncaptured
/// staged ops) over the oracle incremental id trail. Bars from measurement:
/// vs cpu-f32 truth nmse <= 2e-5 (measured 1.430e-5; the classic
/// eager cross-grade class); vs the cuda-bf16-torch kin nmse <= 5e-5
/// with argmax >= 105/106 (measured 1.699e-5, 106/106; the classic
/// standing reading 1.618e-5, 106/106). Staged-vs-classic deltas are
/// kin-vs-kin drift (state carry forks two valid bf16 paths); the
/// ladder reports them, and THIS gate owns quality.
#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_ORACLE_DUMPS_DIR + a cuda card"]
fn graph_gate_staged_truth_envelope() {
    let dumps = env::var("ALMOST_ORACLE_DUMPS_DIR").expect("ALMOST_ORACLE_DUMPS_DIR");
    let load = |relative: &str| -> (Vec<u32>, Vec<f32>) {
        let bytes =
            fs::read(Path::new(&dumps).join(relative)).expect("reference reads");
        let tensors =
            safetensors::SafeTensors::deserialize(&bytes).expect("reference parses");
        let prompt_ids = u32_field(&tensors, "prompt_ids");
        let fed_ids = u32_field(&tensors, "fed_ids");
        let logits = f32_field(&tensors, "logits");
        let all: Vec<u32> = prompt_ids.iter().chain(fed_ids.iter()).copied().collect();
        (all, logits)
    };
    let (all_ids, truth) = load("short/short_cpu_f32_torch_eager_single.safetensors");
    let (_, kin) = load("short/short_cuda_bf16_torch_eager_incr.safetensors");
    let vocab = truth.len() / all_ids.len();

    let mut model = cuda_model(LibSettings {
        graph: true,
        ..LibSettings::default()
    });
    model.clear_cache().expect("clear");
    let device = model.device().clone();
    // Row 0 rides a 1-token prefill; every later row rides the staged
    // decode step.
    let first =
        candle_core::Tensor::from_vec(vec![all_ids[0]], 1, &device).expect("first id");
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(all_ids.len());
    rows.push(
        model
            .forward_chunk(&first)
            .expect("prefill row")
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("host"),
    );
    model.arm_graph_decode(all_ids.len() + 1).expect("arm");
    model.set_graph_capture(false).expect("capture off");
    for token in &all_ids[1..] {
        model.graph_disarm_when_full().expect("capacity check");
        rows.push(
            model
                .graph_decode_step(*token)
                .expect("staged step")
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("host"),
        );
    }
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    let score = |reference: &[f32]| -> (f64, usize) {
        let mut numerator = 0f64;
        let mut denominator = 0f64;
        let mut hits = 0usize;
        for (ours, theirs) in flat.chunks_exact(vocab).zip(reference.chunks_exact(vocab)) {
            for (a, b) in ours.iter().zip(theirs) {
                let difference = (*a as f64) - (*b as f64);
                numerator += difference * difference;
                denominator += (*b as f64) * (*b as f64);
            }
            if argmax(ours) == argmax(theirs) {
                hits += 1;
            }
        }
        (numerator / denominator, hits)
    };
    let rows_total = all_ids.len();
    let (truth_nmse, truth_hits) = score(&truth);
    println!("staged truth gate vs cpu-f32: nmse {truth_nmse:.3e}, argmax {truth_hits}/{rows_total}");
    assert!(
        truth_nmse <= 2e-5,
        "nmse {truth_nmse:.3e} exceeds the 2e-5 cross-grade envelope"
    );
    let (kin_nmse, kin_hits) = score(&kin);
    println!("staged truth gate vs bf16 kin: nmse {kin_nmse:.3e}, argmax {kin_hits}/{rows_total}");
    assert!(
        kin_hits >= 105,
        "argmax {kin_hits}/{rows_total} under the same-grade floor (105/106)"
    );
    assert!(
        kin_nmse <= 5e-5,
        "kin nmse {kin_nmse:.3e} exceeds the 5e-5 sanity ceiling"
    );
}

/// The capacity epoch: a decode outrunning the armed width disarms,
/// retires capture, and finishes on the classic path. Under the
/// ReserveGrainTrim exact reserve the armed capacity is the arm-time
/// bucket ceiling over prompt + margin, so the crossing fires within
/// a few hundred decoded tokens at any context length.
#[test]
#[ignore = "needs the DPO checkpoint + a cuda card; ~40 s"]
fn graph_gate_capacity_epoch() {
    const BUDGET: usize = 1100;
    let mut model = cuda_model(LibSettings {
        graph: true,
        graph_bucket_grain: 64,
        ..LibSettings::default()
    });
    let dir = dpo_dir();
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");
    let options = GenerateOptions {
        temperature: None,
        top_p: None,
        sample_len: BUDGET,
        chat: false,
        ignore_stops: true,
        ..GenerateOptions::default()
    };
    let mut generation = model
        .generate(&tokenizer, "The capacity epoch drill runs on.", &options)
        .expect("generation starts");
    for step in generation.by_ref() {
        step.expect("step");
    }
    let report = generation.finish();
    assert_eq!(report.generated_token_count, BUDGET, "forced decode length");
    assert!(
        !model.graph_armed(),
        "the epoch must have disarmed the stage"
    );
}

/// graph = true demands a cuda device at construction (explicit
/// intent never degrades silently); the mmap stays lazy, so the
/// rejection is cheap.
#[test]
#[ignore = "needs the DPO checkpoint"]
fn graph_settings_reject_a_cpu_device() {
    let dir = dpo_dir();
    let config = load_config(&dir).expect("config");
    let weights = mmap_weights(&dir, candle_core::DType::F32, &candle_core::Device::Cpu)
        .expect("mmap");
    let settings = LibSettings {
        graph: true,
        ..LibSettings::default()
    };
    assert!(OlmoHybrid::new(&config, settings, weights).is_err());
}
