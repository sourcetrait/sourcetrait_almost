//! The CorrectnessCore slice gates: in-process replays of the
//! lastmost oracle (C2) cpu-f32 references (all #[ignore] - they need
//! the local DPO checkpoint and a reference env var), plus the
//! chunked-vs-stateless equivalence.
//!
//! Env: ALMOST_STATELESS_REFERENCE points at the short single-mode reference
//! file (the StatelessGate); ALMOST_ORACLE_DUMPS_DIR points at the oracle
//! dumps root (short/mid/long subdirs - the StateCarry gates). Bars: f32 cross-stack nmse
//! <= 1e-9 with full argmax agreement (the oracle's own algorithmic
//! pin is 1.2e-13); chunked-vs-stateless <= 1e-12.
use crate::*;

const VOCAB: usize = 100352;

fn u32_field(tensors: &safetensors::SafeTensors, name: &str) -> Vec<u32> {
    let view = tensors.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::U32, "{name} dtype");
    view.data()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f32_field(tensors: &safetensors::SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let view = tensors.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name} dtype");
    let data = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (data, view.shape().to_vec())
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in row.iter().enumerate() {
        if *value > row[best] {
            best = index;
        }
    }
    best
}

/// A reference dump's replay ids ([prompt_ids ++ fed_ids]) and its
/// all-position logits, shape-checked.
fn load_reference(path: &str) -> (Vec<u32>, Vec<f32>) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reference {path}: {e}"));
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("reference parses");
    let prompt_ids = u32_field(&tensors, "prompt_ids");
    let fed_ids = u32_field(&tensors, "fed_ids");
    let (logits, shape) = f32_field(&tensors, "logits");
    let all_ids: Vec<u32> = prompt_ids.iter().chain(fed_ids.iter()).copied().collect();
    assert_eq!(shape, [all_ids.len(), VOCAB], "reference logits shape");
    (all_ids, logits)
}

fn build_model_on(device: &candle_core::Device, dtype: candle_core::DType) -> OlmoHybrid {
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("config parses + validates");
    let weights = mmap_weights(&dir, dtype, device).expect("weights mmap");
    OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model builds")
}

fn build_model() -> OlmoHybrid {
    build_model_on(&candle_core::Device::Cpu, candle_core::DType::F32)
}

fn ids_tensor_on(ids: &[u32], device: &candle_core::Device) -> candle_core::Tensor {
    candle_core::Tensor::from_vec(ids.to_vec(), ids.len(), device).expect("ids tensor")
}

fn ids_tensor(ids: &[u32]) -> candle_core::Tensor {
    ids_tensor_on(ids, &candle_core::Device::Cpu)
}

/// f64-accumulated nmse plus per-row argmax hits over flat logits.
fn score(ours: &[f32], reference: &[f32]) -> (f64, usize) {
    assert_eq!(ours.len(), reference.len());
    let mut numerator = 0f64;
    let mut denominator = 0f64;
    let mut argmax_hits = 0usize;
    for (ours_row, reference_row) in ours.chunks_exact(VOCAB).zip(reference.chunks_exact(VOCAB)) {
        for (a, b) in ours_row.iter().zip(reference_row) {
            let difference = (*a as f64) - (*b as f64);
            numerator += difference * difference;
            denominator += (*b as f64) * (*b as f64);
        }
        if argmax(ours_row) == argmax(reference_row) {
            argmax_hits += 1;
        }
    }
    (numerator / denominator, argmax_hits)
}

fn flat(logits: candle_core::Tensor) -> Vec<f32> {
    logits
        .to_dtype(candle_core::DType::F32)
        .expect("f32 cast")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("f32 logits")
}

/// All-position logits via the carried path at the given chunk size.
/// Each chunk's logits move to host f32 as produced, so the model
/// device never accumulates the full logit matrix.
fn chunked_rows(model: &mut OlmoHybrid, ids: &[u32], chunk: usize) -> Vec<f32> {
    model.clear_cache().expect("clear");
    let device = model.device().clone();
    let mut rows = Vec::with_capacity(ids.len() * VOCAB);
    let mut start = 0;
    while start < ids.len() {
        let len = chunk.min(ids.len() - start);
        let logits = model
            .forward_chunk(&ids_tensor_on(&ids[start..start + len], &device))
            .expect("forward_chunk");
        rows.extend(flat(logits));
        start += len;
    }
    assert_eq!(model.context_len(), ids.len());
    rows
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_STATELESS_REFERENCE"]
fn stateless_gate_matches_oracle_short_cpu_f32_single() {
    let reference_path = env::var("ALMOST_STATELESS_REFERENCE").expect(
        "ALMOST_STATELESS_REFERENCE must point at short_cpu_f32_torch_eager_single.safetensors",
    );
    let (all_ids, reference) = load_reference(&reference_path);
    let rows = all_ids.len();
    let model = build_model();
    let ours = model.forward_all(&ids_tensor(&all_ids)).expect("forward_all");
    assert_eq!(ours.dims(), [rows, VOCAB]);
    let (nmse, argmax_hits) = score(&flat(ours), &reference);
    println!("stateless gate: nmse {nmse:.3e}, argmax {argmax_hits}/{rows}");
    assert_eq!(argmax_hits, rows, "argmax agreement (nmse {nmse:.3e})");
    assert!(nmse <= 1e-9, "nmse {nmse:.3e} exceeds the 1e-9 bar");
}

fn oracle_dumps_dir() -> String {
    env::var("ALMOST_ORACLE_DUMPS_DIR")
        .expect("ALMOST_ORACLE_DUMPS_DIR must point at the oracle dumps root (short/mid/long)")
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_ORACLE_DUMPS_DIR"]
fn state_carry_gate_chunked_matches_stateless() {
    let reference_path = Path::new(&oracle_dumps_dir())
        .join("short/short_cpu_f32_torch_eager_single.safetensors");
    let (all_ids, _) = load_reference(reference_path.to_str().expect("utf-8 path"));
    let rows = all_ids.len();
    let mut model = build_model();
    let stateless = flat(model.forward_all(&ids_tensor(&all_ids)).expect("forward_all"));
    // One chunk covering everything, the rule-aligned 64, and a ragged
    // size that never aligns with the internal chunk.
    for chunk in [128usize, 64, 33] {
        let ours = chunked_rows(&mut model, &all_ids, chunk);
        let (nmse, argmax_hits) = score(&ours, &stateless);
        println!("state carry chunked-vs-stateless (chunk {chunk}): nmse {nmse:.3e}, argmax {argmax_hits}/{rows}");
        assert_eq!(argmax_hits, rows, "argmax at chunk {chunk} (nmse {nmse:.3e})");
        assert!(nmse <= 1e-12, "nmse {nmse:.3e} exceeds 1e-12 at chunk {chunk}");
    }
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_ORACLE_DUMPS_DIR"]
fn state_carry_gate_matches_oracle_short_incremental() {
    let reference_path = Path::new(&oracle_dumps_dir())
        .join("short/short_cpu_f32_torch_eager_incr.safetensors");
    let (all_ids, reference) = load_reference(reference_path.to_str().expect("utf-8 path"));
    let rows = all_ids.len();
    let mut model = build_model();
    // Token-at-a-time: every row rides the decode path (the recurrent
    // rule + conv tails + mask-free single-query attention).
    let ours = chunked_rows(&mut model, &all_ids, 1);
    let (nmse, argmax_hits) = score(&ours, &reference);
    println!("state carry incremental gate: nmse {nmse:.3e}, argmax {argmax_hits}/{rows}");
    assert_eq!(argmax_hits, rows, "argmax agreement (nmse {nmse:.3e})");
    assert!(nmse <= 1e-9, "nmse {nmse:.3e} exceeds the 1e-9 bar");
}

fn length_reference_gate(relative: &str, label: &str) {
    let reference_path = Path::new(&oracle_dumps_dir()).join(relative);
    let (all_ids, reference) = load_reference(reference_path.to_str().expect("utf-8 path"));
    let rows = all_ids.len();
    let mut model = build_model();
    let ours = chunked_rows(&mut model, &all_ids, 512);
    let (nmse, argmax_hits) = score(&ours, &reference);
    println!("state carry {label} gate: nmse {nmse:.3e}, argmax {argmax_hits}/{rows}");
    assert_eq!(argmax_hits, rows, "argmax agreement (nmse {nmse:.3e})");
    assert!(nmse <= 1e-9, "nmse {nmse:.3e} exceeds the 1e-9 bar");
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_ORACLE_DUMPS_DIR"]
fn state_carry_gate_matches_oracle_mid_single() {
    length_reference_gate("mid/mid_cpu_f32_torch_eager_single.safetensors", "mid (2K)");
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_ORACLE_DUMPS_DIR; long runtime"]
fn state_carry_gate_matches_oracle_long_single() {
    length_reference_gate("long/long_cpu_f32_torch_eager_single.safetensors", "long (8.5K)");
}

/// The CudaGrade envelope bars: nmse <= 2e-5 and argmax >= 99.8% of
/// rows (the lastmost oracle internal-spread envelope; their bf16 rows run
/// 1.2-1.7e-5 with 1-3 flips per dump).
#[cfg(feature = "cuda")]
fn envelope_gate(ours: &[f32], reference: &[f32], label: &str) {
    let rows = reference.len() / VOCAB;
    let (nmse, argmax_hits) = score(ours, reference);
    let argmax_bar = ((rows as f64) * 0.998).floor() as usize;
    println!("cuda grade {label}: nmse {nmse:.3e}, argmax {argmax_hits}/{rows} (bar {argmax_bar})");
    assert!(
        argmax_hits >= argmax_bar,
        "argmax {argmax_hits}/{rows} under the 99.8% bar (nmse {nmse:.3e})"
    );
    assert!(nmse <= 2e-5, "nmse {nmse:.3e} exceeds the 2e-5 envelope");
}

#[cfg(feature = "cuda")]
fn cuda_bf16_model() -> OlmoHybrid {
    build_model_on(
        &candle_core::Device::new_cuda(0).expect("cuda device"),
        candle_core::DType::BF16,
    )
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + ALMOST_ORACLE_DUMPS_DIR + a cuda card"]
fn cuda_grade_gate_short_single() {
    // SHORT-LENGTH BARS FROM MEASUREMENT (the era-one lesson, third
    // occurrence): at 106 rows cross-grade argmax is tie-dominated -
    // the ORIGINAL stack's own torch-cuda single reads 2.151e-5 nmse,
    // 104/106 vs its own cpu f32 (the untabulated control), and its
    // internal torch-vs-fla rows read 104-105/106. So: the f32
    // cross-grade holds the NMSE envelope (<= 2e-5; ours 1.957e-5,
    // CLOSER to f32 truth than their torch path) with argmax
    // informational, and the same-grade kin comparison carries the
    // argmax bar at their measured floor (>= 104/106). Argmax bars
    // with real statistical power live in the mid/long gates.
    let dumps = oracle_dumps_dir();
    let (all_ids, f32_reference) = load_reference(
        Path::new(&dumps)
            .join("short/short_cpu_f32_torch_eager_single.safetensors")
            .to_str()
            .expect("utf-8"),
    );
    let (cuda_ids, cuda_reference) = load_reference(
        Path::new(&dumps)
            .join("short/short_cuda_bf16_torch_eager_single.safetensors")
            .to_str()
            .expect("utf-8"),
    );
    assert_eq!(all_ids, cuda_ids, "the oracle dumps share one id trail");
    let model = cuda_bf16_model();
    let ours = flat(
        model
            .forward_all(&ids_tensor_on(&all_ids, model.device()))
            .expect("forward_all"),
    );
    let rows = all_ids.len();

    let (truth_nmse, truth_hits) = score(&ours, &f32_reference);
    println!(
        "cuda grade short single vs cpu-f32 (cross-grade): nmse {truth_nmse:.3e}, \
         argmax {truth_hits}/{rows} (informational at this length; their torch control: 2.151e-5, 104/106)"
    );
    assert!(
        truth_nmse <= 2e-5,
        "nmse {truth_nmse:.3e} exceeds the 2e-5 cross-grade envelope"
    );

    let (kin_nmse, kin_hits) = score(&ours, &cuda_reference);
    println!("cuda grade short single vs cuda-bf16-torch (kin): nmse {kin_nmse:.3e}, argmax {kin_hits}/{rows}");
    assert!(
        kin_hits >= 104,
        "argmax {kin_hits}/{rows} under the measured same-grade floor (104/106)"
    );
    assert!(
        kin_nmse <= 5e-5,
        "kin nmse {kin_nmse:.3e} exceeds the sanity ceiling (a third kernel family \
         sits farther from each sibling than their internal 1.64e-5; measured 2.81e-5)"
    );
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + ALMOST_ORACLE_DUMPS_DIR + a cuda card"]
fn cuda_grade_gate_short_incremental() {
    let reference_path = Path::new(&oracle_dumps_dir())
        .join("short/short_cuda_bf16_torch_eager_incr.safetensors");
    let (all_ids, reference) = load_reference(reference_path.to_str().expect("utf-8"));
    let mut model = cuda_bf16_model();
    let ours = chunked_rows(&mut model, &all_ids, 1);
    envelope_gate(&ours, &reference, "incremental vs cuda-bf16-torch");
}

#[cfg(feature = "cuda")]
fn cuda_length_gate(relative: &str, label: &str) {
    let reference_path = Path::new(&oracle_dumps_dir()).join(relative);
    let (all_ids, reference) = load_reference(reference_path.to_str().expect("utf-8"));
    let mut model = cuda_bf16_model();
    let ours = chunked_rows(&mut model, &all_ids, 512);
    envelope_gate(&ours, &reference, label);
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + ALMOST_ORACLE_DUMPS_DIR + a cuda card"]
fn cuda_grade_gate_mid_single() {
    cuda_length_gate(
        "mid/mid_cpu_f32_torch_eager_single.safetensors",
        "mid 2K vs cpu-f32 (cross-grade)",
    );
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + ALMOST_ORACLE_DUMPS_DIR + a cuda card"]
fn cuda_grade_gate_long_single() {
    cuda_length_gate(
        "long/long_cpu_f32_torch_eager_single.safetensors",
        "long 8.5K vs cpu-f32 (cross-grade)",
    );
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "diagnostic: short-dump pairwise spread (cuda + checkpoint + dumps)"]
fn cuda_grade_diag_short_pairwise_spread() {
    let dumps = oracle_dumps_dir();
    let reference_path = |relative: &str| -> String {
        Path::new(&dumps)
            .join(relative)
            .to_str()
            .expect("utf-8")
            .to_string()
    };
    let (all_ids, f32_reference) =
        load_reference(&reference_path("short/short_cpu_f32_torch_eager_single.safetensors"));
    let (_, torch_bf16) =
        load_reference(&reference_path("short/short_cuda_bf16_torch_eager_single.safetensors"));
    let (_, fla_bf16) =
        load_reference(&reference_path("short/short_cuda_bf16_fla_eager_single.safetensors"));
    let model = cuda_bf16_model();
    let ours = flat(
        model
            .forward_all(&ids_tensor_on(&all_ids, model.device()))
            .expect("forward_all"),
    );
    let report = |label: &str, a: &[f32], b: &[f32]| {
        let (nmse, hits) = score(a, b);
        println!("cuda grade diag {label}: nmse {nmse:.3e}, argmax {hits}/106");
    };
    report("their-fla   vs f32 (control; recorded 1.24e-5, 105/106)", &fla_bf16, &f32_reference);
    report("their-torch vs f32 (the gate-1 control, untabulated)", &torch_bf16, &f32_reference);
    report("their-torch vs their-fla (recorded 1.64e-5, 105/106)", &torch_bf16, &fla_bf16);
    report("ours        vs f32 (the failed gate)", &ours, &f32_reference);
    report("ours        vs their-torch (bf16 kin)", &ours, &torch_bf16);
    report("ours        vs their-fla (bf16 kin)", &ours, &fla_bf16);
    for (row, (ours_row, reference_row)) in ours
        .chunks_exact(VOCAB)
        .zip(f32_reference.chunks_exact(VOCAB))
        .enumerate()
    {
        let ours_top = argmax(ours_row);
        let reference_top = argmax(reference_row);
        if ours_top != reference_top {
            let margin = reference_row[reference_top] - reference_row[ours_top];
            println!(
                "cuda grade diag flip row {row}: ref top {reference_top} vs ours {ours_top}, f32 margin {margin:.5}"
            );
        }
    }
}

#[test]
#[ignore = "diagnostic probe: phase timings (needs the checkpoint only)"]
fn probe_phase_timings() {
    let build_start = std::time::Instant::now();
    let mut model = build_model();
    println!("probe: build_model {:.1}s", build_start.elapsed().as_secs_f64());
    let ids: Vec<u32> = (0..106u32).map(|i| 1000 + i * 7).collect();
    let stateless_start = std::time::Instant::now();
    model.forward_all(&ids_tensor(&ids)).expect("forward_all");
    println!(
        "probe: forward_all(106) {:.1}s",
        stateless_start.elapsed().as_secs_f64()
    );
    model.clear_cache().expect("clear");
    let chunk_start = std::time::Instant::now();
    model.forward_chunk(&ids_tensor(&ids)).expect("prefill chunk");
    println!(
        "probe: forward_chunk(106) {:.1}s",
        chunk_start.elapsed().as_secs_f64()
    );
    for step in 0..4u32 {
        let step_start = std::time::Instant::now();
        model.forward_chunk(&ids_tensor(&[2000 + step])).expect("decode step");
        println!(
            "probe: decode step {:.2}s",
            step_start.elapsed().as_secs_f64()
        );
    }
}
