//! The D-slice gates: in-process replays of the lastmost C2 cpu-f32
//! references (all #[ignore] - they need the local DPO checkpoint and
//! a reference env var), plus the chunked-vs-stateless equivalence.
//!
//! Env: ALMOST_D2_REFERENCE points at the short single-mode reference
//! file (the D2 gate); ALMOST_C2_DUMPS_DIR points at the C2 dumps root
//! (short/mid/long subdirs - the D3 gates). Bars: f32 cross-stack nmse
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

fn build_model() -> OlmoHybrid {
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("config parses + validates");
    let weights = mmap_weights(&dir, candle_core::DType::F32, &candle_core::Device::Cpu)
        .expect("weights mmap");
    OlmoHybrid::new(&config, weights).expect("model builds")
}

fn ids_tensor(ids: &[u32]) -> candle_core::Tensor {
    candle_core::Tensor::from_vec(ids.to_vec(), ids.len(), &candle_core::Device::Cpu)
        .expect("ids tensor")
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
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("f32 logits")
}

/// All-position logits via the carried path at the given chunk size.
fn chunked_rows(model: &mut OlmoHybrid, ids: &[u32], chunk: usize) -> Vec<f32> {
    model.clear_cache().expect("clear");
    let mut parts = Vec::new();
    let mut start = 0;
    while start < ids.len() {
        let len = chunk.min(ids.len() - start);
        let logits = model
            .forward_chunk(&ids_tensor(&ids[start..start + len]))
            .expect("forward_chunk");
        parts.push(logits);
        start += len;
    }
    assert_eq!(model.context_len(), ids.len());
    flat(candle_core::Tensor::cat(&parts, 0).expect("cat rows"))
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_D2_REFERENCE"]
fn d2_gate_matches_c2_short_cpu_f32_single() {
    let reference_path = env::var("ALMOST_D2_REFERENCE").expect(
        "ALMOST_D2_REFERENCE must point at short_cpu_f32_torch_eager_single.safetensors",
    );
    let (all_ids, reference) = load_reference(&reference_path);
    let rows = all_ids.len();
    let model = build_model();
    let ours = model.forward_all(&ids_tensor(&all_ids)).expect("forward_all");
    assert_eq!(ours.dims(), [rows, VOCAB]);
    let (nmse, argmax_hits) = score(&flat(ours), &reference);
    println!("d2 gate: nmse {nmse:.3e}, argmax {argmax_hits}/{rows}");
    assert_eq!(argmax_hits, rows, "argmax agreement (nmse {nmse:.3e})");
    assert!(nmse <= 1e-9, "nmse {nmse:.3e} exceeds the 1e-9 bar");
}

fn c2_dumps_dir() -> String {
    env::var("ALMOST_C2_DUMPS_DIR")
        .expect("ALMOST_C2_DUMPS_DIR must point at the C2 dumps root (short/mid/long)")
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_C2_DUMPS_DIR"]
fn d3_gate_chunked_matches_stateless() {
    let reference_path = Path::new(&c2_dumps_dir())
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
        println!("d3 chunked-vs-stateless (chunk {chunk}): nmse {nmse:.3e}, argmax {argmax_hits}/{rows}");
        assert_eq!(argmax_hits, rows, "argmax at chunk {chunk} (nmse {nmse:.3e})");
        assert!(nmse <= 1e-12, "nmse {nmse:.3e} exceeds 1e-12 at chunk {chunk}");
    }
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_C2_DUMPS_DIR"]
fn d3_gate_matches_c2_short_incremental() {
    let reference_path = Path::new(&c2_dumps_dir())
        .join("short/short_cpu_f32_torch_eager_incr.safetensors");
    let (all_ids, reference) = load_reference(reference_path.to_str().expect("utf-8 path"));
    let rows = all_ids.len();
    let mut model = build_model();
    // Token-at-a-time: every row rides the decode path (the recurrent
    // rule + conv tails + mask-free single-query attention).
    let ours = chunked_rows(&mut model, &all_ids, 1);
    let (nmse, argmax_hits) = score(&ours, &reference);
    println!("d3 incremental gate: nmse {nmse:.3e}, argmax {argmax_hits}/{rows}");
    assert_eq!(argmax_hits, rows, "argmax agreement (nmse {nmse:.3e})");
    assert!(nmse <= 1e-9, "nmse {nmse:.3e} exceeds the 1e-9 bar");
}

fn length_reference_gate(relative: &str, label: &str) {
    let reference_path = Path::new(&c2_dumps_dir()).join(relative);
    let (all_ids, reference) = load_reference(reference_path.to_str().expect("utf-8 path"));
    let rows = all_ids.len();
    let mut model = build_model();
    let ours = chunked_rows(&mut model, &all_ids, 512);
    let (nmse, argmax_hits) = score(&ours, &reference);
    println!("d3 {label} gate: nmse {nmse:.3e}, argmax {argmax_hits}/{rows}");
    assert_eq!(argmax_hits, rows, "argmax agreement (nmse {nmse:.3e})");
    assert!(nmse <= 1e-9, "nmse {nmse:.3e} exceeds the 1e-9 bar");
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_C2_DUMPS_DIR"]
fn d3_gate_matches_c2_mid_single() {
    length_reference_gate("mid/mid_cpu_f32_torch_eager_single.safetensors", "mid (2K)");
}

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_C2_DUMPS_DIR; long runtime"]
fn d3_gate_matches_c2_long_single() {
    length_reference_gate("long/long_cpu_f32_torch_eager_single.safetensors", "long (8.5K)");
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
