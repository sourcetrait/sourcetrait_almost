//! The D2 gate: the stateless forward replayed over the lastmost C2
//! short cpu-f32 single-mode reference ids, compared in-process.
//! Bars: nmse (f64 accumulation) <= 1e-9 AND per-row argmax agreement
//! on every row. #[ignore] - needs the local DPO checkpoint plus
//! ALMOST_D2_REFERENCE pointing at the reference safetensors.
use crate::*;

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

#[test]
#[ignore = "needs the DPO checkpoint + ALMOST_D2_REFERENCE"]
fn d2_gate_matches_c2_short_cpu_f32_single() {
    let reference_path = env::var("ALMOST_D2_REFERENCE").expect(
        "ALMOST_D2_REFERENCE must point at short_cpu_f32_torch_eager_single.safetensors",
    );
    let bytes = fs::read(&reference_path).expect("reference file reads");
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("reference parses");

    let prompt_ids = u32_field(&tensors, "prompt_ids");
    let fed_ids = u32_field(&tensors, "fed_ids");
    let (reference_logits, logits_shape) = f32_field(&tensors, "logits");
    let all_ids: Vec<u32> = prompt_ids.iter().chain(fed_ids.iter()).copied().collect();
    let rows = all_ids.len();
    assert_eq!(logits_shape, [rows, 100352], "reference logits shape");

    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("config parses + validates");
    let weights = mmap_weights(&dir, candle_core::DType::F32, &candle_core::Device::Cpu)
        .expect("weights mmap");
    let model = OlmoHybrid::new(&config, weights).expect("model builds");

    let input = candle_core::Tensor::from_vec(all_ids, rows, &candle_core::Device::Cpu)
        .expect("input ids");
    let ours = model.forward_all(&input).expect("forward_all");
    assert_eq!(ours.dims(), [rows, config.vocab_size]);
    let ours: Vec<f32> = ours
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("f32 logits");

    let vocab = config.vocab_size;
    let mut numerator = 0f64;
    let mut denominator = 0f64;
    let mut argmax_hits = 0usize;
    for (ours_row, reference_row) in ours.chunks_exact(vocab).zip(reference_logits.chunks_exact(vocab)) {
        for (a, b) in ours_row.iter().zip(reference_row) {
            let difference = (*a as f64) - (*b as f64);
            numerator += difference * difference;
            denominator += (*b as f64) * (*b as f64);
        }
        if argmax(ours_row) == argmax(reference_row) {
            argmax_hits += 1;
        }
    }
    let nmse = numerator / denominator;
    println!("d2 gate: nmse {nmse:.3e}, argmax {argmax_hits}/{rows}");
    assert_eq!(argmax_hits, rows, "argmax agreement (nmse {nmse:.3e})");
    assert!(nmse <= 1e-9, "nmse {nmse:.3e} exceeds the 1e-9 bar");
}
