//! BurnHybridForward parity gates: the burn oracle replayed against
//! the refquest C2 dumps (same-ids only). The cpu f32 legs are the
//! HARD gates; the cuda bf16 legs run loose sanity bars and REPORT
//! their readings - burn's one-float-type-per-backend means the
//! recurrence runs bf16 there, so those bars pin from measurement
//! (flagged) once the first readings exist.
use crate::*;

fn dumps_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("QUEST_ORACLE_DUMPS_DIR")
            .expect("set QUEST_ORACLE_DUMPS_DIR to the refquest C2 dumps directory"),
    )
}

fn checkpoint_dir() -> PathBuf {
    match std::env::var("QUEST_MODEL_DIR") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => crate::model_dir(crate::consts::DPO_MODEL_NAME).expect("the DPO checkpoint home"),
    }
}

struct DumpReplay {
    ids: Vec<u32>,
    rows: usize,
    vocab: usize,
    logits: Vec<f32>,
}

fn load_dump(relative: &str) -> DumpReplay {
    let path = dumps_dir().join(relative);
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let parsed = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let prompt_ids = dump_read_u32(&parsed, "prompt_ids").expect("prompt_ids");
    let fed_ids = dump_read_u32(&parsed, "fed_ids").expect("fed_ids");
    let (rows, vocab, logits) = dump_read_f32_matrix(&parsed, "logits").expect("logits");
    let mut ids = prompt_ids;
    ids.extend_from_slice(&fed_ids);
    assert_eq!(rows, ids.len(), "dump rows must equal prompt + fed ids");
    DumpReplay { ids, rows, vocab, logits }
}

/// f64-accumulated nmse + per-row argmax agreement.
fn compare(ours: &[f32], reference: &[f32], rows: usize, vocab: usize) -> (f64, usize) {
    assert_eq!(ours.len(), reference.len());
    let mut numerator = 0f64;
    let mut denominator = 0f64;
    let mut argmax_matches = 0usize;
    for row in 0..rows {
        let ours_row = &ours[row * vocab..(row + 1) * vocab];
        let reference_row = &reference[row * vocab..(row + 1) * vocab];
        let mut ours_best = 0usize;
        let mut reference_best = 0usize;
        for column in 0..vocab {
            let delta = ours_row[column] as f64 - reference_row[column] as f64;
            numerator += delta * delta;
            denominator += (reference_row[column] as f64) * (reference_row[column] as f64);
            if ours_row[column] > ours_row[ours_best] {
                ours_best = column;
            }
            if reference_row[column] > reference_row[reference_best] {
                reference_best = column;
            }
        }
        if ours_best == reference_best {
            argmax_matches += 1;
        }
    }
    (numerator / denominator, argmax_matches)
}

fn replay<B: burn::tensor::backend::Backend>(
    dump: &DumpReplay,
    device: B::Device,
) -> (f64, usize) {
    let model_dir = checkpoint_dir();
    let config = HybridCheckpointConfig::load(&model_dir).expect("config");
    let weights = HybridWeights::load(&model_dir).expect("weights");
    let model = HybridModel::<B>::new(&config, weights, device).expect("model build");
    let (rows, ours) = model.forward_all(&dump.ids).expect("forward");
    assert_eq!(rows, dump.rows);
    compare(&ours, &dump.logits, dump.rows, dump.vocab)
}

#[test]
#[ignore]
fn burn_forward_gate_short_cpu_f32() {
    let dump = load_dump("short/short_cpu_f32_torch_eager_single.safetensors");
    let (nmse, argmax_matches) = replay::<CpuBack>(&dump, Default::default());
    println!(
        "burn short cpu f32: nmse {nmse:.3e}, argmax {argmax_matches}/{}",
        dump.rows
    );
    assert!(nmse <= 1e-6, "nmse {nmse:.3e} over the 1e-6 f32 cross-stack bar");
    assert!(
        argmax_matches + 1 >= dump.rows,
        "argmax {argmax_matches}/{} under the rows-1 floor",
        dump.rows
    );
}

#[test]
#[ignore]
fn burn_forward_gate_mid_cpu_f32() {
    let dump = load_dump("mid/mid_cpu_f32_torch_eager_single.safetensors");
    let (nmse, argmax_matches) = replay::<CpuBack>(&dump, Default::default());
    println!(
        "burn mid cpu f32: nmse {nmse:.3e}, argmax {argmax_matches}/{}",
        dump.rows
    );
    assert!(nmse <= 1e-6, "nmse {nmse:.3e} over the 1e-6 f32 cross-stack bar");
    assert!(
        (argmax_matches as f64) / (dump.rows as f64) >= 0.998,
        "argmax {argmax_matches}/{} under the 99.8% floor",
        dump.rows
    );
}

#[cfg(feature = "burn-cuda")]
#[test]
#[ignore]
fn burn_forward_gate_short_cuda_bf16() {
    let dump = load_dump("short/short_cpu_f32_torch_eager_single.safetensors");
    let (nmse, argmax_matches) = replay::<CudaBack>(&dump, Default::default());
    println!(
        "burn short cuda bf16 vs cpu f32 reference: nmse {nmse:.3e}, argmax {argmax_matches}/{}",
        dump.rows
    );
    // Bars pinned from measurement (the f32-recurrence-discipline
    // era reads 2.834e-5, 104/106 - the same kin class as
    // torch-cuda's own 2.151e-5/104 control; 106-row argmax is
    // tie-dominated).
    assert!(nmse <= 5e-5, "nmse {nmse:.3e} over the pinned 5e-5 bar");
    assert!(
        argmax_matches >= 103,
        "argmax {argmax_matches}/{} under the pinned 103 floor",
        dump.rows
    );
}

#[cfg(feature = "burn-cuda")]
#[test]
#[ignore]
fn burn_forward_gate_mid_cuda_bf16() {
    let dump = load_dump("mid/mid_cpu_f32_torch_eager_single.safetensors");
    let (nmse, argmax_matches) = replay::<CudaBack>(&dump, Default::default());
    println!(
        "burn mid cuda bf16 vs cpu f32 reference: nmse {nmse:.3e}, argmax {argmax_matches}/{}",
        dump.rows
    );
    // Bars re-pinned at the f32-recurrence-discipline era (flagged):
    // the reading moved 8.747e-5 -> 1.821e-5 at 2165/2169 - the
    // disciplined f32-state stacks' class at depth.
    assert!(nmse <= 5e-5, "nmse {nmse:.3e} over the pinned 5e-5 bar");
    assert!(
        argmax_matches >= 2160,
        "argmax {argmax_matches}/{} under the pinned 2160 floor",
        dump.rows
    );
}
