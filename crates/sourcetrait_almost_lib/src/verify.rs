use crate::*;

/// Knobs for the internal equivalence battery.
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    /// Use a prompt longer than the sliding window, so the D3 trim, the
    /// Window mask, and the yarn/vanilla split are all live at once.
    pub long: bool,
    /// Additionally diff this device/dtype against a cpu f32 run
    /// (informational; always uses the quick prompt).
    pub cross_device: bool,
    /// How many trailing positions the incremental-decode check covers.
    pub decode_steps: usize,
    /// Run the battery through the flash path (T5); the cross-device
    /// cpu reference model always stays eager.
    pub use_flash_attn: bool,
}

struct Comparison {
    label: String,
    rows: usize,
    max_abs: f32,
    nmse: f64,
    argmax_match: f64,
    bar: Option<f64>,
    /// (row, max_abs) worst offenders; printed when a check fails.
    worst_rows: Vec<(usize, f32)>,
}

const QUICK_PROMPT: &str =
    "Explain briefly why the sky appears blue during the day and reddish at sunset.";
const LONG_SENTENCE: &str =
    "The quick brown fox jumps over the lazy dog while the river keeps rolling east past the old mill. ";

/// Self-consistency battery: every check compares two code paths of the
/// SAME implementation, so agreement bars are tight. Exits with an error
/// when a same-device check exceeds its bar or the T4 sliding-cache bound
/// is breached; cross-device numbers are reported only.
pub fn verify(
    paths: &ModelPaths,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    opts: &VerifyOptions,
) -> AlmostResult<()> {
    let config: Olmo3Config = serde_json::from_reader(std::fs::File::open(&paths.config)?)?;
    let tokenizer = match tokenizers::Tokenizer::from_file(&paths.tokenizer) {
        Ok(tokenizer) => tokenizer,
        Err(error) => snafu::whatever!("loading tokenizer.json failed: {error}"),
    };
    let ids = build_ids(&tokenizer, opts.long, config.sliding_window)?;
    let n = ids.len();
    // Bars pinned from measurement AT THE LENGTH THEY GUARD. f32: bitwise
    // short, ~1.2e-9 at 4352 (accumulation order at length) -> 1e-8.
    // bf16 quick (~53 rows): path-vs-path wobble 1.1-2.2e-4 -> 5e-4.
    // bf16 long (4352+ rows): drift COMPOUNDS with depth x length -
    // chunk-boundary changes alone read 1.28e-2 nmse, two valid bf16
    // stacks (flash vs eager, same ids) read 2.49e-2, and each sits
    // ~4.4-4.7e-2 from f32 truth at argmax 0.9986 (oracle-anchored,
    // 4407 rows) -> 5e-2. Real cache/mask/offset bugs still sit orders
    // above every bar (a visibility error reads order-1).
    let same_device_bar = if dtype == candle_core::DType::F32 {
        1e-8
    } else if opts.long {
        5e-2
    } else {
        5e-4
    };
    // E3 rings: at most window-1 entries persist between prefill chunks,
    // window right after a decode step (the current token's slot).
    let cache_cap = config.sliding_window;
    eprintln!(
        "almost verify: {n} tokens, device {device:?}, dtype {dtype:?}, mode {}",
        if opts.long { "long (window exceeded)" } else { "quick" }
    );

    let load_start = Instant::now();
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths.shards, dtype, device)? };
    let settings = Settings {
        use_flash_attn: opts.use_flash_attn,
    };
    let mut model = Model::new(&config, settings, vb)?;
    eprintln!("almost verify: weights loaded in {:.1}s", load_start.elapsed().as_secs_f32());

    let mut results: Vec<Comparison> = Vec::new();
    // T4: sliding caches never exceed window-1 entries; sampled after
    // every phase, reported as its own check line.
    let mut cache_bound_breaches: Vec<String> = Vec::new();
    let mut max_cache_seen = 0usize;

    // Reference: one full prefill over all positions.
    let reference = forward_all_rows(&mut model, &ids, 0, device, true)?;
    sample_cache_bound(&model, cache_cap, "reference", &mut cache_bound_breaches, &mut max_cache_seen);

    // Determinism: the identical forward, repeated.
    let rerun = forward_all_rows(&mut model, &ids, 0, device, true)?;
    results.push(compare("determinism (same forward twice)", &rerun, &reference, Some(1e-10))?);
    sample_cache_bound(&model, cache_cap, "determinism", &mut cache_bound_breaches, &mut max_cache_seen);

    // Chunked prefill vs single (offset-aware banded masks + mid-sequence
    // trim state).
    let split = if opts.long { 3000.min(n - 8) } else { n / 2 };
    model.clear_kv_cache();
    let chunk_a = forward_all_rows(&mut model, &ids[..split], 0, device, false)?;
    let chunk_b = forward_all_rows(&mut model, &ids[split..], split, device, false)?;
    let chunked = candle_core::Tensor::cat(&[&chunk_a, &chunk_b], 0)?;
    results.push(compare("chunked prefill vs single", &chunked, &reference, Some(same_device_bar))?);
    sample_cache_bound(&model, cache_cap, "chunked", &mut cache_bound_breaches, &mut max_cache_seen);

    // Incremental decode vs single (cache/offset math; in long mode the
    // trimmed-cache maskless decode vs the banded window mask). The
    // boundary state also feeds the E2 snapshot roundtrip: persist,
    // decode, restore, decode again - the two passes must agree at
    // determinism grade.
    let steps = opts.decode_steps.min(n.saturating_sub(2));
    let boundary = n - steps;
    model.clear_kv_cache();
    let _ = model.forward(&tensor_ids(&ids[..boundary], device)?, 0)?;
    let snapshot_path = std::env::temp_dir().join(format!(
        "almost_verify_snapshot_{}.safetensors",
        std::process::id()
    ));
    let snapshot_len = model.snapshot_caches(&snapshot_path)?;
    snafu::ensure_whatever!(
        snapshot_len == boundary,
        "snapshot context length {snapshot_len} != boundary {boundary}"
    );
    let mut rows: Vec<candle_core::Tensor> = Vec::with_capacity(steps);
    for position in boundary..n {
        let row = model.forward(&tensor_ids(&ids[position..position + 1], device)?, position)?;
        rows.push(
            row.squeeze(0)?
                .to_device(&candle_core::Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?,
        );
    }
    let incremental = candle_core::Tensor::cat(&rows, 0)?;
    let reference_tail = reference.narrow(0, boundary, steps)?;
    results.push(compare(
        "incremental decode vs single",
        &incremental,
        &reference_tail,
        Some(same_device_bar),
    )?);
    sample_cache_bound(&model, cache_cap, "incremental", &mut cache_bound_breaches, &mut max_cache_seen);

    // E2 snapshot roundtrip: restore the boundary state and replay the
    // same decode steps; agreement is determinism-grade (same device,
    // same path, restored caches).
    let restored_len = model.restore_caches(&snapshot_path)?;
    std::fs::remove_file(&snapshot_path)?;
    snafu::ensure_whatever!(
        restored_len == boundary,
        "restored context length {restored_len} != boundary {boundary}"
    );
    let mut replay_rows: Vec<candle_core::Tensor> = Vec::with_capacity(steps);
    for position in boundary..n {
        let row = model.forward(&tensor_ids(&ids[position..position + 1], device)?, position)?;
        replay_rows.push(
            row.squeeze(0)?
                .to_device(&candle_core::Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?,
        );
    }
    let replayed = candle_core::Tensor::cat(&replay_rows, 0)?;
    results.push(compare(
        "snapshot-restore decode replay",
        &replayed,
        &incremental,
        Some(1e-10),
    )?);
    sample_cache_bound(&model, cache_cap, "snapshot", &mut cache_bound_breaches, &mut max_cache_seen);

    // Cross-device (informational): quick ids on cpu f32.
    if opts.cross_device {
        if matches!(device, candle_core::Device::Cpu) && dtype == candle_core::DType::F32 {
            eprintln!("almost verify: skipping --cross-device (already on cpu f32)");
        } else {
            let quick_ids = build_ids(&tokenizer, false, config.sliding_window)?;
            let primary = if opts.long {
                forward_all_rows(&mut model, &quick_ids, 0, device, true)?
            } else {
                reference.clone()
            };
            eprintln!("almost verify: loading cpu f32 model for cross-device diff");
            let cpu = candle_core::Device::Cpu;
            let vb_cpu = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&paths.shards, candle_core::DType::F32, &cpu)?
            };
            let mut model_cpu = Model::new(&config, Settings::default(), vb_cpu)?;
            let reference_cpu = forward_all_rows(&mut model_cpu, &quick_ids, 0, &cpu, true)?;
            results.push(compare(
                "cross-device (this vs cpu f32, informational)",
                &primary,
                &reference_cpu,
                None,
            )?);
        }
    }

    let mut failed: Vec<String> = Vec::new();
    println!("check | rows | max_abs | nmse | argmax_match | verdict");
    for result in &results {
        let verdict = match result.bar {
            None => String::from("info"),
            Some(bar) if result.nmse <= bar => String::from("PASS"),
            Some(bar) => {
                failed.push(result.label.clone());
                format!("FAIL (bar {bar:.0e})")
            }
        };
        println!(
            "{} | {} | {:.3e} | {:.3e} | {:.4} | {}",
            result.label, result.rows, result.max_abs, result.nmse, result.argmax_match, verdict
        );
        if verdict.starts_with("FAIL") {
            for (row, row_max) in &result.worst_rows {
                println!("  worst row {row} | max_abs {row_max:.3e}");
            }
        }
    }
    let bound_verdict = if cache_bound_breaches.is_empty() {
        String::from("PASS")
    } else {
        failed.push(String::from("sliding cache bound"));
        format!("FAIL ({})", cache_bound_breaches.join("; "))
    };
    println!(
        "sliding cache bound (max {} <= {}) | {}",
        max_cache_seen, cache_cap, bound_verdict
    );
    snafu::ensure_whatever!(
        failed.is_empty(),
        "verify failed: {}",
        failed.join("; ")
    );
    Ok(())
}

fn sample_cache_bound(
    model: &Model,
    cache_cap: usize,
    phase: &str,
    breaches: &mut Vec<String>,
    max_seen: &mut usize,
) {
    let cached = model.max_sliding_cache_len();
    if cached > *max_seen {
        *max_seen = cached;
    }
    if cached > cache_cap {
        breaches.push(format!("{phase}: {cached}"));
    }
}

fn build_ids(tokenizer: &tokenizers::Tokenizer, long: bool, window: usize) -> AlmostResult<Vec<u32>> {
    let text = if long {
        LONG_SENTENCE.repeat(window / 4)
    } else {
        chat_wrap(QUICK_PROMPT)
    };
    let encoding = match tokenizer.encode(text, true) {
        Ok(encoding) => encoding,
        Err(error) => snafu::whatever!("verify tokenization failed: {error}"),
    };
    let mut ids = encoding.get_ids().to_vec();
    if long {
        // window + 256: enough past-window positions to exercise trim and
        // the banded mask, small enough that the eager attention transient
        // fits beside the weights.
        let target = window + 256;
        snafu::ensure_whatever!(
            ids.len() >= target,
            "long-mode prompt tokenized too short ({} < {target})",
            ids.len()
        );
        ids.truncate(target);
    }
    Ok(ids)
}

fn tensor_ids(ids: &[u32], device: &candle_core::Device) -> AlmostResult<candle_core::Tensor> {
    Ok(candle_core::Tensor::new(ids, device)?.unsqueeze(0)?)
}

/// Full prefill returning (rows, vocab) f32 on cpu; clears the cache first
/// unless the call continues a chunked sequence.
fn forward_all_rows(
    model: &mut Model,
    ids: &[u32],
    offset: usize,
    device: &candle_core::Device,
    clear_first: bool,
) -> AlmostResult<candle_core::Tensor> {
    if clear_first {
        model.clear_kv_cache();
    }
    Ok(model
        .forward_all(&tensor_ids(ids, device)?, offset)?
        .squeeze(0)?
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?)
}

fn compare(
    label: &str,
    candidate: &candle_core::Tensor,
    reference: &candle_core::Tensor,
    bar: Option<f64>,
) -> AlmostResult<Comparison> {
    let (rows, vocab) = candidate.dims2()?;
    let (reference_rows, _) = reference.dims2()?;
    snafu::ensure_whatever!(
        rows == reference_rows,
        "{label}: row mismatch {rows} vs {reference_rows}"
    );
    let a: Vec<f32> = candidate.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = reference.flatten_all()?.to_vec1()?;
    let mut max_abs = 0f32;
    let mut diff_sq = 0f64;
    let mut reference_sq = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        diff_sq += (diff as f64) * (diff as f64);
        reference_sq += (*y as f64) * (*y as f64);
    }
    let nmse = if reference_sq > 0.0 { diff_sq / reference_sq } else { 0.0 };
    let mut matches = 0usize;
    let mut row_peaks: Vec<(usize, f32)> = Vec::with_capacity(rows);
    for row in 0..rows {
        let range = row * vocab..(row + 1) * vocab;
        let argmax_a = argmax(&a[range.clone()]);
        let argmax_b = argmax(&b[range.clone()]);
        if argmax_a == argmax_b {
            matches += 1;
        }
        let mut row_max = 0f32;
        for (x, y) in a[range.clone()].iter().zip(b[range].iter()) {
            let diff = (x - y).abs();
            if diff > row_max {
                row_max = diff;
            }
        }
        row_peaks.push((row, row_max));
    }
    row_peaks.sort_by(|left, right| right.1.total_cmp(&left.1));
    row_peaks.truncate(8);
    Ok(Comparison {
        label: String::from(label),
        rows,
        max_abs,
        nmse,
        argmax_match: matches as f64 / rows as f64,
        bar,
        worst_rows: row_peaks,
    })
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best
}
