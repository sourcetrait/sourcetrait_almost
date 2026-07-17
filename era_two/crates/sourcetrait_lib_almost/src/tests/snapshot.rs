//! PrefixSnapshots locks: the three-tier token resolution and the
//! file format's validation rejections (checkpoint-free, synthetic
//! files), plus the SnapshotCore roundtrip determinism gates
//! (#[ignore]; they need the DPO checkpoint).
use crate::*;
use crate::snapshot::{SNAPSHOT_VERSION, read_snapshot, write_snapshot};

fn temp_root(tag: &str) -> PathBuf {
    let root = env::temp_dir().join(format!(
        "lib_almost_snapshot_{tag}_{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root).expect("clean temp root");
    }
    fs::create_dir_all(&root).expect("create temp root");
    root
}

#[test]
fn snapshot_tokens_resolve_three_tiers() {
    let dir = Path::new("/opt/snaps");
    // Tier a: a pure snake gets the dir + the extension.
    assert_eq!(
        snapshot_path(dir, "chat_a").expect("snake"),
        PathBuf::from("/opt/snaps/chat_a.safetensors")
    );
    // Tier b: a bare relative path resolves under the dir as-is.
    assert_eq!(
        snapshot_path(dir, "sub/run.safetensors").expect("bare"),
        PathBuf::from("/opt/snaps/sub/run.safetensors")
    );
    assert_eq!(
        snapshot_path(dir, "chat_a.safetensors").expect("bare file"),
        PathBuf::from("/opt/snaps/chat_a.safetensors"),
        "the snake and its explicit filename land on the same path"
    );
    // Tier c: normal paths pass through / expand.
    assert_eq!(
        snapshot_path(dir, "/abs/run.safetensors").expect("absolute"),
        PathBuf::from("/abs/run.safetensors")
    );
    assert_eq!(
        snapshot_path(dir, "./here.safetensors").expect("cwd-relative"),
        PathBuf::from("./here.safetensors")
    );
    let home = snapshot_path(dir, "~/run.safetensors").expect("tilde");
    assert!(!home.to_string_lossy().contains('~'));
    assert!(home.is_absolute());
    let via_env = snapshot_path(dir, "$XDG_CACHE_HOME/x.safetensors").expect("env");
    assert!(!via_env.to_string_lossy().contains('$'));
}

fn tiny_tensor(device: &candle_core::Device) -> candle_core::Tensor {
    candle_core::Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), device).expect("tensor")
}

#[test]
fn snapshot_file_roundtrips_tensors_trail_and_metadata() {
    let root = temp_root("file_roundtrip");
    let path = root.join("deep/mini.safetensors");
    let device = candle_core::Device::Cpu;
    write_snapshot(
        &path,
        vec![("gdn_state_0".to_string(), tiny_tensor(&device))],
        "mini-model",
        3,
        &[7, 8, 9],
    )
    .expect("writes (parent dir created)");
    let file = read_snapshot(&path, "mini-model", &device).expect("reads");
    assert_eq!(file.context_len, 3);
    assert_eq!(file.context_ids, vec![7, 8, 9]);
    let state = file.tensor("gdn_state_0").expect("present");
    assert_eq!(state.dims(), [2, 2]);
    assert_eq!(state.dtype(), candle_core::DType::F32);
    assert!(file.tensor("missing").is_err());
    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn snapshot_rejects_wrong_model_id_version_trail_and_truncation() {
    let root = temp_root("rejections");
    let device = candle_core::Device::Cpu;
    let path = root.join("mini.safetensors");
    write_snapshot(
        &path,
        vec![("t".to_string(), tiny_tensor(&device))],
        "mini-model",
        2,
        &[1, 2],
    )
    .expect("writes");
    assert!(
        read_snapshot(&path, "other-model", &device).is_err(),
        "wrong model id"
    );

    // A short trail rejects at write (the caller-bug guard)...
    assert!(
        write_snapshot(
            &root.join("short.safetensors"),
            vec![("t".to_string(), tiny_tensor(&device))],
            "m",
            5,
            &[1, 2],
        )
        .is_err()
    );
    // ...and a doctored file rejects at read.
    let mut info: HashMap<String, String> = HashMap::new();
    info.insert("version".to_string(), SNAPSHOT_VERSION.to_string());
    info.insert("model_id".to_string(), "mini-model".to_string());
    info.insert("context_len".to_string(), "5".to_string());
    let doctored = root.join("doctored.safetensors");
    let ids = candle_core::Tensor::from_vec(vec![1u32, 2], 2, &device).expect("ids");
    safetensors::serialize_to_file(
        vec![("context_ids".to_string(), ids)],
        Some(info.clone()),
        &doctored,
    )
    .expect("writes");
    assert!(
        read_snapshot(&doctored, "mini-model", &device).is_err(),
        "trail shorter than context_len"
    );

    // A wrong format version rejects.
    info.insert("version".to_string(), "0".to_string());
    info.insert("context_len".to_string(), "1".to_string());
    let versioned = root.join("versioned.safetensors");
    let ids = candle_core::Tensor::from_vec(vec![1u32], 1, &device).expect("ids");
    safetensors::serialize_to_file(
        vec![("context_ids".to_string(), ids)],
        Some(info),
        &versioned,
    )
    .expect("writes");
    assert!(
        read_snapshot(&versioned, "mini-model", &device).is_err(),
        "unsupported version"
    );

    // A truncated file rejects at the parse layer.
    let bytes = fs::read(&path).expect("read back");
    let cut = root.join("truncated.safetensors");
    fs::write(&cut, &bytes[..bytes.len() / 2]).expect("writes");
    assert!(
        read_snapshot(&cut, "mini-model", &device).is_err(),
        "truncated file"
    );
    fs::remove_dir_all(&root).expect("cleanup");
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

/// Drive the roundtrip on a device: an uninterrupted N+M forced-greedy
/// run, a save at N on the same model, a FRESH model restore, and a
/// manual decode continuation - the chained ids must reproduce the
/// uninterrupted ids exactly (the state roundtrip is lossless byte
/// motion, so both grades assert equality; a cuda argmax-tie flip
/// here would be a finding to recalibrate on, per the kernel-era
/// discipline).
fn roundtrip_drill(device: &candle_core::Device, dtype: candle_core::DType) {
    const SPLIT: usize = 6;
    const TAIL: usize = 6;
    let root = temp_root(if device.is_cuda() { "roundtrip_cuda" } else { "roundtrip_cpu" });
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("config");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");
    let weights = mmap_weights(&dir, dtype, device).expect("mmap");
    let mut model = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model");

    // The uninterrupted leg: SPLIT + TAIL forced greedy tokens.
    let options = GenerateOptions {
        temperature: None,
        top_p: None,
        sample_len: SPLIT + TAIL,
        chat: true,
        ignore_stops: true,
        ..GenerateOptions::default()
    };
    let mut generation = model
        .generate(&tokenizer, "Name three colors.", &options)
        .expect("uninterrupted generation");
    let mut full_ids = Vec::new();
    for step in generation.by_ref() {
        full_ids.push(step.expect("step").token_id);
    }
    let full_report = generation.finish();
    assert_eq!(full_ids.len(), SPLIT + TAIL, "forced decode length");
    // The trail is prompt + consumed decode ids; the final emitted
    // token never forwards on a sample_len end.
    assert_eq!(
        full_report.context_ids.len(),
        full_report.prompt_token_count + SPLIT + TAIL - 1
    );
    assert_eq!(
        full_report.context_ids[full_report.prompt_token_count..],
        full_ids[..SPLIT + TAIL - 1]
    );

    // The save leg: the same prompt for SPLIT tokens, then snapshot.
    let options_a = GenerateOptions {
        sample_len: SPLIT,
        ..options
    };
    let mut generation = model
        .generate(&tokenizer, "Name three colors.", &options_a)
        .expect("save-leg generation");
    let mut save_ids = Vec::new();
    for step in generation.by_ref() {
        save_ids.push(step.expect("step").token_id);
    }
    let report_a = generation.finish();
    assert_eq!(save_ids, full_ids[..SPLIT], "greedy legs share the prefix");
    assert_eq!(model.context_len(), report_a.context_ids.len());
    let saved_path = model
        .snapshot_caches(&root, "gate_a", "gate-model", &report_a.context_ids)
        .expect("snapshot saves");
    assert!(saved_path.is_file());
    assert_eq!(
        saved_path,
        root.join("gate_a.safetensors"),
        "the snake tier resolved"
    );

    // ONE resident model at a time (a second ~14 GiB weight set OOMs
    // the card, and two ~30 GB f32 hosts brush the 64 GB box): the
    // save-leg model drops before the fresh build.
    drop(model);

    // A fresh model restores and continues: feed the save leg's final
    // emitted-unconsumed token, then greedy onward.
    let weights = mmap_weights(&dir, dtype, device).expect("fresh mmap");
    let mut fresh = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("fresh model");
    let restored = fresh
        .restore_caches(&root, "gate_a", "gate-model")
        .expect("restores");
    assert_eq!(restored.context_ids, report_a.context_ids);
    assert_eq!(fresh.context_len(), restored.context_len);

    let mut chained_tail = Vec::new();
    let mut next = *save_ids.last().expect("save leg decoded");
    for _ in 0..TAIL {
        let ids = candle_core::Tensor::from_vec(vec![next], 1, fresh.device()).expect("ids");
        let logits = fresh.forward_chunk(&ids).expect("decode step");
        let row = logits
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("host");
        next = argmax(&row);
        chained_tail.push(next);
    }
    assert_eq!(
        chained_tail,
        full_ids[SPLIT..],
        "the chained continuation reproduces the uninterrupted ids"
    );

    // Live rejections on the saved file.
    assert!(
        fresh.restore_caches(&root, "gate_a", "another-model").is_err(),
        "wrong model id"
    );
    fs::remove_dir_all(&root).expect("cleanup");
}

/// The SnapshotCore roundtrip gate, cpu f32: chained
/// save/fresh-restore/continue ids byte-equal to the uninterrupted
/// run, trail semantics pinned, the snake tier + rejections live.
#[test]
#[ignore = "needs the DPO checkpoint; ~2-3 min cpu"]
fn snapshot_gate_roundtrip_cpu_f32() {
    roundtrip_drill(&candle_core::Device::Cpu, candle_core::DType::F32);
}

/// The cuda bf16 roundtrip gate (same drill; the state roundtrip is
/// lossless byte motion, so ids assert equal here too).
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + a cuda card"]
fn snapshot_gate_roundtrip_cuda_bf16() {
    roundtrip_drill(
        &candle_core::Device::new_cuda(0).expect("cuda"),
        candle_core::DType::BF16,
    );
}

/// The ContinueSurface exactness gate, cpu f32 greedy: a chained
/// save/fresh-restore/generate_from continuation produces BYTE-EQUAL
/// ids to one monolithic raw prefill of the identical transcript,
/// tokenization seam included (the chat specials are hard BPE
/// boundaries - era-one proven, re-gated fresh here).
#[test]
#[ignore = "needs the DPO checkpoint; ~3-4 min cpu"]
fn snapshot_gate_chained_vs_monolithic_cpu_f32() {
    const TURN_BUDGET: usize = 24;
    let root = temp_root("chained_mono");
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("config");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");
    let device = candle_core::Device::Cpu;
    let weights = mmap_weights(&dir, candle_core::DType::F32, &device).expect("mmap");
    let mut model = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model");

    // Turn 1 (chat): run to the natural stop so the saved context is
    // transcript-complete.
    let options = GenerateOptions::greedy(TURN_BUDGET);
    let mut generation = model
        .generate(&tokenizer, "What is the capital of France?", &options)
        .expect("turn 1");
    let mut turn1_text = String::new();
    for step in generation.by_ref() {
        turn1_text.push_str(&step.expect("step").chunk);
    }
    let report1 = generation.finish();
    turn1_text.push_str(&report1.rest);
    assert_eq!(
        report1.finish_reason,
        Some(FinishReason::StopToken),
        "the chained gate wants a transcript-complete turn"
    );
    model
        .snapshot_caches(&root, "turn1", "gate-model", &report1.context_ids)
        .expect("save");
    drop(model);

    // Turn 2, chained: restore on a fresh model, continue as a chat
    // turn.
    let turn2 = "And of Italy?";
    let weights = mmap_weights(&dir, candle_core::DType::F32, &device).expect("mmap");
    let mut chained = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model");
    let restored = chained
        .restore_caches(&root, "turn1", "gate-model")
        .expect("restore");
    let mut generation = chained
        .generate_from(&tokenizer, &restored, turn2, &options)
        .expect("turn 2 chained");
    let mut chained_ids = Vec::new();
    for step in generation.by_ref() {
        chained_ids.push(step.expect("step").token_id);
    }
    let report2 = generation.finish();
    assert!(
        report2.context_ids.starts_with(&restored.context_ids),
        "the chained trail extends the restored trail"
    );
    drop(chained);

    // The monolithic leg: ONE raw prefill of the identical transcript
    // (turn-1 render + turn-1 text + the continuation render).
    let transcript = format!(
        "{}{}{}",
        chat_wrap("What is the capital of France?"),
        turn1_text,
        chat_continue(turn2),
    );
    let weights = mmap_weights(&dir, candle_core::DType::F32, &device).expect("mmap");
    let mut mono = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model");
    let raw = GenerateOptions {
        chat: false,
        ..options
    };
    let mut generation = mono
        .generate(&tokenizer, &transcript, &raw)
        .expect("monolithic");
    let mut mono_ids = Vec::new();
    for step in generation.by_ref() {
        mono_ids.push(step.expect("step").token_id);
    }
    let mono_report = generation.finish();

    assert_eq!(
        chained_ids, mono_ids,
        "chained-vs-monolithic must be byte-equal at f32 (chained text: {:?})",
        report2.rest
    );
    // The id seam: the monolithic transcript re-encodes to exactly
    // the chained trail prefix (the hard-BPE-boundary property).
    assert_eq!(
        mono_report.context_ids[..restored.context_ids.len()],
        restored.context_ids[..],
        "the transcript re-encoding matches the chained trail"
    );
    fs::remove_dir_all(&root).expect("cleanup");
}

/// The gpu chained drill under evict + graph: save an
/// eviction-compacted context, restore on a fresh model, continue
/// across overflow epochs (store cap held), then save/restore again
/// and continue with a LONG suffix - the reserve regrow on captured
/// buffers parks them, capture retires, and the run completes on the
/// staged-uncaptured path. Prints the save/restore timing row.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + a cuda card; ~2 min"]
fn snapshot_gate_chained_drill_evict_graph_cuda() {
    const CAP: usize = 512;
    let root = temp_root("chained_drill");
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("config");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let settings = LibSettings {
        graph: true,
        graph_bucket_grain: 128,
        eviction: Some(EvictionSettings {
            decode_cap: CAP,
            recent: 128,
            sink: 4,
        }),
        ..LibSettings::default()
    };
    let options = GenerateOptions {
        temperature: None,
        top_p: None,
        sample_len: 48,
        chat: false,
        ignore_stops: true,
        ..GenerateOptions::default()
    };

    // Turn 1: a compacting prompt (> cap), forced decode, save.
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
    let mut model = OlmoHybrid::new(&config, settings.clone(), weights).expect("model");
    let prompt = "the cat sat on the mat ".repeat(160);
    let mut generation = model.generate(&tokenizer, &prompt, &options).expect("turn 1");
    for step in generation.by_ref() {
        step.expect("step");
    }
    let report1 = generation.finish();
    assert!(report1.prompt_token_count > CAP, "compacting prompt");
    let timer = std::time::Instant::now();
    model
        .snapshot_caches(&root, "drill", "gate-model", &report1.context_ids)
        .expect("save");
    let save_seconds = timer.elapsed().as_secs_f64();
    assert!(
        model.context_len() < CAP + evict::OVERFLOW_SLACK + 1,
        "cap held at save"
    );
    assert!(
        report1.context_ids.len() > model.context_len(),
        "the trail outruns the compacted store"
    );
    drop(model);
    // A captured-against model dropped mid-process leaves a recorded
    // driver error behind (the captured-buffer free law; its parked
    // buffers leak, bounded) - DRAIN it before the next load or the
    // fresh build's first fallible call delivers
    // CUDA_ERROR_INVALID_VALUE.
    let _ = device.synchronize();

    // Turn 2: restore on a fresh evict+graph model, continue across
    // overflow epochs.
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
    let mut fresh = OlmoHybrid::new(&config, settings, weights).expect("fresh model");
    let timer = std::time::Instant::now();
    let restored = fresh
        .restore_caches(&root, "drill", "gate-model")
        .expect("restore");
    let restore_seconds = timer.elapsed().as_secs_f64();
    println!(
        "snapshot timing row: save {save_seconds:.2}s, restore {restore_seconds:.2}s \
         ({} store rows, {} trail ids)",
        restored.context_len,
        restored.context_ids.len()
    );
    let continue_options = GenerateOptions {
        sample_len: 200,
        ..options
    };
    let mut generation = fresh
        .generate_from(&tokenizer, &restored, " the dog sat on the log ", &continue_options)
        .expect("turn 2");
    for step in generation.by_ref() {
        step.expect("step");
    }
    let report2 = generation.finish();
    assert_eq!(report2.generated_token_count, 200, "forced decode length");
    assert!(
        fresh.context_len() < CAP + evict::OVERFLOW_SLACK + 1,
        "cap held across the epochs"
    );
    assert!(
        report2.context_ids.len() > restored.context_ids.len(),
        "the trail kept growing"
    );

    // Turn 3: save/restore on the SAME (captured) model, then a LONG
    // suffix forces a capacity regrow - park containment retires
    // capture and the run completes staged-uncaptured.
    fresh
        .snapshot_caches(&root, "drill2", "gate-model", &report2.context_ids)
        .expect("second save");
    let restored2 = fresh
        .restore_caches(&root, "drill2", "gate-model")
        .expect("restore onto the captured model");
    let long_suffix = "the fox ran over the hill and ".repeat(120);
    let mut generation = fresh
        .generate_from(&tokenizer, &restored2, &long_suffix, &continue_options)
        .expect("turn 3");
    for step in generation.by_ref() {
        step.expect("step");
    }
    let report3 = generation.finish();
    assert_eq!(
        report3.generated_token_count, 200,
        "the post-park run completes"
    );
    assert!(
        fresh.context_len() < CAP + evict::OVERFLOW_SLACK + 1,
        "cap held after the park"
    );
    fs::remove_dir_all(&root).expect("cleanup");
}
