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
