//! AdapterLoad locks: the three-tier token resolution, the
//! direction-3 placement + format validations, the checkpoint
//! geometry guard, the rows splice, and the zero-adapter exactness
//! gate (a zero-b adapter load must read value-equal to the plain
//! load).
use crate::*;

use crate::adapter::{
    AdapterDelta,
    AdapterFile,
    adapter_path,
};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lib_quest_adapter_{tag}_{}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// Write a synthetic adapter safetensors (f32 tensors + metadata).
fn write_adapter(
    path: &Path,
    tensors: &[(&str, Vec<usize>, Vec<f32>)],
    metadata: &[(&str, &str)],
) {
    let buffers: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, shape, values)| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (name.to_string(), shape.clone(), bytes)
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> = buffers
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    shape.clone(),
                    bytes,
                )
                .expect("view"),
            )
        })
        .collect();
    let metadata: HashMap<String, String> = metadata
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    safetensors::serialize_to_file(views, Some(metadata), path).expect("adapter write");
}

fn base_metadata(model_id: &str) -> Vec<(&'static str, &str)> {
    vec![
        ("version", "1"),
        ("model_id", model_id),
        ("lora_rank", "4"),
        ("lora_alpha", "8"),
    ]
}

#[test]
fn adapter_token_resolves_in_three_tiers() {
    let dir = PathBuf::from("adapters_home");
    let snake = adapter_path(&dir, "cpt_nushell").expect("snake");
    assert_eq!(snake, dir.join("cpt_nushell.safetensors"));
    let bare = adapter_path(&dir, "sub/dir.file.safetensors").expect("bare");
    assert_eq!(bare, dir.join("sub/dir.file.safetensors"));
    let home = std::env::var("HOME").expect("HOME");
    let expanded = adapter_path(&dir, "~/x.safetensors").expect("expanded");
    assert_eq!(expanded, PathBuf::from(format!("{home}/x.safetensors")));
}

/// The rows merge splices the seven channel rows through f32 and
/// leaves every other row untouched, in the base's own dtype - the
/// vocabulary-wide f32 materialization it replaced is what OOM'd the
/// serve load.
#[test]
fn rows_delta_splices_only_the_channel_rows() {
    let device = candle_core::Device::Cpu;
    let hidden = 4usize;
    let vocab = 100352usize;
    let config_at = consts::TOKEN_EXTRA_ID_0 as usize;
    let run_at = consts::TOKEN_EXTRA_ID_1 as usize;
    let base_values: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i % 251) as f32) * 0.5 - 62.0)
        .collect();
    let base = candle_core::Tensor::from_vec(base_values, (vocab, hidden), &device)
        .expect("base")
        .to_dtype(candle_core::DType::BF16)
        .expect("bf16 base");
    let delta_values: Vec<f32> = (0..7 * hidden).map(|i| ((i + 1) as f32) * 0.25).collect();
    let delta =
        candle_core::Tensor::from_vec(delta_values, (7, hidden), &device).expect("delta");

    let merged = AdapterDelta::Rows { delta: delta.clone() }
        .merge(base.clone())
        .expect("rows merge");
    assert_eq!(merged.dtype(), candle_core::DType::BF16);
    assert_eq!(merged.dims(), base.dims());

    let expected_rows = |at: usize, delta_at: usize, rows: usize| -> Vec<Vec<f32>> {
        ((base
            .narrow(0, at, rows)
            .expect("narrow")
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            + delta.narrow(0, delta_at, rows).expect("delta narrow"))
        .expect("add"))
        .to_dtype(candle_core::DType::BF16)
        .expect("round")
        .to_dtype(candle_core::DType::F32)
        .expect("back")
        .to_vec2::<f32>()
        .expect("host")
    };
    let merged_host = merged
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .to_vec2::<f32>()
        .expect("host");
    let base_host = base
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .to_vec2::<f32>()
        .expect("host");

    assert_eq!(merged_host[config_at], expected_rows(config_at, 0, 1)[0]);
    let expected_run = expected_rows(run_at, 1, 6);
    for offset in 0..6 {
        assert_eq!(merged_host[run_at + offset], expected_run[offset]);
    }
    for row in 0..vocab {
        if row == config_at || (run_at..run_at + 6).contains(&row) {
            continue;
        }
        assert_eq!(merged_host[row], base_host[row], "row {row} moved");
    }
}

#[test]
fn adapter_rejects_off_surface_targets() {
    let dir = temp_dir("off_surface");
    let path = dir.join("bad.safetensors");
    write_adapter(
        &path,
        &[
            ("model.layers.0.linear_attn.k_proj.lora_a", vec![8, 4], vec![0.0; 32]),
            ("model.layers.0.linear_attn.k_proj.lora_b", vec![4, 8], vec![0.0; 32]),
        ],
        &base_metadata("toy/model"),
    );
    let error = AdapterFile::load(&path, "toy/model", &candle_core::Device::Cpu)
        .expect_err("k_proj must reject");
    assert!(
        error.to_string().contains("direction-3"),
        "unexpected error: {error}"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn adapter_rejects_incomplete_pairs_and_wrong_identity() {
    let dir = temp_dir("incomplete");
    let half = dir.join("half.safetensors");
    write_adapter(
        &half,
        &[("model.layers.0.self_attn.q_proj.lora_a", vec![8, 4], vec![0.0; 32])],
        &base_metadata("toy/model"),
    );
    let error = AdapterFile::load(&half, "toy/model", &candle_core::Device::Cpu)
        .expect_err("half pair must reject");
    assert!(error.to_string().contains("incomplete"), "unexpected error: {error}");

    let wrong_model = dir.join("wrong_model.safetensors");
    write_adapter(
        &wrong_model,
        &[
            ("model.layers.0.self_attn.q_proj.lora_a", vec![8, 4], vec![0.0; 32]),
            ("model.layers.0.self_attn.q_proj.lora_b", vec![4, 8], vec![0.0; 32]),
        ],
        &base_metadata("other/model"),
    );
    let error = AdapterFile::load(&wrong_model, "toy/model", &candle_core::Device::Cpu)
        .expect_err("model id must reject");
    assert!(error.to_string().contains("trained for"), "unexpected error: {error}");

    let wrong_rank = dir.join("wrong_rank.safetensors");
    write_adapter(
        &wrong_rank,
        &[
            ("model.layers.0.self_attn.q_proj.lora_a", vec![8, 3], vec![0.0; 24]),
            ("model.layers.0.self_attn.q_proj.lora_b", vec![3, 8], vec![0.0; 24]),
        ],
        &base_metadata("toy/model"),
    );
    let error = AdapterFile::load(&wrong_rank, "toy/model", &candle_core::Device::Cpu)
        .expect_err("rank mismatch must reject");
    assert!(error.to_string().contains("rank"), "unexpected error: {error}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn adapter_geometry_validates_against_the_inventory() {
    let dir = temp_dir("geometry");
    let path = dir.join("adapter.safetensors");
    write_adapter(
        &path,
        &[
            ("model.layers.0.self_attn.q_proj.lora_a", vec![8, 4], vec![0.0; 32]),
            ("model.layers.0.self_attn.q_proj.lora_b", vec![4, 6], vec![0.0; 24]),
            ("model.layers.0.linear_attn.q_conv1d.delta", vec![8, 1, 4], vec![0.0; 32]),
        ],
        &base_metadata("toy/model"),
    );
    let file = AdapterFile::load(&path, "toy/model", &candle_core::Device::Cpu)
        .expect("valid placement loads");

    let inventory = |q_shape: Vec<usize>| {
        vec![
            TensorInfo {
                name: String::from("model.layers.0.self_attn.q_proj.weight"),
                dtype: String::from("BF16"),
                shape: q_shape,
                byte_len: 0,
            },
            TensorInfo {
                name: String::from("model.layers.0.linear_attn.q_conv1d.weight"),
                dtype: String::from("BF16"),
                shape: vec![8, 1, 4],
                byte_len: 0,
            },
        ]
    };
    file.validate_geometry(&inventory(vec![6, 8])).expect("matching geometry passes");
    let error = file
        .validate_geometry(&inventory(vec![8, 8]))
        .expect_err("mismatched geometry rejects");
    assert!(error.to_string().contains("shape"), "unexpected error: {error}");
    fs::remove_dir_all(&dir).ok();
}

/// The zero-adapter exactness gate: an adapter whose every b (and
/// conv delta) is zero merges to the identical model - forward_all
/// logits must read value-equal to the plain load (the f32 merge of
/// +0 casts back to the same bf16 values). Needs the DPO checkpoint.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn adapter_gate_zero_adapter_exact_cuda() {
    let device = candle_core::Device::new_cuda(0).expect("cuda device");
    let dtype = candle_core::DType::BF16;
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let checkpoint = load_config(&dir).expect("config");

    // Build the zero adapter over the full direction-3 surface at
    // rank 4 (a nonzero so the matmul path is exercised; b zero).
    let rank = 4usize;
    let hidden = checkpoint.hidden_size;
    let intermediate = checkpoint.intermediate_size;
    let key_dim = checkpoint.key_dim();
    let value_dim = checkpoint.value_dim();
    let kernel = checkpoint.linear_conv_kernel_dim;
    fn push_pair(
        tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
        rank: usize,
        name: String,
        in_dim: usize,
        out_dim: usize,
    ) {
        tensors.push((format!("{name}.lora_a"), vec![in_dim, rank], vec![0.01; in_dim * rank]));
        tensors.push((format!("{name}.lora_b"), vec![rank, out_dim], vec![0.0; rank * out_dim]));
    }
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for index in 0..checkpoint.num_hidden_layers {
        let prefix = format!("model.layers.{index}");
        match checkpoint.layer_kind(index) {
            LayerKind::LinearAttention => {
                push_pair(&mut tensors, rank, format!("{prefix}.linear_attn.q_proj"), hidden, key_dim);
                push_pair(&mut tensors, rank, format!("{prefix}.linear_attn.g_proj"), hidden, value_dim);
                push_pair(&mut tensors, rank, format!("{prefix}.linear_attn.o_proj"), value_dim, hidden);
                tensors.push((
                    format!("{prefix}.linear_attn.q_conv1d.delta"),
                    vec![key_dim, 1, kernel],
                    vec![0.0; key_dim * kernel],
                ));
            }
            LayerKind::FullAttention => {
                push_pair(&mut tensors, rank, format!("{prefix}.self_attn.q_proj"), hidden, hidden);
                push_pair(&mut tensors, rank, format!("{prefix}.self_attn.o_proj"), hidden, hidden);
            }
        }
        push_pair(&mut tensors, rank, format!("{prefix}.mlp.gate_proj"), hidden, intermediate);
        push_pair(&mut tensors, rank, format!("{prefix}.mlp.up_proj"), hidden, intermediate);
        push_pair(&mut tensors, rank, format!("{prefix}.mlp.down_proj"), intermediate, hidden);
    }
    let adapters_dir = temp_dir("zero_gate");
    let coordinate = LibConfig::default().model;
    let borrowed: Vec<(&str, Vec<usize>, Vec<f32>)> = tensors
        .iter()
        .map(|(name, shape, values)| (name.as_str(), shape.clone(), values.clone()))
        .collect();
    write_adapter(
        &adapters_dir.join("zero_gate.safetensors"),
        &borrowed,
        &base_metadata(&coordinate),
    );

    let ids: Vec<u32> = vec![
        100264, 882, 198, 3923, 374, 279, 6864, 315, 9822, 30, 100265, 198, 100264,
        78191, 198, 791, 6864, 315, 9822, 374, 12366, 13, 100265,
    ];

    let config = LibConfig {
        adapters_dir: adapters_dir.clone(),
        adapter: Some(String::from("zero_gate")),
        ..LibConfig::default()
    };
    let adapted_logits = {
        let weights = load_weights(&config, dtype, &device).expect("adapted load");
        let model = OlmoHybrid::new(&checkpoint, LibSettings::default(), weights)
            .expect("adapted model");
        let ids_tensor =
            candle_core::Tensor::from_vec(ids.clone(), ids.len(), &device).expect("ids");
        let logits = model.forward_all(&ids_tensor).expect("adapted forward");
        logits
            .to_dtype(candle_core::DType::F32)
            .expect("cast")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("host")
    };

    let plain_logits = {
        let weights = mmap_weights(&dir, dtype, &device).expect("plain load");
        let model = OlmoHybrid::new(&checkpoint, LibSettings::default(), weights)
            .expect("plain model");
        let ids_tensor =
            candle_core::Tensor::from_vec(ids.clone(), ids.len(), &device).expect("ids");
        let logits = model.forward_all(&ids_tensor).expect("plain forward");
        logits
            .to_dtype(candle_core::DType::F32)
            .expect("cast")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("host")
    };

    assert_eq!(adapted_logits.len(), plain_logits.len());
    assert!(
        adapted_logits == plain_logits,
        "zero-adapter logits diverge from the plain load"
    );
    fs::remove_dir_all(&adapters_dir).ok();
}
