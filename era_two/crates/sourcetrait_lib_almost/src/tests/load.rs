//! Weight-inventory locks (the D1 gate: the shard's tensor layout
//! equals the pinned ground truth) plus a spot materialization through
//! the mmap VarBuilder path.
use crate::*;

const SHARD_DATA_BYTES: u64 = 14_861_741_376;
const TENSOR_COUNT: usize = 523;

const GDN_LAYER_SUFFIXES: [(&str, &[usize]); 18] = [
    ("input_layernorm.weight", &[3840]),
    ("linear_attn.A_log", &[30]),
    ("linear_attn.a_proj.weight", &[30, 3840]),
    ("linear_attn.b_proj.weight", &[30, 3840]),
    ("linear_attn.dt_bias", &[30]),
    ("linear_attn.g_proj.weight", &[5760, 3840]),
    ("linear_attn.k_conv1d.weight", &[2880, 1, 4]),
    ("linear_attn.k_proj.weight", &[2880, 3840]),
    ("linear_attn.o_norm.weight", &[192]),
    ("linear_attn.o_proj.weight", &[3840, 5760]),
    ("linear_attn.q_conv1d.weight", &[2880, 1, 4]),
    ("linear_attn.q_proj.weight", &[2880, 3840]),
    ("linear_attn.v_conv1d.weight", &[5760, 1, 4]),
    ("linear_attn.v_proj.weight", &[5760, 3840]),
    ("mlp.down_proj.weight", &[3840, 11008]),
    ("mlp.gate_proj.weight", &[11008, 3840]),
    ("mlp.up_proj.weight", &[11008, 3840]),
    ("post_attention_layernorm.weight", &[3840]),
];

const ATTN_LAYER_SUFFIXES: [(&str, &[usize]); 11] = [
    ("mlp.down_proj.weight", &[3840, 11008]),
    ("mlp.gate_proj.weight", &[11008, 3840]),
    ("mlp.up_proj.weight", &[11008, 3840]),
    ("post_attention_layernorm.weight", &[3840]),
    ("post_feedforward_layernorm.weight", &[3840]),
    ("self_attn.k_norm.weight", &[3840]),
    ("self_attn.k_proj.weight", &[3840, 3840]),
    ("self_attn.o_proj.weight", &[3840, 3840]),
    ("self_attn.q_norm.weight", &[3840]),
    ("self_attn.q_proj.weight", &[3840, 3840]),
    ("self_attn.v_proj.weight", &[3840, 3840]),
];

fn check_inventory(model_name: &str) {
    let dir = model_dir(model_name).expect("model dir");
    let tensors = tensor_inventory(&dir).expect("header parses");
    assert_eq!(tensors.len(), TENSOR_COUNT);
    assert!(tensors.iter().all(|t| t.dtype == "BF16"));
    assert_eq!(tensors.iter().map(|t| t.byte_len).sum::<u64>(), SHARD_DATA_BYTES);

    let shape_of = |name: &str| -> Vec<usize> {
        tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("tensor {name} present"))
            .shape
            .clone()
    };
    assert_eq!(shape_of("model.embed_tokens.weight"), vec![100352, 3840]);
    assert_eq!(shape_of("lm_head.weight"), vec![100352, 3840]);
    assert_eq!(shape_of("model.norm.weight"), vec![3840]);

    let layer_suffixes = |layer: usize| -> Vec<(String, Vec<usize>)> {
        let prefix = format!("model.layers.{layer}.");
        let mut set: Vec<(String, Vec<usize>)> = tensors
            .iter()
            .filter(|t| t.name.starts_with(&prefix))
            .map(|t| (t.name[prefix.len()..].to_string(), t.shape.clone()))
            .collect();
        set.sort();
        set
    };
    for layer in [0usize, 30] {
        let expected: Vec<(String, Vec<usize>)> = GDN_LAYER_SUFFIXES
            .iter()
            .map(|(suffix, shape)| (suffix.to_string(), shape.to_vec()))
            .collect();
        assert_eq!(layer_suffixes(layer), expected, "gdn layer {layer}");
    }
    for layer in [3usize, 31] {
        let expected: Vec<(String, Vec<usize>)> = ATTN_LAYER_SUFFIXES
            .iter()
            .map(|(suffix, shape)| (suffix.to_string(), shape.to_vec()))
            .collect();
        assert_eq!(layer_suffixes(layer), expected, "attn layer {layer}");
    }
}

#[test]
fn dpo_inventory_matches_ground_truth() {
    check_inventory(consts::DPO_MODEL_NAME);
}

#[test]
fn base_inventory_matches_ground_truth() {
    check_inventory(consts::BASE_MODEL_NAME);
}

#[test]
fn mmap_spot_load_materializes_embed() {
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &candle_core::Device::Cpu)
        .expect("mmap");
    let embed = weights
        .get((100352, 3840), "model.embed_tokens.weight")
        .expect("embed materializes at the pinned shape");
    assert_eq!(embed.dtype(), candle_core::DType::BF16);
    assert_eq!(embed.dims(), [100352, 3840]);
}
