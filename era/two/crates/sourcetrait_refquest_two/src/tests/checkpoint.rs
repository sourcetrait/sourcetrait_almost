//! Byte-locks over the era-two checkpoint pair. These tests HARD-FAIL when
//! a checkpoint is absent or drifted: the local files are ground truth the
//! whole era diffs against, so absence is an error, not a skip.
use crate::*;
use crate::checkpoint::data_home;

/// The added-token table shared by the DPO checkpoint and era-one olmo3.
const DPO_ADDED_TOKENS: [(u64, &str, bool); 22] = [
    (100256, "<|extra_id_0|>", false),
    (100257, "<|endoftext|>", true),
    (100258, "<|fim_prefix|>", true),
    (100259, "<|fim_middle|>", true),
    (100260, "<|fim_suffix|>", true),
    (100261, "|||PHONE_NUMBER|||", false),
    (100262, "|||EMAIL_ADDRESS|||", false),
    (100263, "|||IP_ADDRESS|||", false),
    (100264, "<|im_start|>", true),
    (100265, "<|im_end|>", true),
    (100266, "<functions>", false),
    (100267, "</functions>", false),
    (100268, "<function_calls>", false),
    (100269, "</function_calls>", false),
    (100270, "<|extra_id_1|>", false),
    (100271, "<|extra_id_2|>", false),
    (100272, "<|extra_id_3|>", false),
    (100273, "<|extra_id_4|>", false),
    (100274, "<|extra_id_5|>", false),
    (100275, "<|extra_id_6|>", false),
    (100276, "<|endofprompt|>", true),
    (100277, "<|pad|>", true),
];

/// The base checkpoint's table: no tool markers; extra_id_1..10 instead.
const BASE_ADDED_TOKENS: [(u64, &str, bool); 22] = [
    (100256, "<|extra_id_0|>", false),
    (100257, "<|endoftext|>", true),
    (100258, "<|fim_prefix|>", true),
    (100259, "<|fim_middle|>", true),
    (100260, "<|fim_suffix|>", true),
    (100261, "|||PHONE_NUMBER|||", false),
    (100262, "|||EMAIL_ADDRESS|||", false),
    (100263, "|||IP_ADDRESS|||", false),
    (100264, "<|im_start|>", true),
    (100265, "<|im_end|>", true),
    (100266, "<|extra_id_1|>", false),
    (100267, "<|extra_id_2|>", false),
    (100268, "<|extra_id_3|>", false),
    (100269, "<|extra_id_4|>", false),
    (100270, "<|extra_id_5|>", false),
    (100271, "<|extra_id_6|>", false),
    (100272, "<|extra_id_7|>", false),
    (100273, "<|extra_id_8|>", false),
    (100274, "<|extra_id_9|>", false),
    (100275, "<|extra_id_10|>", false),
    (100276, "<|endofprompt|>", true),
    (100277, "<|pad|>", true),
];

const SHARD_FILE_BYTES: u64 = 14_861_802_680;
const SHARD_DATA_BYTES: u64 = 14_861_741_376;
const TENSOR_COUNT: usize = 523;

const GDN_LAYER_SUFFIXES: [(&str, &[u64]); 18] = [
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

const ATTN_LAYER_SUFFIXES: [(&str, &[u64]); 11] = [
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

fn shared_config_pins(config: &serde_json::Value) {
    assert_eq!(config["model_type"], "olmo_hybrid");
    assert_eq!(config["architectures"][0], "OlmoHybridForCausalLM");
    assert_eq!(config["hidden_size"], 3840);
    assert_eq!(config["intermediate_size"], 11008);
    assert_eq!(config["num_hidden_layers"], 32);
    assert_eq!(config["num_attention_heads"], 30);
    assert_eq!(config["num_key_value_heads"], 30);
    assert_eq!(config["vocab_size"], 100352);
    assert_eq!(config["rms_norm_eps"], 1e-6);
    assert_eq!(config["hidden_act"], "silu");
    assert_eq!(config["attention_bias"], false);
    assert_eq!(config["tie_word_embeddings"], false);
    assert_eq!(config["eos_token_id"], 100257);
    assert_eq!(config["pad_token_id"], 100277);
    assert_eq!(config["linear_num_key_heads"], 30);
    assert_eq!(config["linear_num_value_heads"], 30);
    assert_eq!(config["linear_key_head_dim"], 96);
    assert_eq!(config["linear_value_head_dim"], 192);
    assert_eq!(config["linear_conv_kernel_dim"], 4);
    assert_eq!(config["linear_allow_neg_eigval"], true);
    assert!(config["rope_parameters"]["rope_theta"].is_null());

    let layer_types = config["layer_types"].as_array().expect("layer_types array");
    assert_eq!(layer_types.len(), 32);
    for (i, layer_type) in layer_types.iter().enumerate() {
        let expected = if i % 4 == 3 { "full_attention" } else { "linear_attention" };
        assert_eq!(layer_type, expected, "layer {i}");
    }
}

fn check_shard(model_dir: &Path) {
    let shard = model_dir.join("model.safetensors");
    assert_eq!(fs::metadata(&shard).expect("shard present").len(), SHARD_FILE_BYTES);

    let tensors = read_safetensors_header(&shard).expect("header parses");
    assert_eq!(tensors.len(), TENSOR_COUNT);
    assert!(tensors.iter().all(|t| t.dtype == "BF16"));
    assert_eq!(tensors.iter().map(TensorInfo::byte_len).sum::<u64>(), SHARD_DATA_BYTES);

    let shape_of = |name: &str| -> Vec<u64> {
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

    let layer_suffixes = |layer: usize| -> Vec<(String, Vec<u64>)> {
        let prefix = format!("model.layers.{layer}.");
        let mut set: Vec<(String, Vec<u64>)> = tensors
            .iter()
            .filter(|t| t.name.starts_with(&prefix))
            .map(|t| (t.name[prefix.len()..].to_string(), t.shape.clone()))
            .collect();
        set.sort();
        set
    };
    for layer in [0usize, 1, 30] {
        let expected: Vec<(String, Vec<u64>)> = GDN_LAYER_SUFFIXES
            .iter()
            .map(|(s, shape)| (s.to_string(), shape.to_vec()))
            .collect();
        assert_eq!(layer_suffixes(layer), expected, "gdn layer {layer}");
    }
    for layer in [3usize, 31] {
        let expected: Vec<(String, Vec<u64>)> = ATTN_LAYER_SUFFIXES
            .iter()
            .map(|(s, shape)| (s.to_string(), shape.to_vec()))
            .collect();
        assert_eq!(layer_suffixes(layer), expected, "attn layer {layer}");
    }
}

#[test]
fn data_home_resolves() {
    assert!(data_home().expect("data home").is_absolute());
}

#[test]
fn dpo_config_pins() {
    let dir = resolve_model_dir(ModelPick::Dpo).expect("dpo dir");
    let config = read_config(&dir).expect("dpo config.json");
    shared_config_pins(&config);
    assert_eq!(config["max_position_embeddings"], 32768);
    assert_eq!(config["rope_parameters"]["rope_type"], "default");
    assert_eq!(config["dtype"], "bfloat16");
}

#[test]
fn base_config_pins() {
    let dir = resolve_model_dir(ModelPick::Base).expect("base dir");
    let config = read_config(&dir).expect("base config.json");
    shared_config_pins(&config);
    assert_eq!(config["max_position_embeddings"], 65536);
    assert!(config["rope_parameters"]["rope_type"].is_null());
}

#[test]
fn dpo_added_token_map() {
    let dir = resolve_model_dir(ModelPick::Dpo).expect("dpo dir");
    let added = read_added_tokens(&dir).expect("dpo added tokens");
    let expected: Vec<(u64, String, bool)> = DPO_ADDED_TOKENS
        .iter()
        .map(|(id, content, special)| (*id, content.to_string(), *special))
        .collect();
    assert_eq!(added, expected);
}

#[test]
fn base_added_token_map() {
    let dir = resolve_model_dir(ModelPick::Base).expect("base dir");
    let added = read_added_tokens(&dir).expect("base added tokens");
    let expected: Vec<(u64, String, bool)> = BASE_ADDED_TOKENS
        .iter()
        .map(|(id, content, special)| (*id, content.to_string(), *special))
        .collect();
    assert_eq!(added, expected);
}

#[test]
fn dpo_chat_template_bytes() {
    let dir = resolve_model_dir(ModelPick::Dpo).expect("dpo dir");
    let bytes = fs::read(dir.join("chat_template.jinja")).expect("template present");
    assert_eq!(bytes.as_slice(), &include_bytes!("fixtures/chat_template_dpo.jinja")[..]);
}

#[test]
fn dpo_shard_header() {
    check_shard(&resolve_model_dir(ModelPick::Dpo).expect("dpo dir"));
}

#[test]
fn base_shard_header() {
    check_shard(&resolve_model_dir(ModelPick::Base).expect("base dir"));
}
