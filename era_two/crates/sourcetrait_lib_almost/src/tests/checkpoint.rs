//! Config pins over the era-two checkpoint pair. HARD-FAIL when a
//! checkpoint is absent or drifted - the local files are ground truth.
use crate::*;

fn shared_pins(config: &OlmoHybridConfig) {
    assert_eq!(config.vocab_size, 100352);
    assert_eq!(config.hidden_size, 3840);
    assert_eq!(config.intermediate_size, 11008);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 30);
    assert_eq!(config.num_key_value_heads, Some(30));
    assert_eq!(config.rms_norm_eps, 1e-6);
    assert!(!config.attention_bias);
    assert!(!config.tie_word_embeddings);
    assert_eq!(config.linear_num_key_heads, 30);
    assert_eq!(config.linear_num_value_heads, 30);
    assert_eq!(config.linear_key_head_dim, 96);
    assert_eq!(config.linear_value_head_dim, 192);
    assert_eq!(config.linear_conv_kernel_dim, 4);
    assert!(config.linear_allow_neg_eigval);
    assert_eq!(config.eos_token_id, 100257);
    assert_eq!(config.pad_token_id, 100277);
    assert!(config.is_nope());
    assert_eq!(config.layer_types.len(), 32);
    for layer_idx in 0..32 {
        let expected = if layer_idx % 4 == 3 {
            LayerKind::FullAttention
        } else {
            LayerKind::LinearAttention
        };
        assert_eq!(config.layer_kind(layer_idx), expected, "layer {layer_idx}");
    }
}

#[test]
fn dpo_config_pins() {
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("dpo config parses + validates");
    shared_pins(&config);
    assert_eq!(config.max_position_embeddings, 32768);
    let rope = config.rope_parameters.as_ref().expect("rope block present");
    assert!(rope.rope_theta.is_none());
    assert_eq!(rope.rope_type.as_deref(), Some("default"));
}

#[test]
fn base_config_pins() {
    let dir = model_dir(consts::BASE_MODEL_NAME).expect("base dir");
    let config = load_config(&dir).expect("base config parses + validates");
    shared_pins(&config);
    assert_eq!(config.max_position_embeddings, 65536);
    let rope = config.rope_parameters.as_ref().expect("rope block present");
    assert!(rope.rope_theta.is_none());
    assert_eq!(rope.rope_type, None);
}
