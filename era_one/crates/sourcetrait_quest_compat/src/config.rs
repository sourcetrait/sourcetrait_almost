#[allow(unused_imports)]
use crate::*;

/// Checkpoint config.json model (transformers 4.57-era flat format:
/// rope_theta + rope_scaling block + explicit layer_types).
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Olmo3Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    #[serde(default)]
    pub(crate) attention_bias: bool,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    #[serde(default)]
    pub(crate) num_key_value_heads: Option<usize>,
    pub(crate) rms_norm_eps: f64,
    pub(crate) hidden_act: candle_nn::Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rope_theta: f64,
    #[serde(default)]
    pub(crate) rope_scaling: Option<RopeScaling>,
    pub(crate) sliding_window: usize,
    #[serde(default)]
    pub(crate) layer_types: Option<Vec<LayerType>>,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
}

impl Olmo3Config {
    pub(crate) fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Explicit layer_types wins; absent, the Olmo 3 default pattern
    /// (every 4th layer full attention) applies.
    pub(crate) fn layer_type(&self, layer_idx: usize) -> LayerType {
        match &self.layer_types {
            Some(types) => types[layer_idx],
            None => {
                if (layer_idx + 1).is_multiple_of(4) {
                    LayerType::Full
                } else {
                    LayerType::Sliding
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
pub(crate) enum LayerType {
    #[serde(rename = "sliding_attention")]
    Sliding,
    #[serde(rename = "full_attention")]
    Full,
}

/// The config.json rope_scaling block; applies to full-attention layers only.
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct RopeScaling {
    pub(crate) rope_type: String,
    #[serde(default)]
    pub(crate) factor: Option<f64>,
    #[serde(default)]
    pub(crate) original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub(crate) beta_fast: Option<f64>,
    #[serde(default)]
    pub(crate) beta_slow: Option<f64>,
    #[serde(default)]
    pub(crate) attention_factor: Option<f64>,
}
