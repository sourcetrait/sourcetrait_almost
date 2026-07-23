#[allow(unused_imports)]
use crate::*;

/// Checkpoint config.json model (transformers 4.57-era flat format:
/// rope_theta + rope_scaling block + explicit layer_types).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Olmo3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default)]
    pub attention_bias: bool,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    pub hidden_act: candle_nn::Activation,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    pub sliding_window: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
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
pub enum LayerType {
    #[serde(rename = "sliding_attention")]
    Sliding,
    #[serde(rename = "full_attention")]
    Full,
}

/// The config.json rope_scaling block; applies to full-attention layers only.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub beta_fast: Option<f64>,
    #[serde(default)]
    pub beta_slow: Option<f64>,
    #[serde(default)]
    pub attention_factor: Option<f64>,
}
