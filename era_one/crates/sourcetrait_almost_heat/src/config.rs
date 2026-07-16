use crate::*;

/// Minimal, independent read of the checkpoint config.json (deliberately
/// not shared with compat: the oracle parses its own inputs).
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct HeatConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    /// Unused by the forward pass (weight shapes carry it); kept so the
    /// config read stays a faithful mirror.
    #[allow(dead_code)]
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    #[serde(default)]
    pub(crate) num_key_value_heads: Option<usize>,
    pub(crate) rms_norm_eps: f64,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rope_theta: f64,
    #[serde(default)]
    pub(crate) rope_scaling: Option<HeatRopeScaling>,
    pub(crate) sliding_window: usize,
    #[serde(default)]
    pub(crate) layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct HeatRopeScaling {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    Sliding,
    Full,
}

impl HeatConfig {
    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub(crate) fn layer_kind(&self, layer_idx: usize) -> HeatResult<LayerKind> {
        match &self.layer_types {
            Some(types) => match types[layer_idx].as_str() {
                "sliding_attention" => Ok(LayerKind::Sliding),
                "full_attention" => Ok(LayerKind::Full),
                other => snafu::whatever!("unknown layer type {other}"),
            },
            None => Ok(if (layer_idx + 1).is_multiple_of(4) {
                LayerKind::Full
            } else {
                LayerKind::Sliding
            }),
        }
    }
}
