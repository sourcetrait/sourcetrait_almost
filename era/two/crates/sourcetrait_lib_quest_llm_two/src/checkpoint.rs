//! The checkpoint's config.json, its policy validation, and its home.
#[allow(unused_imports)]
use crate::*;

/// The config.json model, in the transformers 5.x flat format.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OlmoHybridConfig {
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
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    pub layer_types: Vec<LayerKind>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub linear_allow_neg_eigval: bool,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl OlmoHybridConfig {
    pub(crate) fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GDN key projection width (heads x key head dim; 2880 pinned).
    pub(crate) fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// GDN value/gate projection width (heads x value head dim; 5760).
    pub(crate) fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// The NoPE gate: rope exists only when rope_theta is non-null.
    pub fn is_nope(&self) -> bool {
        self.rope_parameters
            .as_ref()
            .is_none_or(|rope| rope.rope_theta.is_none())
    }

    pub fn layer_kind(&self, layer_idx: usize) -> LayerKind {
        self.layer_types[layer_idx]
    }

    /// The sole-checkpoint policy checks, run before construction.
    pub fn validate(&self) -> LibQuestResult<()> {
        if self.num_kv_heads() != self.num_attention_heads {
            snafu::whatever!(
                "MHA-only by policy: {} kv heads vs {} attention heads",
                self.num_kv_heads(),
                self.num_attention_heads
            );
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            snafu::whatever!(
                "hidden_size {} not divisible by {} heads",
                self.hidden_size,
                self.num_attention_heads
            );
        }
        if self.linear_num_key_heads != self.linear_num_value_heads {
            snafu::whatever!(
                "equal GDN head counts by policy (no repeat_interleave machinery): \
                 {} key heads vs {} value heads",
                self.linear_num_key_heads,
                self.linear_num_value_heads
            );
        }
        if self.layer_types.len() != self.num_hidden_layers {
            snafu::whatever!(
                "layer_types carries {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        let linear = self
            .layer_types
            .iter()
            .filter(|kind| **kind == LayerKind::LinearAttention)
            .count();
        if linear == 0 || linear == self.num_hidden_layers {
            snafu::whatever!(
                "hybrid expects both layer kinds; {} of {} are linear_attention",
                linear,
                self.num_hidden_layers
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
pub enum LayerKind {
    #[serde(rename = "linear_attention")]
    LinearAttention,
    #[serde(rename = "full_attention")]
    FullAttention,
}

/// The config.json rope_parameters block, whose null theta is a signal.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// XDG data home, honoring the spec fallback (~/.local/share).
pub(crate) fn data_home() -> LibQuestResult<PathBuf> {
    if let Ok(dir) = env::var("XDG_DATA_HOME")
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    let Ok(home) = env::var("HOME") else {
        snafu::whatever!("neither XDG_DATA_HOME nor HOME is set");
    };
    Ok(PathBuf::from(home).join(".local/share"))
}

/// A checkpoint's local directory under the XDG data home.
pub fn model_dir(model_name: &str) -> LibQuestResult<PathBuf> {
    Ok(data_home()?
        .join(consts::MODELS_HOME_RELATIVE)
        .join(consts::MODEL_AUTHOR)
        .join(model_name))
}

/// Parse and validate a checkpoint dir's config.json.
pub fn load_config(model_dir: &Path) -> LibQuestResult<OlmoHybridConfig> {
    let config: OlmoHybridConfig =
        serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
    config.validate()?;
    Ok(config)
}
