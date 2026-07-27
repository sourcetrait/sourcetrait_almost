pub(crate) mod adapter;
pub(crate) mod chat;
pub(crate) mod checkpoint;
pub(crate) mod config;
pub mod consts;
pub(crate) mod error;
pub(crate) mod evict;
#[cfg(feature = "cuda")]
pub(crate) mod fused;
#[cfg(feature = "cuda")]
pub(crate) mod fused_prefill;
pub(crate) mod gdn;
pub(crate) mod generate;
#[cfg(feature = "cuda")]
pub(crate) mod graph;
pub(crate) mod load;
pub(crate) mod model {
    pub(crate) mod attn_layer;
    pub(crate) mod gdn_layer;
    pub(crate) mod hybrid;
    pub(crate) mod mask;
    pub(crate) mod mlp;
}
pub(crate) mod needle;
pub(crate) mod norms;
#[cfg(feature = "attn-profile")]
pub(crate) mod profile;
pub(crate) mod snapshot;
pub(crate) mod speculate;
pub(crate) mod tokenizer;

#[allow(unused_imports)]
pub(crate) use std::{
    collections::{
        HashMap,
        VecDeque,
    },
    env,
    fs,
    io,
    path::{
        Path,
        PathBuf,
    },
};

pub(crate) use candle_nn::Module;

pub(crate) use crate::model::attn_layer::AttnLayer;
pub(crate) use crate::model::gdn_layer::GdnLayer;
pub(crate) use crate::model::mask::{
    causal_mask,
    offset_causal_mask,
};
pub(crate) use crate::model::mlp::Mlp;

pub(crate) mod r {
    #[cfg(feature = "flash-attn")]
    pub(crate) mod flash {
        pub(crate) use candle_flash_attn::flash_attn;
    }
    pub(crate) mod sampling {
        pub(crate) use candle_transformers::generation::{
            LogitsProcessor,
            Sampling,
        };
    }
}

pub use sourcetrait_quest_core::{
    QuestCoreError,
    QuestCoreResult,
    channel,
    nu,
};

pub use crate::adapter::{
    adapter_path,
    load_weights,
};
pub use crate::chat::{
    AssistantSpan,
    ChatMessage,
    ChatRender,
    ChatRole,
    EncodedRender,
    TokenSpan,
    chat_continue,
    chat_render,
    chat_wrap,
    encode_render,
    resolve_stop_ids,
};
pub use crate::checkpoint::{
    LayerKind,
    OlmoHybridConfig,
    RopeParameters,
    load_config,
    model_dir,
};
pub use crate::config::{
    ConfigProfile,
    EvictionSettings,
    EvictionToml,
    GenerationToml,
    LibConfig,
    LibConfigToml,
    LibSettings,
    LibSettingsToml,
    SettingsProfile,
};
pub use crate::error::{
    LibQuestError,
    LibQuestResult,
};
pub use crate::load::{
    TensorInfo,
    mmap_weights,
    tensor_inventory,
};
pub use crate::generate::{
    FinishReason,
    GenerateOptions,
    Generation,
    GenerationReport,
    GenerationStep,
};
pub use crate::model::hybrid::{
    ContextMark,
    OlmoHybrid,
};
pub use crate::needle::{
    NeedleCellResult,
    NeedleKey,
    NeedleMode,
    NeedleSpec,
    run_needle_cells,
};
#[cfg(feature = "attn-profile")]
pub use crate::profile::{
    HeadMasses,
    LayerProfile,
};
pub use crate::snapshot::{
    RestoredContext,
    snapshot_path,
};
pub use crate::speculate::{
    DraftPolicy,
    LookupIndex,
    MAX_DRAFT,
};
pub use crate::tokenizer::{
    load_tokenizer,
    verify_token_map,
};

#[cfg(test)]
mod tests {
    mod adapter;
    mod chat;
    mod checkpoint;
    mod config;
    mod evict;
    #[cfg(feature = "cuda")]
    mod fused;
    #[cfg(feature = "cuda")]
    mod fused_prefill;
    mod gdn;
    mod generate;
    #[cfg(feature = "cuda")]
    mod graph;
    mod load;
    mod model;
    mod needle;
    mod norms;
    #[cfg(feature = "attn-profile")]
    mod profile;
    mod snapshot;
    mod speculate;
    mod tokenizer;
}
