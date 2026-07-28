pub(crate) mod adapter;
pub mod channel;
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
pub mod nu;
#[cfg(feature = "attn-profile")]
pub(crate) mod profile;
pub(crate) mod questness {
    pub mod arbitrate;
    pub mod config;
    pub mod contract;
    pub mod evaluate;
    pub mod shape;
    pub mod turn;
}
pub mod railroad;
pub mod session;
pub(crate) mod snapshot;
pub(crate) mod speculate;
pub mod template;
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
    process,
    thread,
    time,
};

pub(crate) use candle_nn::Module;
pub(crate) use nu_protocol::CompareTypes;
pub(crate) use sourcetrait_quest_bridge as bridge;

pub(crate) use crate::channel::{
    Block,
    Envelope,
    Tag,
};
pub(crate) use crate::model::attn_layer::AttnLayer;
pub(crate) use crate::model::gdn_layer::GdnLayer;
pub(crate) use crate::model::mask::{
    causal_mask,
    offset_causal_mask,
};
pub(crate) use crate::model::mlp::Mlp;
pub(crate) use crate::questness::contract::{
    NuSignature,
    check_agreements,
};
pub(crate) use crate::questness::evaluate::QuestnessEvaluator;
pub(crate) use crate::questness::turn;

pub(crate) mod r {
    #[cfg(feature = "flash-attn")]
    pub(crate) mod flash {
        pub(crate) use candle_flash_attn::flash_attn;
    }
    pub(crate) mod hash {
        pub(crate) use xxhash_rust::xxh3::xxh3_64;
    }
    pub(crate) mod nu {
        pub(crate) use nu_parser::parse;
        pub(crate) use nu_protocol::{
            ast::Expr,
            debugger::WithoutDebug,
            engine::{
                EngineState,
                Stack,
                StateWorkingSet,
            },
        };
    }
    pub(crate) mod sampling {
        pub(crate) use candle_transformers::generation::{
            LogitsProcessor,
            Sampling,
        };
    }
}

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
pub use crate::questness::arbitrate::{
    Questness,
    Step,
};
pub use crate::questness::shape::{
    Shape,
    ShapeMember,
};
pub use crate::questness::turn::{
    Answer,
    Assembled,
    Binding,
    Request,
    thought_turn,
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
    mod channel;
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
    mod nu;
    #[cfg(feature = "attn-profile")]
    mod profile;
    mod questness {
        mod arbitrate;
        mod config;
        mod contract;
        mod evaluate;
        mod shape;
        mod turn;
    }
    mod railroad;
    mod session;
    mod snapshot;
    mod speculate;
    mod template;
    mod tokenizer;
}
