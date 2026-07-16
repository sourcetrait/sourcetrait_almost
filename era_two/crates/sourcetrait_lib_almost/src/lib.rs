pub(crate) mod chat;
pub(crate) mod checkpoint;
pub(crate) mod config;
pub mod consts;
pub(crate) mod error;
pub(crate) mod gdn;
pub(crate) mod generate;
#[cfg(feature = "cuda")]
pub(crate) mod graph;
pub(crate) mod load;
pub(crate) mod model;
pub(crate) mod norms;
pub(crate) mod tokenizer;

#[allow(unused_imports)]
pub(crate) use std::{
    collections::HashMap,
    env,
    fs,
    io,
    path::{
        Path,
        PathBuf,
    },
};

pub(crate) use candle_nn::Module;

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

pub use crate::chat::{
    chat_continue,
    chat_wrap,
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
    GenerationToml,
    LibConfig,
    LibConfigToml,
    LibSettings,
    LibSettingsToml,
    SettingsProfile,
};
pub use crate::error::{
    LibAlmostError,
    LibAlmostResult,
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
pub use crate::model::OlmoHybrid;
pub use crate::tokenizer::{
    load_tokenizer,
    verify_token_map,
};

#[cfg(test)]
mod tests {
    mod chat;
    mod checkpoint;
    mod config;
    mod gdn;
    mod generate;
    #[cfg(feature = "cuda")]
    mod graph;
    mod load;
    mod model;
    mod norms;
    mod tokenizer;
}
