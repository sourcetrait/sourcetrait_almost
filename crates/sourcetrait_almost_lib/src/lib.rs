pub(crate) mod chat;
pub(crate) mod config;
pub mod consts;
pub(crate) mod error;
pub(crate) mod generate;
pub(crate) mod hub;
pub(crate) mod load;
pub(crate) mod model;
pub(crate) mod rope;
pub(crate) mod token_stream;
pub(crate) mod verify;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod chat;
    pub(crate) mod model;
    pub(crate) mod rope;
}

pub(crate) use std::{
    collections::{
        BTreeSet,
        HashMap,
    },
    io::{
        self,
    },
    path::{
        Path,
        PathBuf,
    },
    time::Instant,
};

pub(crate) use candle_core::Module;

pub(crate) mod r {
    pub(crate) mod candle {
        pub(crate) use candle_core::utils::cuda_is_available;
        pub(crate) use candle_nn::rotary_emb::rope;
        pub(crate) use candle_transformers::generation::{
            LogitsProcessor,
            Sampling,
        };
    }
}

#[allow(unused_imports)]
pub(crate) use crate::{
    model::{
        banded_mask_values,
        sliding_trim_bounds,
        AttnMask,
    },
    rope::RopeTables,
    token_stream::TokenStream,
};

pub use crate::{
    chat::{
        chat_wrap,
        resolve_stop_ids,
    },
    config::{
        LayerType,
        Olmo3Config,
        RopeScaling,
    },
    error::{
        AlmostError,
        AlmostResult,
    },
    generate::{
        FinishReason,
        GenerateOptions,
        Generation,
        GenerationReport,
        GenerationStep,
    },
    hub::{
        default_model_dir,
        ensure_model,
        ModelPaths,
    },
    load::{
        load_model,
        pick_device,
        pick_dtype,
        LoadedModel,
    },
    model::{
        Model,
        Settings,
    },
    verify::{
        verify,
        VerifyOptions,
    },
};
