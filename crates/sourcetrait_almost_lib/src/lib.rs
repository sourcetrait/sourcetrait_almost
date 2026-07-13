pub(crate) mod chat;
pub(crate) mod checkpoint;
pub(crate) mod config;
pub mod consts;
pub(crate) mod error;
pub(crate) mod evict;
pub(crate) mod generate;
#[cfg(feature = "cuda")]
pub(crate) mod graph;
pub(crate) mod hub;
pub(crate) mod load;
pub(crate) mod model;
pub(crate) mod needle;
#[cfg(feature = "attn-profile")]
pub(crate) mod profile;
pub(crate) mod rope;
pub(crate) mod speculate;
pub(crate) mod token_stream;
pub(crate) mod verify;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod chat;
    pub(crate) mod config;
    pub(crate) mod evict;
    #[cfg(feature = "cuda")]
    pub(crate) mod graph;
    pub(crate) mod model;
    pub(crate) mod needle;
    #[cfg(feature = "attn-profile")]
    pub(crate) mod profile;
    pub(crate) mod rope;
    pub(crate) mod speculate;
}

pub(crate) use std::{
    collections::{
        BTreeSet,
        HashMap,
        VecDeque,
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
    #[cfg(feature = "flash-attn")]
    pub(crate) mod flash {
        pub(crate) use candle_flash_attn::{
            flash_attn,
            flash_attn_windowed,
        };
    }
}

#[allow(unused_imports)]
pub(crate) use crate::{
    model::{
        banded_mask_values,
        sliding_trim_bounds,
        AttnMask,
        CacheMark,
    },
    rope::RopeTables,
    speculate::{
        DraftPolicy,
        LookupIndex,
    },
    token_stream::TokenStream,
};

pub use crate::{
    chat::{
        chat_continue,
        chat_wrap,
        resolve_stop_ids,
    },
    checkpoint::{
        LayerType,
        Olmo3Config,
        RopeScaling,
    },
    config::{
        load_config,
        load_settings,
        resolve_device,
        resolve_dtype,
        Config,
        DeviceConfig,
        DtypeConfig,
        EvictionConfig,
        GenerationConfig,
    },
    error::{
        AlmostError,
        AlmostResult,
    },
    evict::EvictionSettings,
    generate::{
        FinishReason,
        GenerateOptions,
        Generation,
        GenerationReport,
        GenerationStep,
    },
    hub::{
        default_model_dir,
        default_snapshots_dir,
        ensure_model,
        resolve_snapshot,
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
        RestoredContext,
        Settings,
    },
    needle::{
        needle,
        NeedleOptions,
    },
    verify::{
        verify,
        verify_graph,
        VerifyOptions,
    },
};
