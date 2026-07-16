pub(crate) mod chat;
pub(crate) mod cli;
pub(crate) mod config;
pub(crate) mod consts;
pub(crate) mod error;
pub(crate) mod generate;
pub(crate) mod hub;
pub(crate) mod model;
pub(crate) mod rope;
pub mod run;
pub(crate) mod token_stream;
pub(crate) mod verify;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod chat;
    pub(crate) mod model;
    pub(crate) mod rope;
}

pub(crate) use std::{
    collections::BTreeSet,
    io::{
        self,
        Write,
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
        pub(crate) use candle_transformers::{
            generation::{
                LogitsProcessor,
                Sampling,
            },
            utils::repeat_kv,
        };
    }
}

#[allow(unused_imports)]
pub(crate) use crate::{
    chat::{
        chat_wrap,
        resolve_stop_ids,
    },
    cli::{
        Cli,
        Command,
    },
    config::{
        LayerType,
        Olmo3Config,
        RopeScaling,
    },
    error::{
        CompatError,
        CompatResult,
    },
    generate::{
        generate,
        GenerateOptions,
    },
    hub::{
        default_model_dir,
        ensure_model,
        ModelPaths,
    },
    model::Model,
    rope::RopeTables,
    token_stream::TokenStream,
    verify::VerifyOptions,
};
