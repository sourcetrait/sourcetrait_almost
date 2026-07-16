pub(crate) mod cli;
pub(crate) mod config;
pub(crate) mod diff;
pub(crate) mod dump;
pub(crate) mod error;
pub(crate) mod load;
#[cfg(feature = "train")]
pub(crate) mod lora;
pub(crate) mod model;
pub(crate) mod rope;
pub mod run;
pub(crate) mod tensor_io;
#[cfg(feature = "train")]
pub(crate) mod train;
#[cfg(feature = "train")]
pub(crate) mod train_data;

pub(crate) use std::{
    collections::HashMap,
    path::{
        Path,
        PathBuf,
    },
    time::Instant,
};

#[allow(unused_imports)]
pub(crate) use crate::{
    cli::{
        Cli,
        Command,
    },
    config::{
        HeatConfig,
        HeatRopeScaling,
        LayerKind,
    },
    error::{
        HeatError,
        HeatResult,
    },
    load::Weights,
    model::{
        HeatLayer,
        HeatModel,
    },
};

#[cfg(feature = "train")]
#[allow(unused_imports)]
pub(crate) use crate::{
    lora::{
        LayerLora,
        LoraPair,
        ModelLora,
    },
    train::TrainOptions,
    train_data::Packed,
};

/// Reference backend: deterministic host f32, fully isolated from the CUDA
/// stack the oracle checks.
pub(crate) type CpuBack = burn::backend::NdArray<f32>;

/// Fast-oracle backend: bf16 element type so the weights fit Tier-A VRAM.
/// Isolation caveat: shares the driver/toolkit chain with the system under
/// test, and bf16 precision is comparison-grade, not reference-grade.
#[cfg(any(feature = "cuda", feature = "train-cuda"))]
pub(crate) type CudaBack = burn::backend::Cuda<burn::tensor::bf16>;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod rope;
    #[cfg(feature = "train")]
    pub(crate) mod train;
}
