pub(crate) mod cli;
pub(crate) mod config;
pub(crate) mod diff;
pub(crate) mod dump;
pub(crate) mod error;
pub(crate) mod load;
pub(crate) mod model;
pub(crate) mod rope;
pub mod run;
pub(crate) mod tensor_io;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod rope;
}

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
    model::HeatModel,
};

pub(crate) type Back = burn::backend::NdArray<f32>;
pub(crate) type BackDevice = burn::backend::ndarray::NdArrayDevice;
pub(crate) type Tensor1 = burn::tensor::Tensor<Back, 1>;
pub(crate) type Tensor2 = burn::tensor::Tensor<Back, 2>;
pub(crate) type Tensor3 = burn::tensor::Tensor<Back, 3>;
