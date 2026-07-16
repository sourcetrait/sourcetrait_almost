pub mod run;

pub(crate) mod checkpoint;
pub(crate) mod cli;
pub(crate) mod error;
pub(crate) mod generate;
pub(crate) mod pyenv;

#[allow(unused_imports)]
pub(crate) use sourcetrait_lib_almost as lib;

#[allow(unused_imports)]
pub(crate) use std::{
    env,
    fs,
    io::{
        self,
        BufRead,
        Read,
    },
    path::{
        Path,
        PathBuf,
    },
    process,
};

pub(crate) use crate::error::LastmostResult;
#[allow(unused_imports)]
pub(crate) use crate::error::LastmostError;
#[allow(unused_imports)]
pub(crate) use crate::checkpoint::{
    ModelPick,
    TensorInfo,
    read_added_tokens,
    read_config,
    read_safetensors_header,
    resolve_model_dir,
};
pub(crate) use crate::cli::{
    AttnPick,
    Cli,
    Command,
    DevicePick,
    DtypePick,
    GenerateArgs,
};
pub(crate) use crate::generate::generate;
pub(crate) use crate::pyenv::{
    env_python,
    materialize_pysrc,
};

#[cfg(test)]
mod tests {
    mod checkpoint;
}
