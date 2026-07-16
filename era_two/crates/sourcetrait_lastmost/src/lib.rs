pub mod run;

pub(crate) mod bench;
pub(crate) mod checkpoint;
pub(crate) mod cli;
pub(crate) mod diff;
pub(crate) mod driver;
pub(crate) mod dump;
pub(crate) mod error;
pub(crate) mod eval;
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
pub(crate) use crate::bench::bench;
pub(crate) use crate::cli::{
    AttnPick,
    BackendPick,
    BenchArgs,
    Cli,
    Command,
    DevicePick,
    DiffArgs,
    DtypePick,
    DumpArgs,
    EnvArgs,
    EvalArgs,
    EvalSuite,
    GenerateArgs,
    ModePick,
    NeedleArgs,
};
pub(crate) use crate::diff::diff;
pub(crate) use crate::driver::run_driver;
pub(crate) use crate::dump::dump;
pub(crate) use crate::eval::eval;
pub(crate) use crate::generate::generate;
pub(crate) use crate::pyenv::{
    env_python,
    envcheck,
    materialize_pysrc,
};

#[cfg(test)]
mod tests {
    mod checkpoint;
}
