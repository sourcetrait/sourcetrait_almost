pub(crate) mod capability;
pub(crate) mod cli;
pub(crate) mod error;
pub mod run;
pub(crate) mod speculate;

#[allow(unused_imports)]
pub(crate) use std::{
    collections::{
        HashMap,
        VecDeque,
    },
    fs,
    io,
    path::{
        Path,
        PathBuf,
    },
};

pub(crate) use sourcetrait_lib_quest as lib;

pub(crate) use crate::capability::capability_run;
pub(crate) use crate::cli::{
    CapabilityCommand,
    CapabilityRunArgs,
    Cli,
    Command,
    SpeculateCommand,
    SpeculateRecordArgs,
    SpeculateSimulateArgs,
    SpeculateTokensArgs,
};
pub(crate) use crate::speculate::{
    speculate_record,
    speculate_simulate,
    speculate_tokens,
};
#[allow(unused_imports)]
pub(crate) use crate::error::{
    BquestError,
    BquestResult,
};

#[cfg(test)]
mod tests {
    mod speculate;
}
