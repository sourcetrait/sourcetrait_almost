pub(crate) mod cli;
pub mod run;

pub(crate) use std::{
    io::{
        self,
        Write,
    },
    path::PathBuf,
};

pub(crate) use sourcetrait_almost_lib as lib;

#[allow(unused_imports)]
pub(crate) use crate::cli::{
    Cli,
    Command,
    SnapshotAction,
};
