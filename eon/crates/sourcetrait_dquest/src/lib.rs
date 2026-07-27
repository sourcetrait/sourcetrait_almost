pub(crate) mod error;
pub(crate) mod preflight;
pub mod run;
pub(crate) mod style;

pub(crate) use std::{
    env,
    io::{
        self,
        IsTerminal,
        Write,
    },
    path::{
        Path,
        PathBuf,
    },
};

pub(crate) use sourcetrait_cert_lib as srcert;

pub(crate) use crate::error::{
    DquestError,
    DquestResult,
};

#[cfg(test)]
mod tests {
    mod preflight;
    mod run;
    mod style;
}
