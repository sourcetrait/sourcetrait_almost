pub(crate) mod error;

#[allow(unused_imports)]
pub(crate) use std::{
    io,
    path::{
        Path,
        PathBuf,
    },
};

pub use crate::error::{
    LibAlmostError,
    LibAlmostResult,
};
