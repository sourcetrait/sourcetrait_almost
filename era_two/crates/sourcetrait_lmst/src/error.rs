//! Crate-wide error types; module errors wrap into `LmstError` as they
//! land.
#[allow(unused_imports)]
use crate::*;

pub type LmstResult<T> = Result<T, LmstError>;

#[derive(Debug, snafu::Snafu)]
pub enum LmstError {
    #[snafu(transparent)]
    Io { source: io::Error },
    #[snafu(transparent)]
    Json { source: serde_json::Error },
    #[snafu(transparent)]
    Candle { source: candle_core::Error },
    #[snafu(transparent)]
    Lib { source: lib::LibAlmostError },
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
