//! Crate-wide error types; module errors wrap into `LibAlmostError` as they
//! land.
#[allow(unused_imports)]
use crate::*;

pub type LibAlmostResult<T> = Result<T, LibAlmostError>;

#[derive(Debug, snafu::Snafu)]
pub enum LibAlmostError {
    #[snafu(transparent)]
    Io { source: io::Error },
    #[snafu(transparent)]
    Json { source: serde_json::Error },
    #[snafu(transparent)]
    Candle { source: candle_core::Error },
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
