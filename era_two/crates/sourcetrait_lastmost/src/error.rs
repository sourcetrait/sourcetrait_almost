//! Crate-wide error types; module errors wrap into `LastmostError` as they
//! land.
#[allow(unused_imports)]
use crate::*;

pub(crate) type LastmostResult<T> = Result<T, LastmostError>;

#[derive(Debug, snafu::Snafu)]
pub(crate) enum LastmostError {
    #[snafu(transparent)]
    Io { source: io::Error },
    #[snafu(transparent)]
    Json { source: serde_json::Error },
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
