//! Crate-wide error types.
use crate::*;

pub(crate) type CampResult<T> = Result<T, CampError>;

#[derive(Debug, snafu::Snafu)]
pub(crate) enum CampError {
    #[snafu(transparent)]
    Io { source: io::Error },
    #[snafu(transparent)]
    Bridge { source: bridge::BridgeError },
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
