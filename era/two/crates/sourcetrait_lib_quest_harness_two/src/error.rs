//! Crate-wide error types; module errors wrap into `HarnessQuestError` as
//! they land.
#[allow(unused_imports)]
use crate::*;

pub type HarnessQuestResult<T> = Result<T, HarnessQuestError>;

#[derive(Debug, snafu::Snafu)]
pub enum HarnessQuestError {
    #[snafu(transparent)]
    Io { source: io::Error },
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
