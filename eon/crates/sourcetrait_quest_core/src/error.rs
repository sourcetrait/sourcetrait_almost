//! Crate-wide error types; module errors wrap in as they land.
#[allow(unused_imports)]
use crate::*;

pub type QuestCoreResult<T> = Result<T, QuestCoreError>;

#[derive(Debug, snafu::Snafu)]
pub enum QuestCoreError {
    #[snafu(transparent)]
    Io { source: io::Error },
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
