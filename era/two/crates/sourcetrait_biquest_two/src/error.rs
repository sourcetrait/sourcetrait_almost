//! Crate-wide error types; module errors wrap in as they land.
#[allow(unused_imports)]
use crate::*;

pub type BiquestResult<T> = Result<T, BiquestError>;

#[derive(Debug, snafu::Snafu)]
pub enum BiquestError {
    #[snafu(transparent)]
    Io { source: io::Error },
    #[snafu(transparent)]
    Json { source: serde_json::Error },
    #[snafu(transparent)]
    Llm { source: llm::LibQuestError },
    #[snafu(transparent)]
    Harness { source: harness::HarnessQuestError },
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
