//! Crate-wide error types; module errors wrap in as they land.
#[allow(unused_imports)]
use crate::*;

pub type QuestHarnessResult<T> = Result<T, QuestHarnessError>;

#[derive(Debug, snafu::Snafu)]
pub enum QuestHarnessError {
    #[snafu(transparent)]
    Io { source: io::Error },
    #[snafu(transparent)]
    Core {
        source: sourcetrait_quest_core::QuestCoreError,
    },
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
