//! Crate-wide error types; module errors wrap into `LibQuestError` as they
//! land.
#[allow(unused_imports)]
use crate::*;

pub type LibQuestResult<T> = Result<T, LibQuestError>;

#[derive(Debug, snafu::Snafu)]
pub enum LibQuestError {
    #[snafu(transparent)]
    Io { source: io::Error },
    #[snafu(transparent)]
    Json { source: serde_json::Error },
    #[snafu(transparent)]
    Candle { source: candle_core::Error },
    #[snafu(transparent)]
    Toml { source: toml::de::Error },
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
