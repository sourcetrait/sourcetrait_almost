use crate::*;

/// Crate-wide error; module errors wrap in as transparent variants.
#[derive(Debug, snafu::Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum QuestError {
    #[snafu(transparent)]
    Candle { source: candle_core::Error },

    #[snafu(transparent)]
    Hub { source: hf_hub::HFError },

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

pub type QuestResult<T> = Result<T, QuestError>;
