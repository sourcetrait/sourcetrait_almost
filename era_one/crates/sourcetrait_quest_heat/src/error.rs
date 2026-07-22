#[allow(unused_imports)]
use crate::*;

/// Crate-wide error; module errors wrap in as transparent variants.
#[derive(Debug, snafu::Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum HeatError {
    #[snafu(transparent)]
    Io { source: std::io::Error },

    #[snafu(transparent)]
    Json { source: serde_json::Error },

    #[snafu(transparent)]
    Safetensors { source: safetensors::SafeTensorError },

    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

pub type HeatResult<T> = Result<T, HeatError>;
