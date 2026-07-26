//! The one era-agnostic error surface every era raises through.

pub type BridgeResult<T> = Result<T, BridgeError>;

#[derive(Debug, snafu::Snafu)]
pub enum BridgeError {
    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
