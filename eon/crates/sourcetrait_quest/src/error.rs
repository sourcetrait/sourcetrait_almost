//! Crate-wide error types, and how one reaches a nushell caller.
#[allow(unused_imports)]
use crate::*;

pub type QuestPluginResult<T> = Result<T, QuestPluginError>;

#[derive(Debug, snafu::Snafu)]
pub enum QuestPluginError {
    #[snafu(transparent)]
    Bridge {
        source: Box<bridge::BridgeError>,
    },

    #[snafu(transparent)]
    Lib {
        source: Box<harness_lib::HarnessQuestError>,
    },

    #[snafu(transparent)]
    Io { source: std::io::Error },

    #[snafu(display("the answer did not conform to the declared shape"))]
    Envelope { rows: Vec<String> },

    #[snafu(display("the model could not answer from the context it was given"))]
    Insufficient,

    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl From<bridge::BridgeError> for QuestPluginError {
    fn from(source: bridge::BridgeError) -> Self {
        Self::Bridge {
            source: Box::new(source),
        }
    }
}

impl From<harness_lib::HarnessQuestError> for QuestPluginError {
    fn from(source: harness_lib::HarnessQuestError) -> Self {
        Self::Lib {
            source: Box::new(source),
        }
    }
}

impl QuestPluginError {
    /// A shortfall against the declared shape, one row per member.
    ///
    /// The rows are kept rather than flattened into the message, because
    /// a caller that got a shape it did not ask for wants to know which
    /// member was missing rather than that something was.
    pub(crate) fn rows(rows: Vec<String>) -> Self {
        Self::Envelope { rows }
    }
}

/// Every failure reaches a caller as a `LabeledError`, never as text.
///
/// A plugin's stdout is the msgpack protocol channel, so printing a
/// diagnostic there corrupts the protocol rather than informing anyone.
/// This is the only route out.
impl From<QuestPluginError> for nu_protocol::LabeledError {
    fn from(error: QuestPluginError) -> Self {
        let labelled = nu_protocol::LabeledError::new(error.to_string());
        match error {
            QuestPluginError::Envelope { rows } => labelled.with_help(rows.join("\n")),
            _ => labelled,
        }
    }
}
