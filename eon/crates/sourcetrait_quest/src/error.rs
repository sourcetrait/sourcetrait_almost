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
        source: Box<lib::LibQuestError>,
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

impl From<lib::LibQuestError> for QuestPluginError {
    fn from(source: lib::LibQuestError) -> Self {
        Self::Lib {
            source: Box::new(source),
        }
    }
}

impl QuestPluginError {
    /// A repair envelope as an error, one row per diagnostic.
    ///
    /// The rows are kept rather than flattened into the message, because
    /// a caller that got a shape it did not ask for wants to know which
    /// member was missing rather than that something was.
    pub(crate) fn envelope(envelope: &lib::channel::Envelope) -> Self {
        Self::Envelope {
            rows: envelope
                .errors
                .iter()
                .map(|row| match &row.source {
                    Some(source) => format!("{} at {source}: {}", row.kind, row.message),
                    None => format!("{}: {}", row.kind, row.message),
                })
                .collect(),
        }
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
