//! The Request-to-Response seam between Questness and a client harness.
use crate::*;

/// A nu value on the wire, carried as its NUON spelling.
///
/// The newtype exists to control the serialization: a nu value's own
/// derive emits the engine's tagged form, spans included.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestNuValue(pub nu::Value);

impl QuestNuValue {
    pub fn new(value: nu::Value) -> Self {
        Self(value)
    }

    /// The value's NUON spelling, which is what crosses the seam.
    pub fn to_nuon(&self) -> LibQuestResult<String> {
        nu::to_nuon_text(&self.0)
    }

    /// Read one back from its NUON spelling.
    pub fn from_nuon(text: &str) -> LibQuestResult<Self> {
        Ok(Self(nu::from_nuon_text(text)?))
    }

    /// The type the value derives, as a typedef string.
    pub fn declared(&self) -> String {
        self.0.get_type().to_string()
    }
}

impl serde::Serialize for QuestNuValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let text = self.to_nuon().map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&text)
    }
}

impl<'de> serde::Deserialize<'de> for QuestNuValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <String as serde::Deserialize>::deserialize(deserializer)?;
        Self::from_nuon(&text).map_err(serde::de::Error::custom)
    }
}

/// One channel binding as it crosses the seam.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RequestBinding {
    /// The channel as nushell spells it: `$in` or `$args`.
    pub pass: String,
    pub value: QuestNuValue,
}

/// What Questness asks a client harness to run.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HarnessRequest {
    /// The def's name, which IS the mode.
    pub mode: String,
    /// The `<nu>` body as the model wrote it, definition and all.
    pub source: String,
    /// The def's declared output type, so the answer can be checked.
    pub output: String,
    pub bindings: Vec<RequestBinding>,
}

impl HarnessRequest {
    /// The value a channel carries into this request.
    pub fn binding(&self, pass: &str) -> Option<&QuestNuValue> {
        self.bindings
            .iter()
            .find(|binding| binding.pass == pass)
            .map(|binding| &binding.value)
    }
}

/// What a client harness answers with.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HarnessResponse {
    /// The value the mode produced, and the type it declares.
    Value {
        value: QuestNuValue,
        declared: String,
    },
    /// The mode did not run; the reason rides back as a diagnostic.
    Failed { kind: String, message: String },
}

impl HarnessResponse {
    /// Answer with a value, deriving the type it declares.
    pub fn value(value: nu::Value) -> Self {
        let carried = QuestNuValue::new(value);
        let declared = carried.declared();
        Self::Value {
            value: carried,
            declared,
        }
    }

    /// Answer with a failure the model can be told about.
    pub fn failed(kind: &str, message: &str) -> Self {
        Self::Failed {
            kind: kind.to_string(),
            message: message.to_string(),
        }
    }

    /// The `<output>` block this becomes, or the diagnostic instead.
    pub fn as_block(&self) -> Result<channel::Block, channel::Diagnostic> {
        match self {
            Self::Value { value, declared } => {
                let text = value
                    .to_nuon()
                    .map_err(|error| {
                        channel::Diagnostic::new("harness::render", None, &error.to_string())
                    })?;
                Ok(channel::Block::new(
                    channel::Tag::Output,
                    declared,
                    &channel::escape_content(&text),
                ))
            }
            Self::Failed { kind, message } => Err(channel::Diagnostic::new(kind, None, message)),
        }
    }
}

/// A client harness: it services one request and answers.
///
/// Questness is generic over this rather than boxing it, because a given
/// instance talks to one harness for its life.
pub trait ClientHarness {
    fn serve(&mut self, request: &HarnessRequest) -> LibQuestResult<HarnessResponse>;
}
