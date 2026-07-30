//! The channel forms as data models, spoken by both ends of a turn.
use crate::*;

/// A nu value, carried as its NUON spelling.
///
/// The newtype controls the serialization. A nu value's own derive emits
/// the engine's tagged form with a span on every node, and a span is a
/// byte offset into a file that existed in the sending process, so it is
/// meaningless to a receiver. NUON text has nowhere to put one.
#[derive(Debug, Clone, PartialEq)]
pub struct InferValue(pub nu_protocol::Value);

impl InferValue {
    pub fn new(value: nu_protocol::Value) -> Self {
        Self(value)
    }

    /// The value's NUON spelling, which is what crosses.
    pub fn to_nuon(&self) -> BridgeResult<String> {
        let engine_state = nu_protocol::engine::EngineState::new();
        let config = nuon::ToNuonConfig::default();
        match nuon::to_nuon(&engine_state, &self.0, config) {
            Ok(text) => Ok(text),
            Err(error) => snafu::whatever!("nuon render failed: {error}"),
        }
    }

    /// Read one back from its NUON spelling.
    pub fn from_nuon(text: &str) -> BridgeResult<Self> {
        match nuon::from_nuon(text, None) {
            Ok(value) => Ok(Self(value)),
            Err(error) => snafu::whatever!("nuon parse failed: {error}"),
        }
    }

    /// The type the value derives, as a typedef string.
    pub fn declared(&self) -> String {
        self.0.get_type().to_string()
    }
}

impl serde::Serialize for InferValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let text = self.to_nuon().map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&text)
    }
}

impl<'de> serde::Deserialize<'de> for InferValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <String as serde::Deserialize>::deserialize(deserializer)?;
        Self::from_nuon(&text).map_err(serde::de::Error::custom)
    }
}

/// `<|pass|>`: which of nushell's two input channels a form binds to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InferPass {
    In,
    Args,
}

impl InferPass {
    /// The channel as nushell spells it, which is what a block carries.
    pub fn spelling(&self) -> &'static str {
        match self {
            Self::In => "$in",
            Self::Args => "$args",
        }
    }

    /// The name a Liquid template binds it under; the sigil is nushell's
    /// and Liquid's grammar rejects it.
    pub fn binding(&self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Args => "args",
        }
    }

    /// Read one back from a block's payload; anything else is refused.
    pub fn parse(spelling: &str) -> BridgeResult<Self> {
        match spelling.trim() {
            "$in" => Ok(Self::In),
            "$args" => Ok(Self::Args),
            other => snafu::whatever!("a pass binds $in or $args, not {other:?}"),
        }
    }
}

/// `<|input|>` carrying NUON: a typed value the caller hands the model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferNuonInput(pub InferValue);

/// `<|input|>`, over the formats we speak.
///
/// Open by design: one variant is the model and another format is an
/// extension rather than a case this enum failed to enumerate.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InferInput {
    Nuon(InferNuonInput),
}

impl InferInput {
    /// The nu value this input carries, whatever format it arrived as.
    pub fn value(&self) -> &InferValue {
        match self {
            Self::Nuon(nuon) => &nuon.0,
        }
    }
}

/// `<|output|>` carrying NUON: a typed value, in either direction.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferNuonOutput(pub InferValue);

/// `<|output|>`, over the formats we speak. Open on the same terms.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InferOutput {
    Nuon(InferNuonOutput),
}

impl InferOutput {
    /// The nu value this output carries, whatever format it arrived as.
    pub fn value(&self) -> &InferValue {
        match self {
            Self::Nuon(nuon) => &nuon.0,
        }
    }
}

/// `<|config|>`: the runtime record a caller composes per turn.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferConfig(pub InferValue);

/// Unmarked prose, which is the checkpoint's own protocol.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InferText(pub String);

/// `<|liquid|>`: a template, filled from the channel it binds.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InferLiquid(pub String);

/// The `evaluate` body: nu the Thinkspace's own evaluator runs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InferNuEvaluate(pub String);

/// The `execute` body: nu the caller's harness runs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InferNuExecute(pub String);

/// The `call` body: nu the caller's harness runs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InferNuCall(pub String);

/// The `interact` body: nu the caller's harness runs, env surviving.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InferNuInteract(pub String);

/// `<|nu|>`: a definition, and which form its signature matched.
///
/// The variant is the mode rather than a name recovered from the source.
/// Only `Evaluate` is a think turn; the rest are asks that ride back to
/// the caller, and this type never decides where one runs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InferNu {
    Evaluate(InferNuEvaluate),
    Execute(InferNuExecute),
    Call(InferNuCall),
    Interact(InferNuInteract),
}

impl InferNu {
    /// Whether this form is a think turn, which only `evaluate` is.
    pub fn is_think(&self) -> bool {
        matches!(self, Self::Evaluate(_))
    }

    /// The def source as the model wrote it, definition and all.
    pub fn source(&self) -> &str {
        match self {
            Self::Evaluate(body) => &body.0,
            Self::Execute(body) => &body.0,
            Self::Call(body) => &body.0,
            Self::Interact(body) => &body.0,
        }
    }

    /// The mode this form names, which is the def's own head.
    pub fn mode(&self) -> &'static str {
        match self {
            Self::Evaluate(_) => "evaluate",
            Self::Execute(_) => "execute",
            Self::Call(_) => "call",
            Self::Interact(_) => "interact",
        }
    }
}

/// What a caller asks of one turn.
///
/// `output` is how a SEQUENCE continues: a caller that ran an `InferNu`
/// the model asked for sends the result back on it, and the Thinkspace's
/// conversation resumes where it left off.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferRequest {
    pub config: Option<InferConfig>,
    pub inputs: Vec<(InferPass, InferInput)>,
    pub text: Option<InferText>,
    pub output: Option<InferOutput>,
}

/// What a turn answers with.
///
/// `nu` is an ask rather than a result: the model wants the caller's own
/// engine to run it, and the turn pauses until a request carries the
/// answer back. Only a think turn is serviced without leaving.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferResponse {
    pub output: Option<InferOutput>,
    /// What an ask's def consumes, bound to the channels it declares.
    /// An `InferNu` with nothing bound to it is a function with no
    /// arguments to run on.
    pub inputs: Vec<(InferPass, InferInput)>,
    pub text: Option<InferText>,
    pub config: Option<InferConfig>,
    pub nu: Option<InferNu>,
    /// The model said it cannot produce conforming output.
    pub insufficient: bool,
    /// Absent while a turn is paused on an ask, since a turn that has
    /// not ended has no token counts or timings that are not fiction.
    pub report: Option<all::TurnReport>,
}
