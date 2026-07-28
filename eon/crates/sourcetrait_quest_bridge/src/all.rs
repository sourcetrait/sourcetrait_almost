//! THE api: the traits an era bridge implements, the generics it fills,
//! and the model types both ends traffic in.
//!
//! The flow this exists to hold is `eon component -> eon bridge API <-
//! era bridge -> era component`. An eon component programs against what
//! is in this file and never names an era's library; an era bridge
//! implements it against that library. That is what makes another era a
//! swapped dependency rather than an edit to every consumer.
#[allow(unused_imports)]
use crate::*;

/// What an era supplies for something else to drive.
///
/// `?Send` BY NATURE rather than by choice: an era's model may hold
/// thread-bound handles, so one thread owns an engine for its life and a
/// consumer moves the FACTORY rather than the engine.
pub trait Engine {
    fn open(&mut self, options: &ChatOptions) -> Result<EraInfo, String>;

    /// Emit through `chunk` as the answer forms; return the accounting.
    ///
    /// The callback answers whether anyone is still listening, which is
    /// how a consumer cancels without a second channel.
    fn turn(
        &mut self,
        text: &str,
        chunk: &mut dyn FnMut(TurnChunk) -> bool,
    ) -> Result<TurnReport, String>;

    fn reset(&mut self) -> Result<(), String>;
}

/// An era bridge: the entry point an eon component is generic over.
///
/// THIS IS THE GENERIC THE API PROVIDES. A consumer takes `E: Era` and
/// names a concrete era bridge at exactly one instantiation point, so
/// swapping eras is one line rather than a sweep - and nothing in the
/// consumer can reach past this into an era's library, because there is
/// no path from here to one.
///
/// The associated engine is built rather than handed over, because a
/// `FnOnce` returning a not-`Send` value is itself `Send` while the value
/// is not. That is what lets the engine be constructed on the thread that
/// will own it.
pub trait Era {
    type Engine: Engine;
    type Questness: Questness;

    /// The era's identity, without loading anything.
    fn info() -> EraInfo;

    /// Build the engine. Called on the thread that will own it.
    ///
    /// The options decide WHAT LOADS, which is why they arrive here
    /// rather than at `open`. A consumer holding one engine for the
    /// service's life has already built it by the time a session opens,
    /// so the same fields carried on `OpenRequest` cannot be honoured
    /// there and `Engine::open` documents them as unconsulted.
    fn engine(options: &ChatOptions) -> Result<Self::Engine, String>;

    /// Build a Questness. One per Thinkspace, carrying that space's
    /// conversation, so a consumer builds one per space rather than
    /// sharing one.
    fn questness() -> Result<Self::Questness, String>;
}

/// One repair row, addressed by cell-path rather than by span.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct InferDiagnostic {
    pub kind: String,
    /// The cell-path into the offending value, where one applies.
    pub source: Option<String>,
    pub message: String,
}

/// What one emission moved a turn to.
///
/// The split is by WHO ACTS NEXT. `Continue` is the only state that
/// keeps the turn going, and it carries the text to feed the model
/// again; everything else hands control back to the consumer.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// A think turn already ran. Feed this back to the model.
    Continue(String),
    Answered {
        output: Option<InferOutput>,
        text: Option<InferText>,
        config: Option<InferConfig>,
    },
    /// The model asked the CONSUMER's own engine to run something. The
    /// turn pauses until a request carries the result back.
    Ask {
        form: InferNu,
        inputs: Vec<(InferPass, InferInput)>,
    },
    /// The model said it cannot produce conforming output.
    Insufficient,
    /// The emission did not conform; these rows are the feedback.
    Repair(Vec<InferDiagnostic>),
}

/// The turn surface a consumer drives, one instance per Thinkspace.
///
/// It renders what the model is shown, reads what it emits, and runs the
/// one form that stays inside. It never holds the model: a consumer
/// generates against its own engine and hands the emission back here.
///
/// `?Send` is deliberately NOT wanted - a consumer drives this from a
/// blocking task, so the value has to cross a thread boundary.
pub trait Questness: Send {
    /// The text one turn shows the model.
    fn assemble(&mut self, request: &InferRequest) -> Result<String, String>;

    /// Read one emission and take the turn to its next state.
    ///
    /// `insufficient` is supplied rather than scanned for: the tag is
    /// special, so a skip-special decode strips it and only the side
    /// holding token ids can know.
    fn step(&mut self, emission: &str, insufficient: bool) -> Result<Step, String>;
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct EraInfo {
    pub era: String,
    pub model: String,
}

/// Profile tokens and a budget: what a consumer asks an era to load.
#[derive(
    Debug,
    Clone,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct ChatOptions {
    pub dir: Option<String>,
    pub config: Option<String>,
    pub settings: Option<String>,
    pub sample_len: Option<usize>,
}

/// One turn's accounting.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct TurnReport {
    pub finish: FinishReason,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

/// How a turn ended.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum FinishReason {
    StopToken,
    SampleLen,
    Cancelled,
}
