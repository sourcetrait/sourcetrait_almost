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

    /// The era's identity, without loading anything.
    fn info() -> EraInfo;

    /// Build the engine. Called on the thread that will own it.
    fn engine() -> Result<Self::Engine, String>;
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

/// Session-open options: the suite's profile tokens and a budget.
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
