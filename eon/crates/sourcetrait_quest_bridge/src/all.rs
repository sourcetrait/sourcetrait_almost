//! The bridge surface every consumer needs: the Era entry trait and
//! the channel-shaped chat session models.
use crate::*;

/// An era's end-use entry point. Implementations live in per-era
/// crates (sourcetrait_quest_bridge_<era>); eon components interact
/// with an era through this API alone and never see era internals.
pub trait Era {
    /// The era's identity, without loading anything.
    fn info(&self) -> EraInfo;

    /// Open an interactive chat session. The model loads on the
    /// session's own engine thread: `Ready(EraInfo)` arrives on
    /// `events` once it is live, `Error` + `Closed` follow a load
    /// failure.
    fn open_chat(&self, options: &ChatOptions) -> BridgeResult<ChatSession>;
}

/// One chat session's transport pair. Channels, not callbacks - and
/// deliberately transport-shaped: the in-process pair later swaps for
/// a socket/TLS boundary (the dquest server) without touching
/// consumers. Dropping `requests` closes the session; the engine
/// answers `Closed` and exits.
#[derive(Debug)]
pub struct ChatSession {
    pub requests: tokio::sync::mpsc::Sender<ChatRequest>,
    pub events: tokio::sync::mpsc::Receiver<ChatEvent>,
}

/// Consumer -> engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRequest {
    /// The next user turn's text (raw; the era renders its own chat
    /// protocol).
    Turn { text: String },
    /// Stop the in-flight generation early; the turn still reports
    /// (`TurnDone` with `Cancelled`).
    Cancel,
    /// Drop the whole conversation context - a fresh session on the
    /// same loaded model.
    Reset,
    /// End the session; the engine answers `Closed` and exits.
    Close,
}

/// Engine -> consumer.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    /// The model is loaded and the session accepts turns.
    Ready(EraInfo),
    /// Streamed generation text, possibly empty while a multi-token
    /// grapheme is pending.
    Chunk { text: String },
    /// A turn ended - naturally or by Cancel.
    TurnDone(TurnReport),
    /// A recoverable engine error; the session stays open.
    Error { message: String },
    /// The session is over: Close honored, the requests sender
    /// dropped, or the model failed to load.
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EraInfo {
    pub era: String,
    pub model: String,
}

/// Session-open options: the suite's -d/-c/-s profile tokens (era
/// profiles own settings - no knob farm here) plus a decode-budget
/// override.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatOptions {
    pub dir: Option<String>,
    pub config: Option<String>,
    pub settings: Option<String>,
    pub sample_len: Option<usize>,
}

/// One turn's accounting.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnReport {
    pub finish: FinishReason,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

/// How a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    StopToken,
    SampleLen,
    Cancelled,
}
