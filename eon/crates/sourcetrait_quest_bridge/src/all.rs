//! The bridge surface every consumer needs: the entry trait and the
//! session.
use crate::*;

/// An era's end-use entry point, and the only way eon reaches an era.
pub trait Era {
    /// The era's identity, without loading anything.
    fn info(&self) -> EraInfo;

    /// Open a chat session; the model loads on its own engine thread.
    fn open_chat(&self, options: &ChatOptions) -> BridgeResult<ChatSession>;
}

/// One chat session's transport pair: channels rather than callbacks.
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
    /// Stop the in-flight generation early; the turn still reports.
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
    /// Streamed text; may be empty while a grapheme is pending.
    Chunk { text: String },
    /// A turn ended - naturally or by Cancel.
    TurnDone(TurnReport),
    /// A recoverable engine error; the session stays open.
    Error { message: String },
    /// The session is over, however it ended.
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EraInfo {
    pub era: String,
    pub model: String,
}

/// Session-open options: the suite's profile tokens and a budget.
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
