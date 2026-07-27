//! The vocabulary both directions of the wire share.
//!
//! Everything here is DATA. The trait and the channel pair that used to
//! sit beside it described an era holding the model in the consumer's own
//! process, and the daemon holds it now, so what an era once implemented
//! is a transport conversation instead.
#[allow(unused_imports)]
use crate::*;

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    bitcode::Encode,
    bitcode::Decode,
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
    bitcode::Encode,
    bitcode::Decode,
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
    bitcode::Encode,
    bitcode::Decode,
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
    bitcode::Encode,
    bitcode::Decode,
)]
pub enum FinishReason {
    StopToken,
    SampleLen,
    Cancelled,
}
