//! The wire: length-delimited frames carrying bitcode-encoded messages.
use crate::*;

/// A frame larger than this is refused before anything buffers it.
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

/// Open a session; the model loads behind it.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct OpenRequest {
    pub options: all::ChatOptions,
}

/// The session is open and accepts turns.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct OpenResponse {
    pub info: all::EraInfo,
}

/// The session did not open, so nothing is loaded behind it.
///
/// Its own message rather than an outcome inside `OpenResponse`, because
/// a refusal has no era to report and a caller that must branch anyway
/// is better served branching on the message.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct OpenRefusedResponse {
    pub message: String,
}

/// The turn did not run to an ending.
///
/// Its own message because a failed turn has no accounting to give -
/// token counts and timings would be fiction - so folding it into
/// `TurnResponse` would mean inventing a report to carry a failure.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct TurnFailedResponse {
    pub message: String,
}

/// Stop the in-flight generation early; the turn still answers.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct CancelRequest;

/// The cancel was received; whether a turn was running rides `stopped`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct CancelResponse {
    pub stopped: bool,
}

/// Drop the conversation context, keeping the loaded model.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct ResetRequest;

/// The context is dropped and the session accepts turns again.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct ResetResponse;

/// A piece of the in-flight turn's answer.
///
/// Not a notice: it belongs to the `TurnRequest` that is running, and
/// many of them arrive before that request's own response.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct TurnChunk {
    pub text: String,
}

/// A recoverable engine fault; the session stays open.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct ServerFaultNotice {
    pub message: String,
}

/// The daemon is going down and will accept no further turns.
///
/// Distinct from a transport `Close`, which ends one connection: this
/// says the far end itself is leaving, so reconnecting will not help.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct ServerShutdownNotice {
    pub reason: String,
}

/// What the daemon states without being asked.
///
/// Grouped rather than flattened into `ServerToClient` so the split
/// between an ANSWER and a NOTICE is structural, and so a new notice
/// does not widen the top-level language.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum ServerNotice {
    Fault(ServerFaultNotice),
    Shutdown(ServerShutdownNotice),
}

/// Consumer to daemon. Close is the transport's, not the session's.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum ClientToServer {
    Open(OpenRequest),
    Turn(InferRequest),
    Cancel(CancelRequest),
    Reset(ResetRequest),
    Close,
}

/// Daemon to consumer: one answer per request, plus what nobody asked
/// for. Close is the transport's, not the session's.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum ServerToClient {
    Open(OpenResponse),
    OpenRefused(OpenRefusedResponse),
    Turn(InferResponse),
    TurnFailed(TurnFailedResponse),
    TurnChunk(TurnChunk),
    Cancel(CancelResponse),
    Reset(ResetResponse),
    Notice(ServerNotice),
    Close,
}

/// One message per frame: the length codec delimits, bitcode fills.
///
/// Typed per direction, so a half that could only ever send one of them
/// cannot be handed the other.
pub struct BitcodeCodec<T> {
    codec: r::tokio::LengthDelimitedCodec,
    _marker: std::marker::PhantomData<T>,
}

impl<T> BitcodeCodec<T> {
    pub fn new() -> Self {
        Self {
            codec: r::tokio::LengthDelimitedCodec::builder()
                .max_frame_length(MAX_FRAME_BYTES)
                .new_codec(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> Default for BitcodeCodec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::fmt::Debug for BitcodeCodec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitcodeCodec").finish_non_exhaustive()
    }
}

impl<T> r::tokio::Decoder for BitcodeCodec<T>
where
    T: serde::de::DeserializeOwned,
{
    type Item = T;
    type Error = std::io::Error;

    fn decode(
        &mut self,
        src: &mut r::tokio::BytesMut,
    ) -> Result<Option<Self::Item>, Self::Error> {
        let Some(frame) = self.codec.decode(src)? else {
            return Ok(None);
        };
        bitcode::deserialize(&frame).map(Some).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("a frame did not decode: {error}"),
            )
        })
    }
}

impl<T> r::tokio::Encoder<T> for BitcodeCodec<T>
where
    T: serde::Serialize,
{
    type Error = std::io::Error;

    fn encode(
        &mut self,
        item: T,
        dst: &mut r::tokio::BytesMut,
    ) -> Result<(), Self::Error> {
        let bytes = bitcode::serialize(&item).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("a message did not encode: {error}"),
            )
        })?;
        self.codec.encode(bytes.into(), dst)
    }
}
