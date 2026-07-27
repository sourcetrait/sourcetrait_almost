pub mod all;
pub mod cli;
pub mod srvc;
pub mod tui;
pub mod wire;

mod error;

pub(crate) mod r {
    pub(crate) mod tokio {
        pub(crate) use tokio_util::bytes::BytesMut;
        pub(crate) use tokio_util::codec::{
            Decoder,
            Encoder,
            LengthDelimitedCodec,
        };
    }
}

pub use error::{
    BridgeError,
    BridgeResult,
};
pub use wire::{
    BitcodeCodec,
    CancelRequest,
    CancelResponse,
    ClientToServer,
    MAX_FRAME_BYTES,
    OpenRequest,
    OpenResponse,
    ResetRequest,
    ResetResponse,
    ServerFaultNotice,
    ServerNotice,
    ServerShutdownNotice,
    ServerToClient,
    TurnChunk,
    TurnRequest,
    TurnResponse,
};

#[cfg(test)]
mod tests {
    mod wire;
}
