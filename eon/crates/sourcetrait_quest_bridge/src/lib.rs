pub mod all;
pub mod cli;
pub mod client;
pub mod srvc;
pub mod tls;
pub mod tui;
pub mod wire;

mod error;

pub(crate) use std::path::Path;

pub(crate) use sourcetrait_cert_lib as srcert;

/// In scope so `from_pem_file` dispatches; still pathed where named.
pub(crate) use rustls_pki_types::pem::PemObject;

/// In scope so `next`, `send` and `close` dispatch on framed halves.
pub(crate) use futures_util::{
    SinkExt,
    StreamExt,
};

pub(crate) mod r {
    pub(crate) mod tls {
        pub(crate) use rustls_pki_types::{
            CertificateDer,
            PrivateKeyDer,
            ServerName,
        };
        pub(crate) use tokio_rustls::TlsConnector;
        pub(crate) use tokio_rustls::client::TlsStream as ClientStream;
    }
    pub(crate) mod tokio {
        pub(crate) use tokio::sync::mpsc::{
            Receiver,
            Sender,
            channel,
        };
        pub(crate) use tokio_util::bytes::BytesMut;
        pub(crate) use tokio_util::codec::{
            Decoder,
            Encoder,
            FramedRead,
            FramedWrite,
            LengthDelimitedCodec,
        };
        pub(crate) use tokio_util::sync::CancellationToken;
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
    OpenRefusedResponse,
    OpenRequest,
    OpenResponse,
    ResetRequest,
    ResetResponse,
    ServerFaultNotice,
    ServerNotice,
    ServerShutdownNotice,
    ServerToClient,
    TurnChunk,
    TurnFailedResponse,
    TurnRequest,
    TurnResponse,
};

pub use client::{
    TlsClientHandle,
    TlsClientOptions,
};
pub use tls::{
    LOOPBACK_NAME,
    client_config,
    loopback_server_name,
    material,
    server_config,
};

#[cfg(test)]
mod tests {
    mod client;
    mod tls;
    mod wire;
}
