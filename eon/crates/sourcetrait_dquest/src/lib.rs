pub(crate) mod container;
pub(crate) mod error;
pub(crate) mod log;
pub(crate) mod manager;
pub(crate) mod nom;
pub(crate) mod preflight;
pub mod run;
pub(crate) mod serve;
pub(crate) mod style;

pub(crate) use std::{
    env,
    io::{
        self,
        IsTerminal,
        Write,
    },
    path::{
        Path,
        PathBuf,
    },
};

pub(crate) use sourcetrait_cert_lib as srcert;
pub(crate) use sourcetrait_quest_bridge as bridge;
/// The one place an era is named; everything else says `EraBridge`.
pub(crate) use sourcetrait_quest_bridge_two::BridgeTwo as EraBridge;

/// In scope so `next`, `send` and `close` dispatch on framed halves.
pub(crate) use futures_util::{
    SinkExt,
    StreamExt,
};

pub(crate) mod r {
    pub(crate) mod tls {
        pub(crate) use tokio_rustls::TlsAcceptor;
        pub(crate) use tokio_rustls::server::TlsStream as ServerStream;
    }
    pub(crate) mod tokio {
        pub(crate) use tokio::sync::mpsc::{
            Sender,
            channel,
        };
        pub(crate) use tokio::sync::oneshot::Sender as OneShot;
        pub(crate) use tokio::sync::oneshot::Receiver as OneShotWait;
        pub(crate) use tokio::sync::oneshot::channel as oneshot;
        pub(crate) use tokio_util::codec::{
            FramedRead,
            FramedWrite,
        };
    }
}

/// The era-facing contract, which is the API's rather than ours. The
/// container runs whatever satisfies it and never learns which era did.
pub(crate) use bridge::all::Engine;

/// One Thinkspace's turn owner, shared by every session in that space.
///
/// It outlives a session deliberately: a session tears down as soon as
/// its turn is over, while the conversation belongs to the Thinkspace, so
/// a caller returning with the result of an ask resumes rather than
/// starting again.
pub(crate) type ThinkHarnessHandle =
    std::sync::Arc<tokio::sync::Mutex<Box<dyn bridge::all::ThinkHarness>>>;

pub(crate) use crate::container::ContainerHandle;
pub(crate) use crate::error::{
    DquestError,
    DquestResult,
};

#[cfg(test)]
mod tests {
    mod manager;
    mod material;
    mod nom;
    mod preflight;
    mod run;
    mod serve;
    mod style;
}
