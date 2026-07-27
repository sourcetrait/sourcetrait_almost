pub(crate) mod container;
pub(crate) mod error;
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

pub(crate) use crate::container::ContainerHandle;
pub(crate) use crate::error::{
    DquestError,
    DquestResult,
};

#[cfg(test)]
mod tests {
    mod material;
    mod preflight;
    mod run;
    mod serve;
    mod style;
}
