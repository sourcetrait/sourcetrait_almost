//! Crate-wide error types; module errors wrap in as they land.
#[allow(unused_imports)]
use crate::*;

pub type DquestResult<T> = Result<T, DquestError>;

#[derive(Debug, snafu::Snafu)]
pub enum DquestError {
    #[snafu(display("{context}: {source}"))]
    Cert {
        context: String,
        source: Box<srcert::CertError>,
    },

    #[snafu(display(
        "${variable} is not set, so the daemon cannot locate its certificate \
         material"
    ))]
    Unset { variable: &'static str },

    #[snafu(display("binding {address}: {source}"))]
    Listen {
        address: std::net::SocketAddr,
        source: std::io::Error,
    },

    #[snafu(transparent)]
    Bridge {
        source: Box<bridge::BridgeError>,
    },

    #[snafu(display("starting the runtime: {source}"))]
    Runtime { source: std::io::Error },

    #[snafu(transparent)]
    Io { source: std::io::Error },

    #[snafu(display("a session path segment is a plain name; got {segment:?}"))]
    Segment { segment: String },
}
