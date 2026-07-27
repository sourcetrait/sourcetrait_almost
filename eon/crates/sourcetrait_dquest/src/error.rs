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
        "no certificate for the {name:?} profile yet, so the daemon cannot \
         serve TLS.\nIts profile is at {profile}; edit it if you want \
         different names, then mint from it:\n    srcert generate {name} \
         <dir>\n    srcert install {name} <dir>\nwhere <dir> is a staging \
         directory that install consumes."
    ))]
    CertificateOwed { name: String, profile: String },
}
