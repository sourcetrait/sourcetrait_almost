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
}
