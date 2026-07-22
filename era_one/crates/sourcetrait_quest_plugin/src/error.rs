#[allow(unused_imports)]
use crate::*;

/// Crate-wide error: the exceptional paths only - a failed conformance
/// CHECK is a data-shaped CheckReport, never an error.
#[derive(Debug, snafu::Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum PluginError {
    #[snafu(display("typedef parse failed: {message}"))]
    Typedef { message: String },

    #[snafu(display("nuon parse failed: {message}"))]
    Nuon { message: String },

    #[snafu(display("nuon render failed: {message}"))]
    NuonRender { message: String },
}

pub type PluginResult<T> = Result<T, PluginError>;
