use crate::*;

/// dquest: the inference daemon serving quest and camp.
#[derive(Debug, clap::Parser)]
#[command(name = "dquest", version, about)]
struct Cli {}

/// How far startup got.
pub(crate) enum Started {
    /// Preconditions met; the daemon may serve.
    Serving,
    /// The operator owes it a certificate and has been told so.
    CertificateOwed,
}

/// Binary entry: start, saying nothing unless something is wrong.
pub fn run() {
    let _cli = <Cli as clap::Parser>::parse();
    match start() {
        Ok(Started::Serving) => {}
        Ok(Started::CertificateOwed) => std::process::exit(1),
        Err(error) => {
            style::fail(&format!("Unable to start: {error}"));
            std::process::exit(1);
        }
    }
}

/// Satisfy the certificate precondition, then serve.
fn start() -> DquestResult<Started> {
    let config_home = srcert::config_home().map_err(|source| DquestError::Cert {
        context: "resolving the configuration home".to_string(),
        source: Box::new(source),
    })?;

    let profile = preflight::ensure_profile(&config_home)?;
    if let preflight::Profile::Written { live } = &profile {
        style::fail(&certificate_owed(live));
        return Ok(Started::CertificateOwed);
    }
    Ok(Started::Serving)
}

/// The one thing this binary says to a user.
pub(crate) fn certificate_owed(live: &Path) -> String {
    format!(
        "Unable to start: no certificate for the {name} profile yet, so the \
         daemon cannot serve TLS.\nIts profile is at {path}; edit it if you \
         want different names, then mint from it:\n    srcert generate \
         {plain} <dir>\n    srcert install {plain} <dir>\nwhere <dir> is a \
         staging directory that install consumes.",
        name = style::named(preflight::PROFILE),
        path = style::resource(live),
        plain = preflight::PROFILE,
    )
}
