use crate::*;

/// dquest: the inference daemon serving quest and camp.
#[derive(Debug, clap::Parser)]
#[command(name = "dquest", version, about)]
struct Cli {}

/// Binary entry: start, reporting to stderr and exiting non-zero.
pub fn run() {
    let _cli = <Cli as clap::Parser>::parse();
    if let Err(error) = start() {
        eprintln!("dquest: {error}");
        std::process::exit(1);
    }
}

/// Satisfy the certificate precondition, then serve.
fn start() -> DquestResult<()> {
    let config_home = srcert::config_home().map_err(|source| DquestError::Cert {
        context: "resolving the configuration home".to_string(),
        source: Box::new(source),
    })?;

    let profile = preflight::ensure_profile(&config_home)?;
    if let preflight::Profile::Written { live } = &profile {
        return Err(DquestError::CertificateOwed {
            name: preflight::PROFILE.to_string(),
            profile: live.display().to_string(),
        });
    }
    eprintln!("dquest: srcert profile {}", profile.live().display());

    eprintln!("dquest: the serving half is not built yet");
    Ok(())
}
