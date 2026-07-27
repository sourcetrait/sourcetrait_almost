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

/// The variable naming the user whose Thinkspace this serves.
pub(crate) const USER_ENV: &str = "USER";

/// Binary entry: start, saying nothing unless something is wrong.
pub fn run() {
    let _cli = <Cli as clap::Parser>::parse();
    let outcome = match start() {
        Ok(Started::Serving) => serve_forever(),
        Ok(Started::CertificateOwed) => std::process::exit(1),
        Err(error) => Err(error),
    };
    if let Err(error) = outcome {
        style::fail(&format!("Unable to start: {error}"));
        std::process::exit(1);
    }
}

/// Serve until the listener dies, which for a daemon is until it is
/// stopped.
fn serve_forever() -> DquestResult<()> {
    let secret_data = preflight::secret_data_home()?;
    let files = bridge::material(&secret_data, preflight::PROFILE);
    let config = bridge::server_config(&files).map_err(|source| DquestError::Bridge {
        source: Box::new(source),
    })?;
    let username = match env::var(USER_ENV) {
        Ok(name) if !name.trim().is_empty() => name,
        _ => return Err(DquestError::Unset { variable: USER_ENV }),
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|source| DquestError::Runtime { source })?;

    runtime.block_on(async move {
        let listener = serve::bind(bridge::LOOPBACK_ADDRESS).await?;
        // The engine loads on the container's own thread; a failure
        // there leaves the daemon up and refusing, rather than exiting
        // before any client can be told why. The factory comes from the
        // API's `Era` trait, so this is the only line that names one.
        let container = ContainerHandle::spawn(<EraBridge as bridge::all::Era>::engine);
        let manager =
            manager::SessionManager::for_user(&username, preflight::session_log_root());
        serve::serve(listener, config, container, manager).await;
        Ok(())
    })
}

/// Satisfy the certificate preconditions, then serve.
fn start() -> DquestResult<Started> {
    let config_home = srcert::config_home().map_err(|source| DquestError::Cert {
        context: "resolving the configuration home".to_string(),
        source: Box::new(source),
    })?;
    let secret_data = preflight::secret_data_home()?;

    let profile = preflight::ensure_profile(&config_home)?;
    if !preflight::material_exists(&secret_data) {
        style::fail(&certificate_owed(&profile.live));
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
