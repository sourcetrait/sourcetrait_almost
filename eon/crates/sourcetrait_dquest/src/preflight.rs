//! The startup preconditions, before the daemon is allowed to serve.
use crate::*;

/// The profile and the variable that locates its material, both from the
/// transport rather than restated here.
///
/// A daemon and a client disagreeing about which profile they present is
/// a fault that should not be expressible, so the transport owns them and
/// both ends read the same two names.
pub(crate) use bridge::{
    PROFILE,
    SECRET_DATA_ENV,
};

/// The profile text shipped with the binary, placed on a first start.
pub(crate) const PROFILE_TEXT: &str = include_str!("../defaults/srcert/quest.toml");

/// The variable naming the cache home session logs are written under.
pub(crate) const CACHE_HOME_ENV: &str = "XDG_CACHE_HOME";

/// Where this daemon's session logs go, or None to write none.
///
/// The cache home is a path the PLATFORM defines, so it is looked up
/// rather than invented; the leaf below it is ours and is created on
/// demand. An unset variable turns logging OFF rather than refusing a
/// start, which is the same rule that makes a logging failure never fail
/// a turn: the log records the work rather than being part of it.
pub(crate) fn session_log_root() -> Option<PathBuf> {
    let home = env::var(CACHE_HOME_ENV).ok()?;
    if home.trim().is_empty() {
        return None;
    }
    Some(
        PathBuf::from(home)
            .join("sourcetrait")
            .join("dquest")
            .join("sessions"),
    )
}

/// Place the shipped profile when the operator has none.
pub(crate) fn ensure_profile(config_home: &Path) -> DquestResult<srcert::ProfileInstall> {
    srcert::install_profile(config_home, PROFILE, PROFILE_TEXT).map_err(|source| {
        DquestError::Cert {
            context: format!("installing the {PROFILE:?} srcert profile"),
            source: Box::new(source),
        }
    })
}

/// The secret data home this daemon's key material lives under.
pub(crate) fn secret_data_home() -> DquestResult<PathBuf> {
    named_home(env::var(SECRET_DATA_ENV).ok().as_deref())
}

/// The home a raw variable value names; absent and blank both refuse.
pub(crate) fn named_home(value: Option<&str>) -> DquestResult<PathBuf> {
    match value {
        Some(value) if !value.trim().is_empty() => Ok(PathBuf::from(value)),
        _ => Err(DquestError::Unset {
            variable: SECRET_DATA_ENV,
        }),
    }
}

/// Whether the material this daemon would serve TLS with is installed.
pub(crate) fn material_exists(secret_data: &Path) -> bool {
    srcert::check_exists(secret_data, PROFILE)
}
