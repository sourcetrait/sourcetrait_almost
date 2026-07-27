//! The startup preconditions, before the daemon is allowed to serve.
use crate::*;

/// The srcert profile this daemon mints its own authority under.
pub(crate) const PROFILE: &str = "quest";

/// The profile text shipped with the binary, placed on a first start.
pub(crate) const PROFILE_TEXT: &str = include_str!("../defaults/srcert/quest.toml");

/// The variable naming the secret data home; there is no flag for it.
pub(crate) const SECRET_DATA_ENV: &str = "XDGX_SECRET_DATA_HOME";

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
