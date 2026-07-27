//! The startup preconditions, before the daemon is allowed to serve.
use crate::*;

/// The srcert profile this daemon mints its own authority under.
pub(crate) const PROFILE: &str = "quest";

/// The profile text shipped with the binary, placed on a first start.
pub(crate) const PROFILE_TEXT: &str = include_str!("../defaults/srcert/quest.toml");

/// Whether the operator already had a profile, and where it lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Profile {
    /// One was already in place; only the reference copy was refreshed.
    Present { live: PathBuf },
    /// None was in place, so the shipped text was written.
    Written { live: PathBuf },
}

/// Place the shipped profile when the operator has none.
pub(crate) fn ensure_profile(config_home: &Path) -> DquestResult<Profile> {
    let installed = srcert::install_profile(config_home, PROFILE, PROFILE_TEXT).map_err(
        |source| DquestError::Cert {
            context: format!("installing the {PROFILE:?} srcert profile"),
            source: Box::new(source),
        },
    )?;
    let live = installed.live;
    if installed.written {
        return Ok(Profile::Written { live });
    }
    Ok(Profile::Present { live })
}
