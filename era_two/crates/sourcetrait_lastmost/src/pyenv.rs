//! The pinned python environment home and driver-script materialization.
use crate::*;

pub(crate) const LASTMOST_DATA_SUBDIR: &str = "sourcetrait/almost/lastmost";

const GENERATE_DRIVER: &str = include_str!("../pysrc/generate.py");

/// The lastmost data home (the venv and materialized pysrc live here).
pub(crate) fn lastmost_home() -> LastmostResult<PathBuf> {
    Ok(checkpoint::data_home()?.join(LASTMOST_DATA_SUBDIR))
}

/// The pinned environment's python; errors with the provisioning hint.
pub(crate) fn env_python() -> LastmostResult<PathBuf> {
    let python = lastmost_home()?.join("env/bin/python");
    if !python.is_file() {
        snafu::whatever!(
            "pinned python env missing at {}; provision it: cd <crate>/pyenv && \
             UV_PROJECT_ENVIRONMENT=<data-home>/{}/env uv sync --frozen",
            python.display(),
            LASTMOST_DATA_SUBDIR
        );
    }
    Ok(python)
}

/// Write the embedded driver scripts into the data home (only when their
/// content changed); returns the pysrc dir.
pub(crate) fn materialize_pysrc() -> LastmostResult<PathBuf> {
    let dir = lastmost_home()?.join("pysrc");
    fs::create_dir_all(&dir)?;
    let path = dir.join("generate.py");
    let current = fs::read_to_string(&path).unwrap_or_default();
    if current != GENERATE_DRIVER {
        fs::write(&path, GENERATE_DRIVER)?;
    }
    Ok(dir)
}
