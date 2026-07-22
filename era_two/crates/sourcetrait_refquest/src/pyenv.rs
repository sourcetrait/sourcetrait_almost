//! The pinned python environment home and driver-script materialization.
use crate::*;

pub(crate) const REFQUEST_DATA_SUBDIR: &str = "sourcetrait/quest/refquest";

const PYSRC_FILES: [(&str, &str); 7] = [
    ("common.py", include_str!("../pysrc/common.py")),
    ("generate.py", include_str!("../pysrc/generate.py")),
    ("dump.py", include_str!("../pysrc/dump.py")),
    ("needle.py", include_str!("../pysrc/needle.py")),
    ("needle_vllm.py", include_str!("../pysrc/needle_vllm.py")),
    ("bench_vllm.py", include_str!("../pysrc/bench_vllm.py")),
    ("envcheck.py", include_str!("../pysrc/envcheck.py")),
];

/// The refquest data home (the venv and materialized pysrc live here).
pub(crate) fn refquest_home() -> RefquestResult<PathBuf> {
    Ok(checkpoint::data_home()?.join(REFQUEST_DATA_SUBDIR))
}

/// The pinned environment's python; errors with the provisioning hint.
pub(crate) fn env_python() -> RefquestResult<PathBuf> {
    let python = refquest_home()?.join("env/bin/python");
    if !python.is_file() {
        snafu::whatever!(
            "pinned python env missing at {}; provision it: cd <crate>/pyenv && \
             UV_PROJECT_ENVIRONMENT=<data-home>/{}/env uv sync --frozen",
            python.display(),
            REFQUEST_DATA_SUBDIR
        );
    }
    Ok(python)
}

/// Run `refquest env`: the pinned-environment drift check.
pub(crate) fn envcheck(args: &EnvArgs) -> RefquestResult<()> {
    let python = env_python()?;
    let driver = materialize_pysrc()?.join("envcheck.py");
    let mut cmd = process::Command::new(&python);
    cmd.arg(&driver);
    run_driver(cmd, args.record.as_deref())
}

/// Write the embedded driver scripts into the data home (only when their
/// content changed); returns the pysrc dir.
pub(crate) fn materialize_pysrc() -> RefquestResult<PathBuf> {
    let dir = refquest_home()?.join("pysrc");
    fs::create_dir_all(&dir)?;
    for (name, content) in PYSRC_FILES {
        let path = dir.join(name);
        let current = fs::read_to_string(&path).unwrap_or_default();
        if current != content {
            fs::write(&path, content)?;
        }
    }
    Ok(dir)
}
