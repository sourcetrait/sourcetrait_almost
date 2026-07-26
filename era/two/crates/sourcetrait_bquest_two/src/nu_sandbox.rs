//! Running nushell that we did not write, under bubblewrap.
use crate::*;

use std::process::{
    Command,
    Stdio,
};

/// The sandbox binary; absent is a hard error, never a fallback.
const SANDBOX_BIN: &str = "bwrap";
/// The nushell binary, resolved on PATH to the estate's pinned one.
const NU_BIN: &str = "nu";

/// Filesystem roots the sandbox binds; required, never created.
const ROOT: &str = "/";
const DEV: &str = "/dev";
const PROC: &str = "/proc";

/// How long a generated pipeline may run before it is killed.
const DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// Poll interval while waiting on the child.
const POLL: std::time::Duration = std::time::Duration::from_millis(20);

/// What one sandboxed run produced.
#[derive(Debug, Clone)]
pub(crate) struct NuOutcome {
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) ok: bool,
    pub(crate) timed_out: bool,
}

/// Confirm the sandbox is available, once per verification pass.
pub(crate) fn require_sandbox() -> BquestResult<()> {
    let probe = Command::new(SANDBOX_BIN)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    match probe {
        Ok(status) if status.success() => Ok(()),
        _ => snafu::whatever!(
            "{SANDBOX_BIN} is not available, and generated nushell is never run \
             outside it"
        ),
    }
}

/// Run nushell source in the sandbox and return what it produced.
pub(crate) fn run_nu(source: &str, timeout: std::time::Duration) -> BquestResult<NuOutcome> {
    let mut child = Command::new(SANDBOX_BIN)
        .args([
            "--ro-bind", ROOT, ROOT,
            "--dev", DEV,
            "--proc", PROC,
            "--unshare-net",
            "--die-with-parent",
            "--new-session",
            NU_BIN,
            "--no-config-file",
            "-c",
        ])
        .arg(source)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let started = std::time::Instant::now();
    let mut timed_out = false;
    loop {
        match child.try_wait()? {
            Some(_) => break,
            None => {
                if started.elapsed() >= timeout {
                    let _ = child.kill();
                    timed_out = true;
                    break;
                }
                std::thread::sleep(POLL);
            }
        }
    }
    let output = child.wait_with_output()?;
    Ok(NuOutcome {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        ok: !timed_out && output.status.success(),
        timed_out,
    })
}

/// The VALUE a pipeline produced, or None if it produced no NUON.
pub(crate) fn pipeline_value(source: &str) -> BquestResult<Option<lib::nu::Value>> {
    let wrapped = format!("({source}) | to nuon");
    let outcome = run_nu(&wrapped, DEFAULT_TIMEOUT)?;
    if !outcome.ok {
        return Ok(None);
    }
    Ok(lib::nu::from_nuon_text(outcome.stdout.trim()).ok())
}
