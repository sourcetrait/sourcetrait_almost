//! Running nushell that we did not write.
//!
//! Three of the task families are verified by EXECUTION - the answer
//! is a pipeline, and whether it is right is whether it produces the
//! right value. That means running model output, so it runs under
//! bubblewrap with the root read-only, no network, no session, and
//! dying with its parent.
//!
//! ## DEV
//! In-process evaluation was never an option here even though the
//! library already carries the nushell crates. A verifier runs
//! generated code, and generated code is arbitrary: an in-process
//! eval would hand it the trainer's own address space, filesystem
//! and network. Bubblewrap is unprivileged, already present because
//! Flatpak ships it, and alters nothing.
//!
//! The comparison is on VALUES, never on stdout text. Two correct
//! pipelines can render the same value differently - a table prints
//! as a bordered grid and as a NUON literal - so the runner appends
//! `to nuon` and the caller compares parsed values. Comparing the
//! rendered text would score formatting and call it correctness.
//!
//! A generated pipeline can loop forever, so every run carries a
//! timeout and the child is killed on it. Without one a single
//! non-terminating answer stalls a whole verification pass.
use crate::*;

use std::process::{
    Command,
    Stdio,
};

/// The sandbox binary. Absent is a hard error rather than a silent
/// fallback to running unsandboxed - the whole point is that model
/// output never runs unconfined.
const SANDBOX_BIN: &str = "bwrap";
/// The nushell binary, resolved on PATH so the estate's pinned
/// version is what runs.
const NU_BIN: &str = "nu";

/// Filesystem roots the sandbox binds. These are platform paths
/// rather than ours, so they are REQUIRED to exist rather than
/// created.
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

/// Confirm the sandbox is available. Called before a verification
/// pass so a missing sandbox fails once and loudly rather than once
/// per item.
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

/// Run a pipeline and return the VALUE it produced, by appending
/// `to nuon` and parsing what comes back. None when the pipeline
/// failed, timed out, or emitted something that is not NUON.
pub(crate) fn pipeline_value(source: &str) -> BquestResult<Option<lib::nu::Value>> {
    // A trailing semicolon or comment would swallow the appended
    // stage, so the pipeline is parenthesised rather than concatenated.
    let wrapped = format!("({source}) | to nuon");
    let outcome = run_nu(&wrapped, DEFAULT_TIMEOUT)?;
    if !outcome.ok {
        return Ok(None);
    }
    Ok(lib::nu::from_nuon_text(outcome.stdout.trim()).ok())
}
