//! QuestHarness: the plugin's own nu engine, for what the model asks of
//! it.
use crate::*;

/// The confinement binary; a Fedora box has it because Flatpak ships it.
pub(crate) const SANDBOX_BIN: &str = "bwrap";

/// The shell an ask runs in.
pub(crate) const SHELL_BIN: &str = "nu";

/// The procfs mount point, which nushell needs in order to start.
const PROC: &str = "/proc";

/// How long an ask may run before it is killed.
pub(crate) const DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// How often the wait loop looks at the child.
const POLL: std::time::Duration = std::time::Duration::from_millis(10);

/// What an ask may see and touch while it runs.
///
/// This is the fixture WORLD, and it is a plain parameter rather than a
/// test mode: the same harness points at a constructed world under a lock
/// and at the real machine in use, so what the locks exercise is what
/// ships.
#[derive(Debug, Clone)]
pub(crate) struct QuestWorld {
    pub(crate) sandbox: PathBuf,
    pub(crate) shell: PathBuf,
    /// Directories bound read-write; everything else is read-only.
    pub(crate) writable: Vec<PathBuf>,
    pub(crate) network: bool,
    pub(crate) timeout: std::time::Duration,
}

impl Default for QuestWorld {
    /// The confined default: no network and no writable path.
    fn default() -> Self {
        Self {
            sandbox: PathBuf::from(SANDBOX_BIN),
            shell: PathBuf::from(SHELL_BIN),
            writable: Vec::new(),
            network: false,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

impl QuestWorld {
    /// The confinement arguments this world asks for, before the `--`.
    ///
    /// A FRESH PROCFS IS ALWAYS MOUNTED, and that is forced rather than
    /// chosen. A read-only bind of the host's `/proc` is what `--ro-bind
    /// / /` carries along, and mounting a plain directory over it stops
    /// nushell from starting at all: it resolves its own executable
    /// through `/proc/self/exe` and panics before a single line runs.
    fn argv(&self) -> Vec<String> {
        let root = String::from("/");
        let mut argv = vec![
            String::from("--ro-bind"),
            root.clone(),
            root,
            String::from("--dev"),
            String::from("/dev"),
            String::from("--proc"),
            String::from(PROC),
            String::from("--die-with-parent"),
        ];
        if !self.network {
            argv.push(String::from("--unshare-net"));
        }
        for path in &self.writable {
            let target = path.display().to_string();
            argv.push(String::from("--bind"));
            argv.push(target.clone());
            argv.push(target);
        }
        argv
    }
}

/// The plugin's harness: it runs what the model asked the CALLER to run.
///
/// Questness never reaches this. An ask arrives on an `InferResponse`,
/// this runs it, and the value goes back on the next request.
#[derive(Debug, Clone, Default)]
pub(crate) struct QuestHarness {
    world: QuestWorld,
}

impl QuestHarness {
    /// The script one ask becomes.
    ///
    /// A `<nu>` body DECLARES a def and does not call one, so the call is
    /// appended. Both channels arrive as NUON literals, because an
    /// external process has no pipeline to be handed one and NUON is
    /// valid nu literal syntax by construction.
    fn script(
        &self,
        form: &bridge::InferNu,
        inputs: &[(bridge::InferPass, bridge::InferInput)],
    ) -> QuestPluginResult<String> {
        let mut call = String::new();
        if let Some(value) = bound(inputs, bridge::InferPass::In) {
            call.push_str(&value.to_nuon()?);
            call.push_str(" | ");
        }
        call.push_str(form.mode());
        if let Some(value) = bound(inputs, bridge::InferPass::Args) {
            call.push(' ');
            call.push_str(&value.to_nuon()?);
        }
        Ok(format!("{}\n({call}) | to nuon --raw\n", form.source()))
    }

    /// Run one ask, or say why there is no value.
    ///
    /// Both streams are drained by their own thread while the wait loop
    /// polls. Reading them after the wait would deadlock the moment an
    /// ask wrote more than a pipe buffer holds.
    pub(crate) fn run(
        &self,
        form: &bridge::InferNu,
        inputs: &[(bridge::InferPass, bridge::InferInput)],
    ) -> QuestPluginResult<harness_lib::nu::Value> {
        let script = self.script(form, inputs)?;
        let mut child = std::process::Command::new(&self.world.sandbox)
            .args(self.world.argv())
            .arg("--")
            .arg(&self.world.shell)
            .arg("-n")
            .arg("-c")
            .arg(&script)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        let out = drain(child.stdout.take());
        let err = drain(child.stderr.take());

        let deadline = std::time::Instant::now() + self.world.timeout;
        let mut timed_out = false;
        let code = loop {
            match child.try_wait()? {
                Some(status) => break status.code().unwrap_or(-1),
                None if std::time::Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break -1;
                }
                None => std::thread::sleep(POLL),
            }
        };
        let stdout = out.join().unwrap_or_default();
        let stderr = err.join().unwrap_or_default();

        if timed_out {
            snafu::whatever!(
                "`{}` was still running after {:?} and was stopped",
                form.mode(),
                self.world.timeout
            );
        }
        if code != 0 {
            snafu::whatever!(
                "`{}` did not run: {}",
                form.mode(),
                trimmed_reason(&stderr, code)
            );
        }
        match harness_lib::nu::from_nuon_text(stdout.trim()) {
            Ok(value) => Ok(value),
            Err(error) => snafu::whatever!(
                "`{}` did not answer in NUON: {error}",
                form.mode()
            ),
        }
    }
}

/// The value bound to a channel, if the ask carried one.
fn bound(
    inputs: &[(bridge::InferPass, bridge::InferInput)],
    pass: bridge::InferPass,
) -> Option<&bridge::InferValue> {
    inputs
        .iter()
        .find(|(bound, _)| *bound == pass)
        .map(|(_, input)| input.value())
}

/// Read one stream to its end on a thread of its own.
fn drain(
    stream: Option<impl std::io::Read + Send + 'static>,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let Some(mut stream) = stream else {
            return String::new();
        };
        let mut text = String::new();
        let _ = std::io::Read::read_to_string(&mut stream, &mut text);
        text
    })
}

/// What a non-zero exit says, or the code when it said nothing.
fn trimmed_reason(stderr: &str, code: i32) -> &str {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        return match code {
            127 => "the sandbox could not run the shell",
            _ => "it exited without saying why",
        };
    }
    trimmed
}
