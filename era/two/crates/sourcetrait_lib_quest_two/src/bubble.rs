//! A client harness that answers by running the mode under bubblewrap.
use crate::*;

/// The confinement binary; a Fedora box has it because Flatpak ships it.
pub const SANDBOX_BIN: &str = "bwrap";

/// The shell a mode runs in.
pub const SHELL_BIN: &str = "nu";

/// The procfs mount point, which nushell needs in order to start.
const PROC: &str = "/proc";

/// How long a mode may run before it is killed.
pub const DEFAULT_TIMEOUT: time::Duration = time::Duration::from_secs(10);

/// How often the wait loop looks at the child.
const POLL: time::Duration = time::Duration::from_millis(10);

/// The kinds a refusal comes back under.
const RAN_BADLY: &str = "harness::bubble";
const TIMED_OUT: &str = "harness::timeout";
const BAD_NUON: &str = "harness::nuon";
const BAD_SHAPE: &str = "harness::conformance";

/// What a request may see and touch while it runs.
///
/// This is the fixture WORLD, and it is a plain parameter rather than a
/// test mode: the same harness points at a constructed world under a lock
/// and at the real machine in use, so what the locks exercise is what
/// ships.
#[derive(Debug, Clone)]
pub struct BubbleWorld {
    pub sandbox: PathBuf,
    pub shell: PathBuf,
    /// A directory whose files are bound OVER the matching `/proc`
    /// entries, one by one, on top of a freshly mounted procfs.
    pub proc: Option<PathBuf>,
    /// Directories bound read-write; everything else is read-only.
    pub writable: Vec<PathBuf>,
    pub network: bool,
    pub timeout: time::Duration,
}

impl Default for BubbleWorld {
    /// The confined default: no network, no writable path, real `/proc`.
    fn default() -> Self {
        Self {
            sandbox: PathBuf::from(SANDBOX_BIN),
            shell: PathBuf::from(SHELL_BIN),
            proc: None,
            writable: Vec::new(),
            network: false,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

impl BubbleWorld {
    /// The confinement arguments this world asks for, before the `--`.
    ///
    /// A FRESH PROCFS IS ALWAYS MOUNTED, and that is forced rather than
    /// chosen. A read-only bind of the host's `/proc` is what `--ro-bind
    /// / /` carries along, and mounting a plain directory over it - the
    /// shape the verifier profile beside this one uses - stops nushell
    /// from starting at all: it resolves its own executable through
    /// `/proc/self/exe` and panics with "no /proc/self/exe available"
    /// before a single line of the mode runs.
    ///
    /// So a mocked world is bound FILE BY FILE over that fresh procfs.
    /// The trade is real and worth stating: an unmocked path reads the
    /// sandbox's own procfs rather than failing loudly. That is still not
    /// a leak of host state - the procfs is the sandbox's - but it does
    /// mean a mode can read something nobody wrote for it.
    pub(crate) fn argv(&self) -> Vec<String> {
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
        for (source, target) in self.proc_binds() {
            argv.push(String::from("--ro-bind"));
            argv.push(source);
            argv.push(target);
        }
        for path in &self.writable {
            let target = path.display().to_string();
            argv.push(String::from("--bind"));
            argv.push(target.clone());
            argv.push(target);
        }
        argv
    }

    /// Each mock file paired with the `/proc` entry it stands in for.
    ///
    /// Only what the mock directory actually holds, and only files: a
    /// bind needs its target to exist, and `/proc` is not writable even
    /// inside the sandbox, so an entry the real procfs lacks cannot be
    /// invented here.
    fn proc_binds(&self) -> Vec<(String, String)> {
        let Some(root) = &self.proc else {
            return Vec::new();
        };
        let Ok(entries) = fs::read_dir(root) else {
            return Vec::new();
        };
        let mut binds: Vec<(String, String)> = entries
            .flatten()
            .filter(|entry| entry.path().is_file())
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                Some((entry.path().display().to_string(), format!("{PROC}/{name}")))
            })
            .collect();
        binds.sort();
        binds
    }
}

/// A client harness whose world is a parameter.
#[derive(Debug, Clone, Default)]
pub struct BubbleHarness {
    world: BubbleWorld,
}

impl BubbleHarness {
    pub fn new(world: BubbleWorld) -> Self {
        Self { world }
    }

    pub fn world(&self) -> &BubbleWorld {
        &self.world
    }

    /// Change the world between turns, which is what makes a run a
    /// SCRIPT rather than a fixture value.
    pub fn set_world(&mut self, world: BubbleWorld) {
        self.world = world;
    }

    /// The script one request becomes.
    ///
    /// A `<nu>` body DECLARES a def and does not call one, so the call is
    /// appended. Both channels arrive as NUON literals here, where the
    /// inside path pipes `$in` as a real value - an external process has
    /// no pipeline to hand it, and NUON is valid nu literal syntax by
    /// construction, which is the same property the grammar rests on.
    fn script(&self, request: &harness::HarnessRequest) -> LibQuestResult<String> {
        let mut call = String::new();
        if let Some(input) = request.binding("$in") {
            call.push_str(&input.to_nuon()?);
            call.push_str(" | ");
        }
        call.push_str(&request.mode);
        if let Some(args) = request.binding("$args") {
            call.push(' ');
            call.push_str(&args.to_nuon()?);
        }
        Ok(format!("{}\n({call}) | to nuon --raw\n", request.source))
    }

    /// Run one script to an ending, or kill it at the deadline.
    ///
    /// Both streams are drained by their own thread while the wait loop
    /// polls. Reading them after the wait instead would deadlock the
    /// moment a mode wrote more than a pipe buffer holds.
    fn run(&self, script: &str) -> LibQuestResult<Ran> {
        let mut child = process::Command::new(&self.world.sandbox)
            .args(self.world.argv())
            .arg("--")
            .arg(&self.world.shell)
            .arg("-n")
            .arg("-c")
            .arg(script)
            .stdin(process::Stdio::null())
            .stdout(process::Stdio::piped())
            .stderr(process::Stdio::piped())
            .spawn()?;

        let out = drain(child.stdout.take());
        let err = drain(child.stderr.take());

        let deadline = time::Instant::now() + self.world.timeout;
        let mut timed_out = false;
        let code = loop {
            match child.try_wait()? {
                Some(status) => break status.code().unwrap_or(-1),
                None if time::Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break -1;
                }
                None => thread::sleep(POLL),
            }
        };

        Ok(Ran {
            code,
            timed_out,
            stdout: out.join().unwrap_or_default(),
            stderr: err.join().unwrap_or_default(),
        })
    }
}

/// One run's ending.
struct Ran {
    code: i32,
    timed_out: bool,
    stdout: String,
    stderr: String,
}

impl harness::ClientHarness for BubbleHarness {
    /// Run the mode and answer with what it produced.
    ///
    /// A mode that fails is a RESPONSE rather than an error: Questness
    /// turns a refusal into a repair envelope the model can read, so the
    /// only `Err` here is the sandbox failing to start at all.
    fn serve(
        &mut self,
        request: &harness::HarnessRequest,
    ) -> LibQuestResult<harness::HarnessResponse> {
        let script = self.script(request)?;
        let ran = self.run(&script)?;

        if ran.timed_out {
            return Ok(harness::HarnessResponse::failed(
                TIMED_OUT,
                &format!(
                    "`{}` was still running after {:?} and was stopped",
                    request.mode, self.world.timeout
                ),
            ));
        }
        if ran.code != 0 {
            return Ok(harness::HarnessResponse::failed(
                RAN_BADLY,
                trimmed_reason(&ran.stderr, ran.code),
            ));
        }

        let value = match nu::from_nuon_text(ran.stdout.trim()) {
            Ok(value) => value,
            Err(error) => {
                return Ok(harness::HarnessResponse::failed(
                    BAD_NUON,
                    &format!("`{}` did not answer in NUON: {error}", request.mode),
                ));
            }
        };

        // The request carries the declared output type so the answer can
        // be checked, and this is where that is spent. A mode whose body
        // disagrees with its own signature is caught here rather than
        // reaching the model as a well-formed wrong answer.
        if !request.output.is_empty() {
            match nu::parse_typedef(&request.output) {
                Ok(declared) => {
                    if let Err(error) = nu::conform(&value, &declared) {
                        return Ok(harness::HarnessResponse::failed(
                            BAD_SHAPE,
                            &format!("`{}` answered off its own signature: {error}", request.mode),
                        ));
                    }
                }
                Err(error) => {
                    return Ok(harness::HarnessResponse::failed(
                        BAD_SHAPE,
                        &format!("`{}` declares an unreadable output type: {error}", request.mode),
                    ));
                }
            }
        }

        Ok(harness::HarnessResponse::value(value))
    }
}

/// Read one stream to its end on a thread of its own.
fn drain(stream: Option<impl io::Read + Send + 'static>) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let Some(mut stream) = stream else {
            return String::new();
        };
        let mut text = String::new();
        let _ = stream.read_to_string(&mut text);
        text
    })
}

/// What a non-zero exit says, or the code when it said nothing.
fn trimmed_reason(stderr: &str, code: i32) -> &str {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        return match code {
            127 => "the sandbox could not run the shell",
            _ => "the mode exited without saying why",
        };
    }
    trimmed
}
