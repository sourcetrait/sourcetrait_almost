use crate::*;

/// camp: the interactive terminal chat (a bridge consumer over the
/// era-two engine).
#[derive(Debug, clap::Parser)]
#[command(name = "camp", version, about)]
struct Cli {
    /// Profile root override: profile names resolve under
    /// <dir>/config and <dir>/settings instead of the XDG suite root
    #[arg(short = 'd', long = "dir")]
    dir: Option<String>,
    /// Config profile token: a pure snake resolves under the profile
    /// root; anything else is a component-toml path; absent = the
    /// `default` profile with the embedded-base fallback
    #[arg(short = 'c', long = "config")]
    config: Option<String>,
    /// Settings profile token: the same rules as --config
    #[arg(short = 's', long = "settings")]
    settings: Option<String>,
}

/// Binary entry: parse, run, exit non-zero on error. The TUI owns
/// stdout once the alternate screen opens; diagnostics before that go
/// to stderr.
pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    if let Err(error) = drive(cli) {
        eprintln!("camp: {error}");
        std::process::exit(1);
    }
}

/// Open the session (the engine thread spawns off-runtime), then run
/// the async chat loop on a current-thread runtime - camp's futures
/// are channel receives, so no driver features are needed.
fn drive(cli: Cli) -> CampResult<()> {
    let era = bridge_two::BridgeTwo;
    let info = era.info();
    eprintln!(
        "camp: era {} ({}) - the model loads on the session thread",
        info.era, info.model
    );
    let session = era.open_chat(&bridge::all::ChatOptions {
        dir: cli.dir,
        config: cli.config,
        settings: cli.settings,
        sample_len: None,
    })?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(chat::chat(session))
}
