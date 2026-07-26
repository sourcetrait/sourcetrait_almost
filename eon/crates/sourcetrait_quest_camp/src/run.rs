use crate::*;

/// camp: the interactive terminal chat, over an era through the bridge.
#[derive(Debug, clap::Parser)]
#[command(name = "camp", version, about)]
struct Cli {
    /// Profile root override for -c and -s name resolution
    #[arg(short = 'd', long = "dir")]
    dir: Option<String>,
    /// Config profile: a snake resolves under the root, else a path
    #[arg(short = 'c', long = "config")]
    config: Option<String>,
    /// Settings profile token: the same rules as --config
    #[arg(short = 's', long = "settings")]
    settings: Option<String>,
}

/// Binary entry: parse, run, exit non-zero on error.
pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    if let Err(error) = drive(cli) {
        eprintln!("camp: {error}");
        std::process::exit(1);
    }
}

/// Open the session, then run the chat loop on a current-thread runtime.
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
