//! The lmst command surface: the global -d/-c/-s profile flagset over
//! every subcommand (the suite's -d/-c/-s framework, lib-resolved).
use crate::*;

/// lmst: the training, development, and testing tool.
#[derive(Debug, clap::Parser)]
#[command(name = "lmst", version, about)]
pub(crate) struct Cli {
    /// Profile root override: profile names resolve under
    /// <dir>/config and <dir>/settings instead of the XDG suite root.
    #[arg(short = 'd', long = "dir", global = true)]
    pub(crate) dir: Option<PathBuf>,
    /// Config profile token: a pure snake resolves under the profile
    /// root; anything else is a component-toml path; absent = the
    /// `default` profile with the embedded-base fallback.
    #[arg(short = 'c', long = "config", global = true)]
    pub(crate) config: Option<String>,
    /// Settings profile token: the same rules as --config.
    #[arg(short = 's', long = "settings", global = true)]
    pub(crate) settings: Option<String>,
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum Command {
    /// The CapabilityRatchet instrument (fixture-driven eval legs).
    Capability {
        #[command(subcommand)]
        command: CapabilityCommand,
    },
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum CapabilityCommand {
    /// Drive the engine over rendered olmo-eval fixture requests and
    /// emit predictions in their JSONL shape (scoring stays their
    /// code, run CPU-only in the eval env).
    Run(CapabilityRunArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct CapabilityRunArgs {
    /// Fixture root (a render's -O dir carrying requests/); absent =
    /// the capability home's fixtures/full.
    #[arg(long)]
    pub(crate) fixtures: Option<PathBuf>,
    /// Output root (predictions/ lands beneath it); absent = the
    /// capability home's runs/engine_<settings token or default>.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Comma-separated task_name filter; absent = every task found.
    #[arg(long)]
    pub(crate) tasks: Option<String>,
}
