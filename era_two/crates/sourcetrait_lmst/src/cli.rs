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
    /// The Speculation:DepthProbe instrument (recorded greedy streams
    /// + the offline policy/cost-model simulator).
    Speculate {
        #[command(subcommand)]
        command: SpeculateCommand,
    },
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum CapabilityCommand {
    /// Drive the engine over rendered olmo-eval fixture requests and
    /// emit predictions in their JSONL shape (scoring stays their
    /// code, run CPU-only in the eval env).
    Run(CapabilityRunArgs),
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum SpeculateCommand {
    /// Record plain greedy transcripts (token streams + text) from a
    /// .nuon fixture plan into per-transcript .nuon artifacts.
    Record(SpeculateRecordArgs),
    /// Replay recorded streams through the lookup index under
    /// candidate policies x pass-cost models; emit the depth report.
    Simulate(SpeculateSimulateArgs),
    /// Print a transcript's emitted tokens as indexed decoded pieces
    /// (the phase-annotation aid).
    Tokens(SpeculateTokensArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct SpeculateRecordArgs {
    /// The transcript fixture plan (.nuon table:
    /// name/kind/turns/budget).
    #[arg(long)]
    pub(crate) fixtures: PathBuf,
    /// Artifact directory (spec_transcript_<name>.nuon lands here).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Comma-separated transcript-name filter; absent = every plan
    /// row.
    #[arg(long)]
    pub(crate) only: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct SpeculateSimulateArgs {
    /// The recorded-transcript directory (spec_transcript_*.nuon +
    /// optional sibling spec_phases_*.nuon annotations).
    #[arg(long)]
    pub(crate) transcripts: PathBuf,
    /// The report artifact path (.nuon).
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(crate) struct SpeculateTokensArgs {
    /// The recorded-transcript directory.
    #[arg(long)]
    pub(crate) transcripts: PathBuf,
    /// The transcript name (spec_transcript_<name>.nuon).
    #[arg(long)]
    pub(crate) name: String,
    /// Restrict to one turn index; absent = every turn.
    #[arg(long)]
    pub(crate) turn: Option<usize>,
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
