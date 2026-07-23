//! The bquest command surface: the global -d/-c/-s profile flagset over
//! every subcommand (the suite's -d/-c/-s framework, lib-resolved).
use crate::*;

/// bquest: the training, development, and testing tool.
#[derive(Debug, clap::Parser)]
#[command(name = "bquest", version, about)]
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
    /// Self-documentation of the always-moving surface.
    Doc {
        #[command(subcommand)]
        command: DocCommand,
    },
    /// The training-mix pipeline (corpus trees -> documents ->
    /// packed chunks).
    Mix {
        #[command(subcommand)]
        command: MixCommand,
    },
    /// The Speculation:DepthProbe instrument (recorded greedy streams
    /// + the offline policy/cost-model simulator).
    Speculate {
        #[command(subcommand)]
        command: SpeculateCommand,
    },
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum MixCommand {
    /// Render corpus trees into dolma-field document tables (one
    /// file per document, verbatim text, identity in metadata).
    Render(MixRenderArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct MixRenderArgs {
    /// The render spec (.nuon table: name / corpus_dir / repo /
    /// license / kind per corpus tree).
    #[arg(long)]
    pub(crate) spec: PathBuf,
    /// Output directory (documents_<name>.nuon lands per spec row).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Verbatim added/created stamp carried into every document
    /// (deterministic; absent = empty).
    #[arg(long)]
    pub(crate) stamp: Option<String>,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum DocCommand {
    /// Print the whole command tree as an eye-tree listing - one
    /// `name # summary` line per category/topic/action, no flag or
    /// parameter detail (the grammar signature-block style).
    Cli,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum CapabilityCommand {
    /// Drive the engine over rendered olmo-eval fixture requests and
    /// emit predictions in their JSONL shape (scoring stays their
    /// code, run CPU-only in the eval env).
    Run(CapabilityRunArgs),
    /// Convert fixture requests and standing runs' predictions from
    /// their JSONL to whole-value .nuon mirrors (lossless,
    /// gate-verified in place; provenance siblings written).
    Convert(CapabilityConvertArgs),
    /// Score a converted run in pure rust (MC logprob accuracy, gsm8k
    /// exact-match, IFBench ifeval): per-item scores + per-task
    /// aggregates as .nuon.
    Score(CapabilityScoreArgs),
    /// Render a nuon run's predictions back to the reference JSONL
    /// tree (the reverse adapter for rescore.py comparisons).
    Bridge(CapabilityBridgeArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct CapabilityBridgeArgs {
    /// The nuon run directory (carrying predictions/).
    #[arg(long)]
    pub(crate) run: PathBuf,
    /// JSONL output root; absent = <run>/jsonl.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
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
pub(crate) struct CapabilityConvertArgs {
    /// Fixture render root (carrying requests/); absent = the
    /// capability home's fixtures/full.
    #[arg(long)]
    pub(crate) fixtures: Option<PathBuf>,
    /// Runs root whose child run directories carry predictions/;
    /// absent = the capability home's runs.
    #[arg(long)]
    pub(crate) runs: Option<PathBuf>,
    /// Nuon output root; absent = the capability home's nuon.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Comma-separated task-token filter (the sanitized spec, e.g.
    /// mmlu_anatomy); absent = every task found.
    #[arg(long)]
    pub(crate) tasks: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct CapabilityScoreArgs {
    /// Converted nuon run directory (carrying predictions/); absent =
    /// the capability home's nuon/runs/engine_default.
    #[arg(long)]
    pub(crate) run: Option<PathBuf>,
    /// Converted nuon fixtures root (carrying requests/); absent =
    /// the capability home's nuon/fixtures.
    #[arg(long)]
    pub(crate) fixtures: Option<PathBuf>,
    /// Output root (scores/ and aggregates.nuon land beneath it);
    /// absent = the run directory itself.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Checker-data directory (the copied reference data files);
    /// absent = the capability home's checker_data.
    #[arg(long)]
    pub(crate) checker_data: Option<PathBuf>,
    /// Comma-separated task-token filter; absent = every task found.
    #[arg(long)]
    pub(crate) tasks: Option<String>,
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
