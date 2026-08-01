//! The command surface: a global profile flagset over a verb tree.
use crate::*;

/// bquest: the training, development, and testing tool.
#[derive(Debug, clap::Parser)]
#[command(name = "bquest", version, about)]
pub(crate) struct Cli {
    /// Profile root override, in place of the XDG suite root.
    #[arg(short = 'd', long = "dir", global = true)]
    pub(crate) dir: Option<PathBuf>,
    /// Config profile token: a snake, or a component-toml path.
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
    /// Our own bench: does the model do the job we built it for.
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
    /// Task generators: synthesised, mechanically-verifiable examples.
    Taskgen {
        #[command(subcommand)]
        command: TaskgenCommand,
    },
    /// Training assets as data, laid out by stage and syllabus.
    Syllabus {
        #[command(subcommand)]
        command: SyllabusCommand,
    },
    /// Sampled, verifier-graded generation - the reinforcement
    /// stage's data source.
    Rollout {
        #[command(subcommand)]
        command: RolloutCommand,
    },
    /// The speculation depth probe: record, then replay offline.
    Speculate {
        #[command(subcommand)]
        command: SpeculateCommand,
    },
    /// The training stages, in chain order from the gate onward.
    Train {
        #[command(subcommand)]
        command: TrainCommand,
    },
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum TrainCommand {
    /// The three toy-config gradient locks, on cpu f32.
    Gate,
    /// Continued pretraining over a packed-chunks artifact.
    Cpt(TrainCptArgs),
    /// Supervised tuning: the loss covers the assistant turns alone.
    Sft(TrainSftArgs),
    /// Preference tuning over pairs, against the frozen base.
    Dpo(TrainDpoArgs),
    /// Reinforcement tuning over verifier-scored rollout groups.
    Rlvr(TrainRlvrArgs),
}

/// The knobs every stage loop shares.
#[derive(Debug, clap::Args)]
pub(crate) struct StageArgs {
    /// The adapter artifact path (.safetensors).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Continue a previous stage's adapter; without it, the base.
    #[arg(long)]
    pub(crate) resume: Option<PathBuf>,
    /// LoRA rank (fresh adapters only).
    #[arg(long, default_value_t = 64)]
    pub(crate) rank: usize,
    /// LoRA alpha; absent = 2 * rank (fresh adapters only).
    #[arg(long)]
    pub(crate) alpha: Option<f64>,
    /// Peak learning rate (linear warmup then constant).
    #[arg(long, default_value_t = 1e-5)]
    pub(crate) learning_rate: f64,
    /// Linear warmup steps.
    #[arg(long, default_value_t = 10)]
    pub(crate) warmup_steps: usize,
    /// Optimizer steps; absent = one pass over the input.
    #[arg(long)]
    pub(crate) steps: Option<usize>,
    /// Cross-entropy head-chunk rows (the logits never materialize
    /// whole).
    #[arg(long, default_value_t = 128)]
    pub(crate) loss_chunk: usize,
    /// Adapter-init seed.
    #[arg(long, default_value_t = 299_792_458)]
    pub(crate) seed: u64,
    /// The NUON-lines step log; absent = <out stem>_steps.nuon.
    #[arg(long)]
    pub(crate) log: Option<PathBuf>,
    /// stderr progress cadence.
    #[arg(long, default_value_t = 10)]
    pub(crate) log_every: usize,
}

/// The supervised loss normalization `train sft` applies per example.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum SftLoss {
    /// The standing per-example mean over the row's own count.
    Example,
    /// Token-uniform: the sum over the pack-mean supervised count.
    Token,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TrainSftArgs {
    /// A packed instruction artifact, from `mix instruct`.
    #[arg(long)]
    pub(crate) chunks: PathBuf,
    /// Rows folded into one optimizer step.
    #[arg(long, default_value_t = 1)]
    pub(crate) accumulate: usize,
    /// Loss normalization: example mean, or token-uniform.
    #[arg(long, value_enum, default_value_t = SftLoss::Example)]
    pub(crate) loss: SftLoss,
    #[command(flatten)]
    pub(crate) stage: StageArgs,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TrainDpoArgs {
    /// A preference-pair table (.nuon: prompt / chosen / rejected).
    #[arg(long)]
    pub(crate) pairs: PathBuf,
    /// The window each candidate pads to; a pair exceeding it drops.
    #[arg(long, default_value_t = 1024)]
    pub(crate) seq_len: usize,
    /// The preference sharpness.
    #[arg(long, default_value_t = 0.1)]
    pub(crate) beta: f64,
    #[command(flatten)]
    pub(crate) stage: StageArgs,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TrainRlvrArgs {
    /// A scored-rollout artifact (from `rollout run`).
    #[arg(long)]
    pub(crate) rollouts: PathBuf,
    /// The window each rollout pads to.
    #[arg(long, default_value_t = 1024)]
    pub(crate) seq_len: usize,
    #[command(flatten)]
    pub(crate) stage: StageArgs,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum BenchCommand {
    /// Answer every bench prompt greedily and report the pass rate.
    Run(BenchRunArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct BenchRunArgs {
    /// A verifiable-prompt table (a taskgen bench split).
    #[arg(long)]
    pub(crate) prompts: PathBuf,
    /// The per-item report (.nuon).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Decode budget per answer.
    #[arg(long, default_value_t = 256)]
    pub(crate) max_tokens: usize,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum TaskgenCommand {
    /// Every family at once, each answer machine-checked before it ships.
    All(TaskgenAllArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct TaskgenAllArgs {
    /// Output directory; the per-stage tables and the bench land in
    /// it.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Examples to generate before the bench split.
    #[arg(long, default_value_t = 512)]
    pub(crate) count: usize,
    /// Every nth example is held out as bench rather than trained on.
    #[arg(long, default_value_t = 10)]
    pub(crate) bench_every: usize,
    /// The generation seed - the whole run is deterministic in it.
    #[arg(long, default_value_t = 299_792_458)]
    pub(crate) seed: u64,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum SyllabusCommand {
    /// Render every method into a run's railroad, as one committed REV.
    Emit(SyllabusEmitArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct SyllabusEmitArgs {
    /// The syllabus tree's root; its stage directories are walked.
    #[arg(long)]
    pub(crate) root: PathBuf,
    /// A railroad to commit into; absent = lay a fresh one.
    #[arg(long)]
    pub(crate) railroad: Option<PathBuf>,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum RolloutCommand {
    /// Sample replies, grade each, and emit the scored groups.
    Run(RolloutRunArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct RolloutRunArgs {
    /// A verifiable-prompt table (.nuon: prompt / verifier /
    /// reference).
    #[arg(long)]
    pub(crate) prompts: PathBuf,
    /// The scored-rollout artifact (.nuon).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Replies per prompt: the group an advantage is computed over.
    #[arg(long, default_value_t = 8)]
    pub(crate) group: usize,
    /// Decode budget per reply.
    #[arg(long, default_value_t = 256)]
    pub(crate) max_tokens: usize,
    /// Sampling temperature; the group needs spread, so never zero.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) temperature: f64,
    /// The base sampling seed (each reply offsets from it).
    #[arg(long, default_value_t = 299_792_458)]
    pub(crate) seed: u64,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TrainCptArgs {
    /// The packed-chunks artifact (a mix pack .safetensors).
    #[arg(long)]
    pub(crate) chunks: PathBuf,
    /// The adapter artifact path (.safetensors).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// LoRA rank; the production value is fixed library-wide.
    #[arg(long, default_value_t = 64)]
    pub(crate) rank: usize,
    /// LoRA alpha; absent = 2 * rank, the era-one anchor.
    #[arg(long)]
    pub(crate) alpha: Option<f64>,
    /// Peak learning rate (linear warmup then constant).
    #[arg(long, default_value_t = 2e-4)]
    pub(crate) learning_rate: f64,
    /// Linear warmup steps.
    #[arg(long, default_value_t = 10)]
    pub(crate) warmup_steps: usize,
    /// Optimizer steps; absent = one pass over the chunks.
    #[arg(long)]
    pub(crate) steps: Option<usize>,
    /// Cross-entropy head-chunk rows (the logits never materialize
    /// whole).
    #[arg(long, default_value_t = 128)]
    pub(crate) loss_chunk: usize,
    /// Adapter-init seed (deterministic Box-Muller draws).
    #[arg(long, default_value_t = 299_792_458)]
    pub(crate) seed: u64,
    /// The NUON-lines step log; absent = <out stem>_steps.nuon
    /// beside the adapter.
    #[arg(long)]
    pub(crate) log: Option<PathBuf>,
    /// stderr progress cadence (the log file gets every step).
    #[arg(long, default_value_t = 10)]
    pub(crate) log_every: usize,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum MixCommand {
    /// Pack document tables into shuffled training chunks.
    Pack(MixPackArgs),
    /// Render corpus trees into dolma-field document tables.
    Render(MixRenderArgs),
    /// Sample documents across tables to a text-byte budget.
    Sample(MixSampleArgs),
    /// Pack instruction examples one per row, assistant turns masked.
    Instruct(MixInstructArgs),
    /// Stream one zstd dolma shard into a document table.
    Rip(MixRipArgs),
    /// Render a pack's rows as decoded tokens, one line per position.
    Tokens(MixTokensArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct MixTokensArgs {
    /// A packed artifact (.safetensors) from `mix instruct` or `mix pack`.
    #[arg(long)]
    pub(crate) chunks: PathBuf,
    /// The NUON-lines table: row, position, id, piece, supervised.
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(crate) struct MixRipArgs {
    /// The zstd-compressed dolma JSONL shard.
    #[arg(long)]
    pub(crate) shard: PathBuf,
    /// The document table (.nuon; a provenance sidecar lands beside
    /// it).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// The source stream this shard belongs to.
    #[arg(long)]
    pub(crate) name: String,
    /// The hub dataset the shard came from (rides metadata.repo).
    #[arg(long)]
    pub(crate) hub: String,
    /// The shard's path inside the dataset; absent = its file name.
    #[arg(long = "shard-path")]
    pub(crate) shard_path: Option<String>,
    /// `docs` or `code`; nothing on this side should be code.
    #[arg(long, default_value = "docs")]
    pub(crate) kind: String,
    /// The upstream license (rides metadata.license).
    #[arg(long, default_value = "odc-by")]
    pub(crate) license: String,
    /// Stop once text bytes cross this; the last document overshoots.
    #[arg(long)]
    pub(crate) budget_bytes: usize,
    /// Verbatim added/created stamp; absent = empty.
    #[arg(long)]
    pub(crate) stamp: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct MixInstructArgs {
    /// A supervised example table (.nuon: messages per row).
    #[arg(long)]
    pub(crate) examples: PathBuf,
    /// The packed artifact (.safetensors; a .nuon provenance sidecar
    /// lands beside it).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// The window each example pads to; a longer one is dropped.
    #[arg(long, default_value_t = 1024)]
    pub(crate) seq_len: usize,
}

#[derive(Debug, clap::Args)]
pub(crate) struct MixSampleArgs {
    /// documents_<name>.nuon tables; rows sample across the union.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) documents: Vec<PathBuf>,
    /// The sampled documents table (.nuon; a provenance sidecar
    /// lands beside it).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Stop once text bytes cross this; the last document overshoots.
    #[arg(long)]
    pub(crate) budget_bytes: usize,
    /// The deterministic sample seed.
    #[arg(long, default_value_t = 299_792_458)]
    pub(crate) seed: u64,
}

#[derive(Debug, clap::Args)]
pub(crate) struct MixPackArgs {
    /// documents_<name>.nuon tables, consumed in the given order
    /// (the mix order).
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) documents: Vec<PathBuf>,
    /// The packed-chunks artifact; a provenance sidecar lands beside it.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Training window; chunks carry seq_len + 1 ids.
    #[arg(long, default_value_t = 1024)]
    pub(crate) seq_len: usize,
    /// The deterministic pack seed (FIM draws + the chunk shuffle).
    #[arg(long, default_value_t = 299_792_458)]
    pub(crate) seed: u64,
    /// Apply the lineage FIM transform to CODE documents.
    #[arg(long)]
    pub(crate) fim: bool,
}

#[derive(Debug, clap::Args)]
pub(crate) struct MixRenderArgs {
    /// The render spec: one row per corpus tree.
    #[arg(long)]
    pub(crate) spec: PathBuf,
    /// Output directory; one document table lands per spec row.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Verbatim added/created stamp; absent = empty.
    #[arg(long)]
    pub(crate) stamp: Option<String>,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum DocCommand {
    /// Print the whole command tree, one summary line per node.
    Cli,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum CapabilityCommand {
    /// Drive the engine over rendered olmo-eval fixture requests.
    Run(CapabilityRunArgs),
    /// Mirror fixture requests and run predictions into nuon.
    Convert(CapabilityConvertArgs),
    /// Score a converted run: per-item scores plus aggregates.
    Score(CapabilityScoreArgs),
    /// Render a nuon run's predictions back to their JSONL tree.
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
    /// Record plain greedy transcripts from a fixture plan.
    Record(SpeculateRecordArgs),
    /// Replay recorded streams under every policy and cost model.
    Simulate(SpeculateSimulateArgs),
    /// Print a transcript's emitted tokens as decoded pieces.
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
    /// The recorded-transcript directory, with any phase siblings.
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
    /// Fixture render root; absent = the capability home's.
    #[arg(long)]
    pub(crate) fixtures: Option<PathBuf>,
    /// Runs root; absent = the capability home's.
    #[arg(long)]
    pub(crate) runs: Option<PathBuf>,
    /// Nuon output root; absent = the capability home's nuon.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Comma-separated task-token filter; absent = every task found.
    #[arg(long)]
    pub(crate) tasks: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct CapabilityScoreArgs {
    /// Converted nuon run directory; absent = the default engine run.
    #[arg(long)]
    pub(crate) run: Option<PathBuf>,
    /// Converted nuon fixtures root; absent = the capability home's.
    #[arg(long)]
    pub(crate) fixtures: Option<PathBuf>,
    /// Output root; absent = the run directory itself.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Checker-data directory; absent = the capability home's.
    #[arg(long)]
    pub(crate) checker_data: Option<PathBuf>,
    /// Comma-separated task-token filter; absent = every task found.
    #[arg(long)]
    pub(crate) tasks: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct CapabilityRunArgs {
    /// Fixture root; absent = the capability home's fixtures.
    #[arg(long)]
    pub(crate) fixtures: Option<PathBuf>,
    /// Output root; absent = the capability home's run for these settings.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Comma-separated task_name filter; absent = every task found.
    #[arg(long)]
    pub(crate) tasks: Option<String>,
}
