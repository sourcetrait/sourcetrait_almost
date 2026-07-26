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
    /// Our own bench: does the model do the job we built it for.
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
    /// Task generators: synthesised examples whose answers are
    /// mechanically verifiable, for every stage plus the bench.
    Taskgen {
        #[command(subcommand)]
        command: TaskgenCommand,
    },
    /// Sampled, verifier-graded generation - the reinforcement
    /// stage's data source.
    Rollout {
        #[command(subcommand)]
        command: RolloutCommand,
    },
    /// The Speculation:DepthProbe instrument (recorded greedy streams
    /// + the offline policy/cost-model simulator).
    Speculate {
        #[command(subcommand)]
        command: SpeculateCommand,
    },
    /// The training stages: gradient locks, continued pretraining,
    /// and the supervised, preference and reinforcement legs.
    Train {
        #[command(subcommand)]
        command: TrainCommand,
    },
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum TrainCommand {
    /// The toy-config gradient locks on cpu f32: adapter-off
    /// exactness, chain-vs-full gradient equivalence, 60-step
    /// descent (needs a train build).
    Gate,
    /// LoRA CPT over a packed-chunks artifact -> one adapter
    /// safetensors + a NUON-lines step log (needs a train-cuda
    /// build; the model resolves through the global -c config).
    Cpt(TrainCptArgs),
    /// Supervised tuning over a packed instruction artifact: the loss
    /// covers the assistant turns alone.
    Sft(TrainSftArgs),
    /// Preference tuning over pairs, graded against the frozen base
    /// (the adapter-off path is the reference model).
    Dpo(TrainDpoArgs),
    /// Reinforcement tuning over verifier-scored rollout groups.
    Rlvr(TrainRlvrArgs),
}

/// The knobs every stage loop shares, so a posture reads the same
/// whichever objective is running.
#[derive(Debug, clap::Args)]
pub(crate) struct StageArgs {
    /// The adapter artifact path (.safetensors).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Continue a previous stage's adapter rather than starting from
    /// the base - the checkpoint chain. Rank and alpha then come from
    /// that artifact and --rank / --alpha are ignored.
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

#[derive(Debug, clap::Args)]
pub(crate) struct TrainSftArgs {
    /// A packed instruction artifact (from `mix instruct`), carrying
    /// ids and their loss mask.
    #[arg(long)]
    pub(crate) chunks: PathBuf,
    /// Rows folded into one optimizer step.
    #[arg(long, default_value_t = 1)]
    pub(crate) accumulate: usize,
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
    /// Answer every bench prompt greedily and report the pass rate -
    /// what the model can DO, against the general battery's reading
    /// of what a posture is breaking.
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
    /// Every family at once: NUON conversion and formatting, nushell
    /// from a shell command and from prose, and error location. Each
    /// answer is checked by machine before it ships.
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
pub(crate) enum RolloutCommand {
    /// Sample replies to verifiable prompts through the engine, grade
    /// each against its verifier, and emit scored groups.
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
    /// Replies sampled per prompt - the group an advantage is
    /// computed over. One carries no relative signal.
    #[arg(long, default_value_t = 8)]
    pub(crate) group: usize,
    /// Decode budget per reply.
    #[arg(long, default_value_t = 256)]
    pub(crate) max_tokens: usize,
    /// Sampling temperature; the group needs spread, so greedy would
    /// make every member identical.
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
    /// LoRA rank (the library-wide fixed-rank discipline decides the
    /// production value; sweeps ride this knob).
    #[arg(long, default_value_t = 64)]
    pub(crate) rank: usize,
    /// LoRA alpha; absent = 2 * rank (the era-one anchor).
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
    /// Pack document tables into shuffled training chunks (EOS-joined
    /// token stream, seq_len + 1 rows, optional FIM on code).
    Pack(MixPackArgs),
    /// Render corpus trees into dolma-field document tables (one
    /// file per document, verbatim text, identity in metadata).
    Render(MixRenderArgs),
    /// Sample documents from tables (one seeded shuffle across the
    /// union) until a text-byte budget is crossed - the admixture
    /// leg sampler.
    Sample(MixSampleArgs),
    /// Pack instruction examples one per row, masking the assistant
    /// turns, for supervised tuning.
    Instruct(MixInstructArgs),
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
    /// The window each example pads to; a longer example is DROPPED
    /// and reported, never truncated - a clipped reply teaches the
    /// model to stop mid-answer.
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
    /// Stop once cumulative text bytes cross this budget (the last
    /// document overshoots; the overshoot is reported).
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
    /// The packed-chunks artifact (.safetensors; a .nuon provenance
    /// sidecar lands beside it).
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
