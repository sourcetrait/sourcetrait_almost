//! The lastmost command-line surface: verbs over ai2's original stack.
use crate::*;

/// Baselining of the original model and its original tools.
#[derive(Debug, clap::Parser)]
#[command(name = "lastmost", version, about)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum Command {
    /// Original-stack generation through the pinned python environment.
    Generate(GenerateArgs),
    /// All-position f32 logits dump (the era-one safetensors contract).
    Dump(DumpArgs),
    /// Compare two logits dumps: nmse + per-row argmax (same-ids only).
    Diff(DiffArgs),
    /// Evaluation batteries through the original stack.
    Eval(EvalArgs),
}

/// Arguments for `lastmost eval`.
#[derive(Debug, clap::Args)]
pub(crate) struct EvalArgs {
    #[command(subcommand)]
    pub(crate) suite: EvalSuite,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum EvalSuite {
    /// Needle/passkey retrieval grid at one context length.
    Needle(NeedleArgs),
}

/// Arguments for `lastmost eval needle`.
#[derive(Debug, clap::Args)]
pub(crate) struct NeedleArgs {
    /// Which local hybrid checkpoint to drive.
    #[arg(long, value_enum, default_value = "dpo")]
    pub(crate) model: ModelPick,
    /// Explicit checkpoint directory (overrides --model).
    #[arg(long)]
    pub(crate) model_dir: Option<PathBuf>,
    /// Battery spec json (depths, keys, filler seed, templates).
    #[arg(long)]
    pub(crate) spec: PathBuf,
    /// Target total prompt tokens for this batch.
    #[arg(long)]
    pub(crate) length: u64,
    /// hf = transformers eager/fla; vllm = the production serving grade.
    #[arg(long, value_enum, default_value = "hf")]
    pub(crate) backend: BackendPick,
    /// vllm only: fraction of total VRAM the engine may claim.
    #[arg(long, default_value_t = 0.82)]
    pub(crate) gpu_mem_util: f64,
    /// vllm only: skip cuda-graph capture.
    #[arg(long)]
    pub(crate) enforce_eager: bool,
    /// vllm only: allow max_model_len beyond the config cap (safe on the
    /// NoPE hybrid; sets VLLM_ALLOW_LONG_MAX_MODEL_LEN=1).
    #[arg(long)]
    pub(crate) allow_long: bool,
    #[arg(long, value_enum, default_value = "cuda")]
    pub(crate) device: DevicePick,
    #[arg(long, value_enum, default_value = "bf16")]
    pub(crate) dtype: DtypePick,
    /// sdpa is the battery default (eager OOMs at 8K+ on Tier A).
    #[arg(long, value_enum, default_value = "sdpa")]
    pub(crate) attn: AttnPick,
    /// Force the in-tree torch GDN paths (auto-forced on cpu).
    #[arg(long)]
    pub(crate) no_fla: bool,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    /// Write the full JSON-lines event record here.
    #[arg(long)]
    pub(crate) record: Option<PathBuf>,
}

/// Arguments for `lastmost generate`. Prompts ride files, never argv.
#[derive(Debug, clap::Args)]
pub(crate) struct GenerateArgs {
    /// Which local hybrid checkpoint to drive.
    #[arg(long, value_enum, default_value = "dpo")]
    pub(crate) model: ModelPick,
    /// Explicit checkpoint directory (overrides --model).
    #[arg(long)]
    pub(crate) model_dir: Option<PathBuf>,
    /// File holding the prompt text.
    #[arg(long)]
    pub(crate) prompt_file: PathBuf,
    /// No chat template; pure continuation.
    #[arg(long)]
    pub(crate) raw: bool,
    /// Sampled decode (default greedy).
    #[arg(long)]
    pub(crate) sample: bool,
    #[arg(long, default_value_t = 0.6)]
    pub(crate) temperature: f64,
    #[arg(long, default_value_t = 0.95)]
    pub(crate) top_p: f64,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    #[arg(long, default_value_t = 64)]
    pub(crate) max_new_tokens: usize,
    #[arg(long, value_enum, default_value = "cuda")]
    pub(crate) device: DevicePick,
    #[arg(long, value_enum, default_value = "bf16")]
    pub(crate) dtype: DtypePick,
    /// Attention implementation on the 8 full-attention layers.
    #[arg(long, value_enum, default_value = "eager")]
    pub(crate) attn: AttnPick,
    /// Force the in-tree torch GDN paths (auto-forced on cpu).
    #[arg(long)]
    pub(crate) no_fla: bool,
    /// torch.use_deterministic_algorithms(True) in the driver.
    #[arg(long)]
    pub(crate) determinism: bool,
    /// Stop token string (repeatable; defaults per template mode).
    #[arg(long)]
    pub(crate) stop: Vec<String>,
    /// Write the full JSON-lines event record here.
    #[arg(long)]
    pub(crate) record: Option<PathBuf>,
}

/// Arguments for `lastmost dump`.
#[derive(Debug, clap::Args)]
pub(crate) struct DumpArgs {
    /// Which local hybrid checkpoint to drive.
    #[arg(long, value_enum, default_value = "dpo")]
    pub(crate) model: ModelPick,
    /// Explicit checkpoint directory (overrides --model).
    #[arg(long)]
    pub(crate) model_dir: Option<PathBuf>,
    /// Output safetensors path (parent dirs are created).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Prompt file (exactly one of --prompt-file / --ids-from).
    #[arg(long)]
    pub(crate) prompt_file: Option<PathBuf>,
    /// Replay this dump's prompt+fed ids verbatim.
    #[arg(long)]
    pub(crate) ids_from: Option<PathBuf>,
    /// Greedy tokens to generate when not replaying.
    #[arg(long = "gen", default_value_t = 64)]
    pub(crate) gen_tokens: usize,
    /// single = chunked prefill path; incremental = recurrent decode path.
    #[arg(long, value_enum, default_value = "single")]
    pub(crate) mode: ModePick,
    /// No chat template; pure continuation.
    #[arg(long)]
    pub(crate) raw: bool,
    #[arg(long, value_enum, default_value = "cuda")]
    pub(crate) device: DevicePick,
    #[arg(long, value_enum, default_value = "bf16")]
    pub(crate) dtype: DtypePick,
    #[arg(long, value_enum, default_value = "eager")]
    pub(crate) attn: AttnPick,
    /// Force the in-tree torch GDN paths (auto-forced on cpu).
    #[arg(long)]
    pub(crate) no_fla: bool,
    /// torch.use_deterministic_algorithms(True) in the driver.
    #[arg(long)]
    pub(crate) determinism: bool,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    /// Write the full JSON-lines event record here.
    #[arg(long)]
    pub(crate) record: Option<PathBuf>,
}

/// Arguments for `lastmost diff`.
#[derive(Debug, clap::Args)]
pub(crate) struct DiffArgs {
    /// The dump under test.
    pub(crate) candidate: PathBuf,
    /// The reference dump.
    pub(crate) reference: PathBuf,
    /// Worst rows to report.
    #[arg(long, default_value_t = 5)]
    pub(crate) top: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ModePick {
    Single,
    Incremental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum BackendPick {
    Hf,
    Vllm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum DevicePick {
    Cuda,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum DtypePick {
    Bf16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum AttnPick {
    Eager,
    Sdpa,
}
