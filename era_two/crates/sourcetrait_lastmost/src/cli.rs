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
