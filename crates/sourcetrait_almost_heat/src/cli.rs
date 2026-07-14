use crate::*;

/// heat: independent burn-based Olmo 3 parity oracle (checks the burn).
#[derive(Debug, clap::Parser)]
#[command(name = "heat", version, about)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum Command {
    /// Replay an lmst dump's token ids through the burn model, writing a
    /// dump in the same format
    Dump {
        /// lmst parity dump supplying prompt_ids + fed_ids
        #[arg(long)]
        ids_from: PathBuf,
        /// Output dump path
        #[arg(long)]
        out: PathBuf,
        /// cpu (ndarray f32, reference grade) or cuda (bf16, fast grade;
        /// needs a --features cuda build)
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Checkpoint directory (config.json + safetensors); defaults to
        /// lmst's default model dir
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
    /// Compare the logits of two dumps (candidate vs reference)
    Diff {
        candidate: PathBuf,
        reference: PathBuf,
        /// How many worst rows to list
        #[arg(long, default_value_t = 8)]
        top: usize,
    },
    /// Train a LoRA adapter over the frozen checkpoint (plain LM loss
    /// over packed corpus chunks; needs a --features cuda,train build)
    #[cfg(feature = "train")]
    Train {
        /// Corpus roots, walked recursively (Cargo.toml, .rs, .nu,
        /// .nuon, README.md)
        #[arg(long, required = true)]
        data: Vec<PathBuf>,
        /// Directory names to skip while walking (dot-dirs always skip)
        #[arg(long)]
        exclude: Vec<String>,
        /// Adapter output path (safetensors)
        #[arg(long)]
        out: PathBuf,
        /// tokenizer.json; defaults to <model-dir>/tokenizer.json
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Checkpoint directory; defaults to the shared model dir
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 1024)]
        seq_len: usize,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 64)]
        rank: usize,
        /// LoRA alpha; 0 resolves to 2 * rank
        #[arg(long, default_value_t = 0.0)]
        alpha: f64,
        #[arg(long, default_value_t = 2e-4)]
        lr: f64,
        /// Linear lr warmup steps
        #[arg(long, default_value_t = 10)]
        warmup: usize,
        /// Chunk-shuffle seed
        #[arg(long, default_value_t = 299_792_458)]
        seed: u64,
        /// Rows per lm-head/loss chunk (bounds the logits transient)
        #[arg(long, default_value_t = 512)]
        loss_chunk: usize,
        #[arg(long, default_value_t = 10)]
        log_every: usize,
    },
}

/// Mirrors lmst's default layout so the two binaries share a checkpoint:
/// $XDG_CACHE_HOME/huggingface/model/<owner>--<name>, ~/.cache the fallback
/// when the variable is unset or empty (the XDG spec's own default).
pub(crate) fn default_model_dir() -> PathBuf {
    let base = match std::env::var("XDG_CACHE_HOME") {
        Ok(cache_home) if !cache_home.is_empty() => PathBuf::from(cache_home),
        _ => {
            let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
            PathBuf::from(home).join(".cache")
        }
    };
    base.join("huggingface")
        .join("model")
        .join("allenai--Olmo-3-7B-Instruct")
}
