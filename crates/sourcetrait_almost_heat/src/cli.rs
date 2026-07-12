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
}

/// Mirrors lmst's default layout so the two binaries share a checkpoint.
pub(crate) fn default_model_dir() -> PathBuf {
    let base = match std::env::var("HF_HOME") {
        Ok(home) => PathBuf::from(home),
        Err(_) => {
            let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
            PathBuf::from(home).join(".cache").join("huggingface")
        }
    };
    base.join("lmst").join("allenai--Olmo-3-7B-Instruct")
}
