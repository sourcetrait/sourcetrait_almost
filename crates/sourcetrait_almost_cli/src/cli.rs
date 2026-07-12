use crate::*;

/// almost: the Olmo 3 7B Instruct engine (thin drivers over the lib port).
#[derive(Debug, clap::Parser)]
#[command(name = "almost", version, about)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum Command {
    /// Download the checkpoint (config, tokenizer, safetensors shards)
    Pull {
        #[arg(long, default_value = lib::consts::DEFAULT_MODEL_ID)]
        model_id: String,
        /// Defaults to $XDG_CACHE_HOME/huggingface/model/<owner>--<name>
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
    /// Generate a completion for a single prompt
    Prompt {
        prompt: String,
        /// Feed the prompt raw, skipping the chat-template wrap
        #[arg(long)]
        raw: bool,
        /// Argmax decoding (deterministic); overrides temperature/top-p
        #[arg(long)]
        greedy: bool,
        #[arg(long, default_value_t = lib::consts::DEFAULT_TEMPERATURE)]
        temperature: f64,
        #[arg(long, default_value_t = lib::consts::DEFAULT_TOP_P)]
        top_p: f64,
        /// Maximum new tokens to generate
        #[arg(long, default_value_t = 256)]
        sample_len: usize,
        /// Force CPU even when CUDA is available
        #[arg(long)]
        cpu: bool,
        /// bf16 or f32; defaults to bf16 on cuda and f32 on cpu
        #[arg(long)]
        dtype: Option<String>,
        #[arg(long, default_value_t = 299792458)]
        seed: u64,
        /// Write a parity dump (logits over every fed position) here
        #[arg(long)]
        dump_logits: Option<PathBuf>,
        #[arg(long, default_value = lib::consts::DEFAULT_MODEL_ID)]
        model_id: String,
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
    /// Run the internal equivalence battery against the checkpoint
    Verify {
        /// Use a prompt longer than the sliding window (exercises the D3
        /// trim, the banded window mask, and the yarn/vanilla split)
        #[arg(long)]
        long: bool,
        /// Also diff against a cpu f32 run (informational)
        #[arg(long)]
        cross_device: bool,
        /// Trailing positions covered by the incremental-decode check
        #[arg(long, default_value_t = 16)]
        decode_steps: usize,
        /// Force CPU even when CUDA is available
        #[arg(long)]
        cpu: bool,
        /// bf16 or f32; defaults to bf16 on cuda and f32 on cpu
        #[arg(long)]
        dtype: Option<String>,
        #[arg(long, default_value = lib::consts::DEFAULT_MODEL_ID)]
        model_id: String,
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
}
