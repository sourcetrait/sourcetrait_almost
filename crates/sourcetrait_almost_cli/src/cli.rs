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
        /// Config profile snake or TOML path (model intent); absent =
        /// config/default.toml, else code defaults
        #[arg(short = 'c', long)]
        config: Option<String>,
    },
    /// Generate a completion for a single prompt
    Prompt {
        prompt: String,
        /// Feed the prompt raw, skipping the chat-template wrap
        #[arg(long)]
        raw: bool,
        /// Config profile snake or TOML path (declared intent: model,
        /// device, dtype, generation, eviction)
        #[arg(short = 'c', long)]
        config: Option<String>,
        /// Settings profile snake or TOML path (surgical runtime
        /// overrides; absent auto-pairs the config profile's name, else
        /// derives reactively via Settings::from_config)
        #[arg(short = 's', long)]
        settings: Option<String>,
        /// Write a parity dump (logits over every fed position) here
        #[arg(long)]
        dump_logits: Option<PathBuf>,
    },
    /// Run the internal equivalence battery (default) or the retrieval
    /// needle battery (--needle) against the checkpoint
    Verify {
        /// Config profile snake or TOML path
        #[arg(short = 'c', long)]
        config: Option<String>,
        /// Settings profile snake or TOML path
        #[arg(short = 's', long)]
        settings: Option<String>,
        /// Equivalence battery: use a prompt longer than the sliding
        /// window (exercises the D3 trim, the banded window mask, and
        /// the yarn/vanilla split)
        #[arg(long)]
        long: bool,
        /// Equivalence battery: also diff against a cpu f32 run
        /// (informational)
        #[arg(long)]
        cross_device: bool,
        /// Equivalence battery: trailing positions covered by the
        /// incremental-decode check
        #[arg(long, default_value_t = 16)]
        decode_steps: usize,
        /// Run the retrieval needle/passkey battery instead (A2 phase 1:
        /// depth x length passkey grid, single + multi, exact match)
        #[arg(long)]
        needle: bool,
        /// With --needle: write the battery's per-cell results here as
        /// JSON
        #[arg(long, requires = "needle")]
        needle_out: Option<PathBuf>,
        /// With --needle: run the A2 observation pass (attn-profile
        /// build, eager) and write the per-head attention-mass profile
        /// here as JSON
        #[arg(long, requires = "needle")]
        profile_out: Option<PathBuf>,
        /// Needle battery case-generator seed (decoding stays greedy)
        #[arg(long, default_value_t = 299792458)]
        seed: u64,
    },
}
