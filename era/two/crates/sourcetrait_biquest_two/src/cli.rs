//! The command surface: a global profile flagset over a verb tree.
use crate::*;

/// biquest: the QuestImaginary foundation and training tool.
#[derive(Debug, clap::Parser)]
#[command(name = "biquest", version, about)]
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
    /// Self-documentation of the always-moving surface.
    Doc {
        #[command(subcommand)]
        command: DocCommand,
    },
    /// Tokenize a string and print the token table.
    Tokenize(TokenizeArgs),
    /// The ImagineQuestTokenizer: layers, census, admission, ledger.
    Tokenizer {
        #[command(subcommand)]
        command: TokenizerCommand,
    },
}

#[derive(Debug, clap::Args)]
pub(crate) struct TokenizeArgs {
    /// The text to tokenize.
    pub(crate) text: String,
    /// An admitted wordlist (.nuon) as the dictionary; absent = the
    /// embedded full English set.
    #[arg(long)]
    pub(crate) admitted: Option<PathBuf>,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum DocCommand {
    /// Print the whole command tree, one summary line per node.
    Cli,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum TokenizerCommand {
    /// Report the embedded character layer's shape.
    Ucd,
    /// Extract the folded single-piece word set from a wiktextract dump.
    Dictionary(TokenizerDictionaryArgs),
    /// Count folded word types across corpus trees.
    Census(TokenizerCensusArgs),
    /// Gate census counts through dictionary membership.
    Admit(TokenizerAdmitArgs),
    /// Measure Quill tokens against the checkpoint tokenizer.
    Ledger(TokenizerLedgerArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct TokenizerDictionaryArgs {
    /// The wiktextract English dump (.jsonl, decompressed).
    #[arg(long)]
    pub(crate) dump: PathBuf,
    /// The word-set artifact: one folded word per line, sorted.
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TokenizerCensusArgs {
    /// Corpus files or trees, walked sorted.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) roots: Vec<PathBuf>,
    /// The type-count artifact: word, tab, count; census order.
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TokenizerAdmitArgs {
    /// A census type-count artifact.
    #[arg(long)]
    pub(crate) counts: PathBuf,
    /// A dictionary word-set artifact.
    #[arg(long)]
    pub(crate) words: PathBuf,
    /// The admitted wordlist (.nuon); row order is the id order.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Census occurrences below this are not admitted.
    #[arg(long, default_value_t = 1)]
    pub(crate) min_count: u64,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TokenizerLedgerArgs {
    /// The admitted wordlist (.nuon).
    #[arg(long)]
    pub(crate) admitted: PathBuf,
    /// Corpus files or trees, walked sorted.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) roots: Vec<PathBuf>,
    /// The ledger report (.nuon), per-file rows included.
    #[arg(long)]
    pub(crate) out: PathBuf,
}
