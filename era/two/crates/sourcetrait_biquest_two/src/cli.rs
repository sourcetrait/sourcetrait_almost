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
    /// Assemble Quill assembly text into wire token ids.
    Assemble(AssembleArgs),
    /// A Quill file to its token table (the assembler test surface).
    Assembler(AssemblerArgs),
    /// The ImagineQuestAssociations store.
    Associations {
        #[command(subcommand)]
        command: AssociationsCommand,
    },
    /// Disassemble wire token ids back into Quill assembly.
    Disassemble(DisassembleArgs),
    /// Self-documentation of the always-moving surface.
    Doc {
        #[command(subcommand)]
        command: DocCommand,
    },
    /// The ImagineQuestMatrix: the hand-built embedding artifact.
    Matrix {
        #[command(subcommand)]
        command: MatrixCommand,
    },
    /// Tokenize a string and print the token table.
    Tokenize(TokenizeArgs),
    /// The ImagineQuestTokenizer: layers and the ledger.
    Tokenizer {
        #[command(subcommand)]
        command: TokenizerCommand,
    },
    /// The BiquestTrainer: the organism's checkpoint and training.
    Trainer {
        #[command(subcommand)]
        command: TrainerCommand,
    },
    /// The WikimediaDumpTool: raw dump and export page access.
    Wikimedia {
        #[command(subcommand)]
        command: WikimediaCommand,
    },
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum WikimediaCommand {
    /// The dictionary-side corpus: every vocabulary word's document.
    Corpus(WikimediaCorpusArgs),
    /// A word's markdown document from its fold-matched page set.
    Document(WikimediaDocumentArgs),
    /// The ruled typography normalization over text or a file.
    Normalize(WikimediaNormalizeArgs),
    /// One page's wikitext from an export save or a dump.
    Page(WikimediaPageArgs),
    /// One page's parsed block tree, as a NUON table.
    Parse(WikimediaParseArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct WikimediaCorpusArgs {
    /// The pages-articles multistream dump (.xml.bz2).
    #[arg(long)]
    pub(crate) source: PathBuf,
    /// The multistream index (.txt or .txt.bz2).
    #[arg(long)]
    pub(crate) index: PathBuf,
    /// The corpus tree: sharded word documents plus audit.nuonl,
    /// missing_words.txt, and provenance.nuon.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// A dictionary word file (file order = id order); absent = the
    /// embedded full English set.
    #[arg(long)]
    pub(crate) words: Option<PathBuf>,
    /// Render at most this many words (a smoke lever).
    #[arg(long)]
    pub(crate) limit: Option<usize>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct WikimediaDocumentArgs {
    /// The vocabulary word (fold-matched against page titles).
    #[arg(long)]
    pub(crate) word: String,
    /// The page source: an export .xml or a pages-articles dump
    /// (.xml.bz2), streamed; with --index, the multistream dump,
    /// seeked per matching title.
    #[arg(long)]
    pub(crate) source: PathBuf,
    /// The multistream index (.txt or .txt.bz2) to seek with.
    #[arg(long)]
    pub(crate) index: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct WikimediaNormalizeArgs {
    /// The text to normalize; --file supplies it instead.
    pub(crate) text: Option<String>,
    /// Read the text from a file.
    #[arg(long)]
    pub(crate) file: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct WikimediaParseArgs {
    /// The exact page title (enwiktionary ns0 is case-sensitive).
    #[arg(long)]
    pub(crate) title: String,
    /// The page source: an export .xml or a pages-articles dump
    /// (.xml.bz2), streamed; with --index, the multistream dump,
    /// seeked to the title's block.
    #[arg(long)]
    pub(crate) source: PathBuf,
    /// The multistream index (.txt or .txt.bz2) to seek with.
    #[arg(long)]
    pub(crate) index: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct WikimediaPageArgs {
    /// The exact page title (enwiktionary ns0 is case-sensitive).
    #[arg(long)]
    pub(crate) title: String,
    /// The page source: an export .xml or a pages-articles dump
    /// (.xml.bz2), streamed; with --index, the multistream dump,
    /// seeked to the title's block.
    #[arg(long)]
    pub(crate) source: PathBuf,
    /// The multistream index (.txt or .txt.bz2) to seek with.
    #[arg(long)]
    pub(crate) index: Option<PathBuf>,
    /// Print the page record (text_bytes in place of text).
    #[arg(long)]
    pub(crate) meta: bool,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum TrainerCommand {
    /// The organism's fresh checkpoint from a matrix artifact.
    Init(TrainerInitArgs),
    /// Full-parameter training over a text corpus (a train build).
    Train(TrainerTrainArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct TrainerTrainArgs {
    /// The organism checkpoint directory to train from.
    #[arg(long)]
    pub(crate) organism: PathBuf,
    /// The trained checkpoint directory (log and provenance beside).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Corpus files or trees, tokenized and packed in walk order.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) roots: Vec<PathBuf>,
    /// A dictionary word file (file order = id order); absent = the
    /// embedded full English set.
    #[arg(long)]
    pub(crate) words: Option<PathBuf>,
    /// Positions per packed chunk (chunks carry seq_len + 1 ids).
    #[arg(long, default_value_t = 2048)]
    pub(crate) seq_len: usize,
    /// Optimizer steps to run.
    #[arg(long)]
    pub(crate) steps: usize,
    /// Peak learning rate (linear warmup, then constant).
    #[arg(long, default_value_t = 3e-4)]
    pub(crate) learning_rate: f64,
    /// Steps of linear warmup to the peak rate.
    #[arg(long, default_value_t = 100)]
    pub(crate) warmup_steps: usize,
    /// Chunks summed into one optimizer step.
    #[arg(long, default_value_t = 1)]
    pub(crate) accumulate: usize,
    /// Positions per loss chunk (the softmax-transient lever).
    #[arg(long, default_value_t = 128)]
    pub(crate) loss_chunk: usize,
    /// Steps between stderr progress lines.
    #[arg(long, default_value_t = 10)]
    pub(crate) log_every: usize,
    /// Steps between mid-run checkpoints; 0 saves the final one only.
    #[arg(long, default_value_t = 0)]
    pub(crate) checkpoint_every: usize,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TrainerInitArgs {
    /// The ImagineQuestMatrix artifact (.safetensors).
    #[arg(long)]
    pub(crate) matrix: PathBuf,
    /// The organism checkpoint directory.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Layer count; a multiple of four (three GDN then one attention).
    #[arg(long, default_value_t = 8)]
    pub(crate) layers: usize,
    /// The deterministic core-init seed.
    #[arg(long, default_value_t = 299_792_458)]
    pub(crate) seed: u64,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TokenizeArgs {
    /// The text to tokenize.
    pub(crate) text: String,
    /// A dictionary word file (one word per line, file order = id
    /// order); absent = the embedded full English set.
    #[arg(long)]
    pub(crate) words: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct AssemblerArgs {
    /// The Quill assembly file.
    pub(crate) file: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(crate) struct AssembleArgs {
    /// The assembly text; --file supplies it instead.
    pub(crate) text: Option<String>,
    /// Read the assembly from a file.
    #[arg(long)]
    pub(crate) file: Option<PathBuf>,
    /// A keyword-page binding table; absent = the embedded default
    /// (the vendored Syntax.nuon).
    #[arg(long)]
    pub(crate) syntax: Option<PathBuf>,
    /// A dictionary word file (file order = id order); absent = the
    /// embedded full English set.
    #[arg(long)]
    pub(crate) words: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct DisassembleArgs {
    /// The wire as a NUON int list; --file supplies it instead.
    pub(crate) ids: Option<String>,
    /// Read the wire (NUON int list) from a file.
    #[arg(long)]
    pub(crate) file: Option<PathBuf>,
    /// A keyword-page binding table; absent = the embedded default
    /// (the vendored Syntax.nuon).
    #[arg(long)]
    pub(crate) syntax: Option<PathBuf>,
    /// A dictionary word file (file order = id order); absent = the
    /// embedded full English set.
    #[arg(long)]
    pub(crate) words: Option<PathBuf>,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum AssociationsCommand {
    /// Build the whole store from the embedded UCD and the raw
    /// enwiktionary dump.
    Build(AssociationsBuildArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct AssociationsBuildArgs {
    /// The enwiktionary page source: an export .xml or a
    /// pages-articles dump (.xml.bz2), streamed whole.
    #[arg(long)]
    pub(crate) source: PathBuf,
    /// The store directory; the class files land inside it.
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum DocCommand {
    /// Print the whole command tree, one summary line per node.
    Cli,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum MatrixCommand {
    /// Build the associative embedding matrix from the embedded
    /// layers plus the dictionary word file.
    Build(MatrixBuildArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct MatrixBuildArgs {
    /// The dictionary word file (file order = id order).
    #[arg(long)]
    pub(crate) words: PathBuf,
    /// The embedding artifact (.safetensors).
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Width per head; hidden is 32 heads times this.
    #[arg(long, default_value_t = 32)]
    pub(crate) head_dim: usize,
    /// Unminted rows reserved above the dictionary for expansion.
    #[arg(long, default_value_t = 8192)]
    pub(crate) reserve: usize,
    /// The deterministic build seed.
    #[arg(long, default_value_t = 299_792_458)]
    pub(crate) seed: u64,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum TokenizerCommand {
    /// Report the embedded character layer's shape.
    Ucd,
    /// Extract the folded connected-word set from the raw dump.
    Dictionary(TokenizerDictionaryArgs),
    /// Measure Quill tokens against the checkpoint tokenizer.
    Ledger(TokenizerLedgerArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct TokenizerDictionaryArgs {
    /// The enwiktionary page source: an export .xml or a
    /// pages-articles dump (.xml.bz2), streamed whole.
    #[arg(long)]
    pub(crate) source: PathBuf,
    /// The word-set artifact: one folded word per line, sorted.
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TokenizerLedgerArgs {
    /// The dictionary word file (file order = id order).
    #[arg(long)]
    pub(crate) words: PathBuf,
    /// Corpus files or trees, walked sorted.
    #[arg(long, required = true, num_args = 1..)]
    pub(crate) roots: Vec<PathBuf>,
    /// The ledger report (.nuon), per-file rows included.
    #[arg(long)]
    pub(crate) out: PathBuf,
}
