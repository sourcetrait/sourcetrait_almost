use crate::*;

pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    let outcome = match &cli.command {
        Command::Doc { command } => match command {
            DocCommand::Cli => doc_cli(),
        },
        Command::Tokenize(args) => tokenize_text(args),
        Command::Tokenizer { command } => match command {
            TokenizerCommand::Ucd => tokenizer_ucd(),
            TokenizerCommand::Dictionary(args) => tokenizer_dictionary(args),
            TokenizerCommand::Census(args) => tokenizer_census(args),
            TokenizerCommand::Admit(args) => tokenizer_admit(args),
            TokenizerCommand::Ledger(args) => tokenizer_ledger(&cli, args),
        },
    };
    if let Err(error) = outcome {
        eprintln!("biquest: {error}");
        std::process::exit(1);
    }
}
