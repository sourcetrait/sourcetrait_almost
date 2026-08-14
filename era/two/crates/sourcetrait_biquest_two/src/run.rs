use crate::*;

pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    let outcome = match &cli.command {
        Command::Assemble(args) => assemble_text(args),
        Command::Associations { command } => match command {
            AssociationsCommand::Build(args) => associations_build(args),
        },
        Command::Disassemble(args) => disassemble_wire(args),
        Command::Doc { command } => match command {
            DocCommand::Cli => doc_cli(),
        },
        Command::Matrix { command } => match command {
            MatrixCommand::Build(args) => matrix_build(args),
        },
        Command::Tokenize(args) => tokenize_text(args),
        Command::Tokenizer { command } => match command {
            TokenizerCommand::Ucd => tokenizer_ucd(),
            TokenizerCommand::Dictionary(args) => tokenizer_dictionary(args),
            TokenizerCommand::Ledger(args) => tokenizer_ledger(&cli, args),
        },
        Command::Trainer { command } => match command {
            TrainerCommand::Init(args) => trainer_init(args),
            TrainerCommand::Train(args) => trainer_train(args),
        },
    };
    if let Err(error) = outcome {
        eprintln!("biquest: {error}");
        std::process::exit(1);
    }
}
