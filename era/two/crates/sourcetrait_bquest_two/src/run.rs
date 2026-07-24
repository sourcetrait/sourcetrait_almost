use crate::*;

pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    let outcome = match &cli.command {
        Command::Capability { command } => match command {
            CapabilityCommand::Run(args) => capability_run(&cli, args),
            CapabilityCommand::Convert(args) => capability_convert(args),
            CapabilityCommand::Score(args) => capability_score(args),
            CapabilityCommand::Bridge(args) => capability_bridge(args),
        },
        Command::Doc { command } => match command {
            DocCommand::Cli => doc_cli(),
        },
        Command::Mix { command } => match command {
            MixCommand::Pack(args) => mix_pack(&cli, args),
            MixCommand::Render(args) => mix_render(args),
        },
        Command::Speculate { command } => match command {
            SpeculateCommand::Record(args) => speculate_record(&cli, args),
            SpeculateCommand::Simulate(args) => speculate_simulate(args),
            SpeculateCommand::Tokens(args) => speculate_tokens(&cli, args),
        },
        Command::Train { command } => train_dispatch(&cli, command),
    };
    if let Err(error) = outcome {
        eprintln!("bquest: {error}");
        std::process::exit(1);
    }
}

#[cfg(feature = "train")]
fn train_dispatch(cli: &Cli, command: &TrainCommand) -> BquestResult<()> {
    match command {
        TrainCommand::Gate => train_gate_verb(),
        TrainCommand::Cpt(args) => train_cpt(cli, args),
    }
}

#[cfg(not(feature = "train"))]
fn train_dispatch(_cli: &Cli, _command: &TrainCommand) -> BquestResult<()> {
    snafu::whatever!("the train verbs need a train build (--features train / train-cuda)")
}
