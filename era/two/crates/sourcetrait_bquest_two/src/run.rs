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
    };
    if let Err(error) = outcome {
        eprintln!("bquest: {error}");
        std::process::exit(1);
    }
}
