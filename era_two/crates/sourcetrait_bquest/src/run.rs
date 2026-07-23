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
