use crate::*;

pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    let outcome = match &cli.command {
        Command::Capability { command } => match command {
            CapabilityCommand::Run(args) => capability_run(&cli, args),
        },
    };
    if let Err(error) = outcome {
        eprintln!("lmst: {error}");
        std::process::exit(1);
    }
}
