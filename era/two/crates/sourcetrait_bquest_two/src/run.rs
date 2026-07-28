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
            MixCommand::Sample(args) => mix_sample(args),
            MixCommand::Instruct(args) => mix_instruct(&cli, args),
            MixCommand::Rip(args) => mix_rip(args),
        },
        Command::Bench { command } => match command {
            BenchCommand::Run(args) => bench_run(&cli, args),
        },
        Command::Taskgen { command } => match command {
            TaskgenCommand::All(args) => taskgen_all(args),
        },
        Command::Syllabus { command } => match command {
            SyllabusCommand::Emit(args) => syllabus_emit(args),
        },
        Command::Rollout { command } => match command {
            RolloutCommand::Run(args) => rollout_run(&cli, args),
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
        TrainCommand::Sft(args) => train_sft(cli, args),
        TrainCommand::Dpo(args) => train_dpo(cli, args),
        TrainCommand::Rlvr(args) => train_rlvr(cli, args),
    }
}

#[cfg(not(feature = "train"))]
fn train_dispatch(_cli: &Cli, _command: &TrainCommand) -> BquestResult<()> {
    snafu::whatever!("the train verbs need a train build (--features train / train-cuda)")
}
