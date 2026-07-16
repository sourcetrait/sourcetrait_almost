//! Verb dispatch for the lastmost binary.
use crate::*;

/// Parse the cli and dispatch; non-zero process exit on error.
pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    let result = match &cli.command {
        Command::Generate(args) => generate(args),
        Command::Dump(args) => dump(args),
        Command::Diff(args) => diff(args),
        Command::Eval(args) => eval(args),
        Command::Bench(args) => bench(args),
    };
    if let Err(e) = result {
        eprintln!("lastmost: {e}");
        process::exit(1);
    }
}
