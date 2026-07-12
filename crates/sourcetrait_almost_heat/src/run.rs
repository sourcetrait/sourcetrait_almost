use crate::*;

/// Binary entry: parse, dispatch, exit non-zero on error.
pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    if let Err(error) = dispatch(cli) {
        eprintln!("heat: {error}");
        std::process::exit(1);
    }
}

fn dispatch(cli: Cli) -> HeatResult<()> {
    match cli.command {
        Command::Dump {
            ids_from,
            out,
            model_dir,
        } => {
            let dir = model_dir.unwrap_or_else(cli::default_model_dir);
            dump::dump(&dir, &ids_from, &out)
        }
        Command::Diff {
            candidate,
            reference,
            top,
        } => diff::diff(&candidate, &reference, top),
    }
}
