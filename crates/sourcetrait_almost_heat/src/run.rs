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
            device,
            model_dir,
        } => {
            let device = match device.as_str() {
                "cpu" => dump::OracleDevice::Cpu,
                "cuda" => dump::OracleDevice::Cuda,
                other => snafu::whatever!("unsupported device {other}; use cpu or cuda"),
            };
            let dir = model_dir.unwrap_or_else(cli::default_model_dir);
            dump::dump(&dir, &ids_from, &out, device)
        }
        Command::Diff {
            candidate,
            reference,
            top,
        } => diff::diff(&candidate, &reference, top),
        #[cfg(feature = "train")]
        Command::Train {
            data,
            exclude,
            out,
            tokenizer,
            model_dir,
            seq_len,
            steps,
            rank,
            alpha,
            lr,
            warmup,
            seed,
            loss_chunk,
            log_every,
        } => {
            let model_dir = model_dir.unwrap_or_else(cli::default_model_dir);
            let tokenizer = tokenizer.unwrap_or_else(|| model_dir.join("tokenizer.json"));
            let options = train::TrainOptions {
                data_roots: data,
                exclude_dirs: exclude,
                tokenizer,
                model_dir,
                out,
                seq_len,
                steps,
                rank,
                alpha: if alpha == 0.0 { 2.0 * rank as f64 } else { alpha },
                learning_rate: lr,
                warmup_steps: warmup,
                seed,
                loss_chunk,
                log_every,
            };
            train::train(&options)
        }
    }
}
