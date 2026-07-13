use crate::*;

/// Binary entry: parse, dispatch, exit non-zero on error. stdout carries
/// payload only (generated text, verify table); diagnostics go to stderr.
pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    if let Err(error) = dispatch(cli) {
        eprintln!("almost: {error}");
        std::process::exit(1);
    }
}

fn dispatch(cli: Cli) -> lib::AlmostResult<()> {
    match cli.command {
        Command::Pull { model_id, model_dir } => {
            let dir = model_dir.unwrap_or_else(|| lib::default_model_dir(&model_id));
            let paths = lib::ensure_model(&model_id, &dir)?;
            eprintln!(
                "almost: model ready at {} ({} shards)",
                dir.display(),
                paths.shards.len()
            );
            Ok(())
        }
        Command::Prompt {
            prompt,
            raw,
            greedy,
            flash,
            speculate,
            temperature,
            top_p,
            sample_len,
            cpu,
            dtype,
            seed,
            dump_logits,
            model_id,
            model_dir,
        } => {
            let dir = model_dir.unwrap_or_else(|| lib::default_model_dir(&model_id));
            let paths = lib::ensure_model(&model_id, &dir)?;
            let device = lib::pick_device(cpu)?;
            let dtype = lib::pick_dtype(dtype.as_deref(), &device)?;
            eprintln!(
                "almost: device {device:?}, dtype {dtype:?}{}",
                if flash { ", flash" } else { "" }
            );

            let settings = lib::Settings {
                use_flash_attn: flash,
            };
            let loaded = lib::load_model(&paths, &device, dtype, settings)?;
            eprintln!("almost: weights loaded in {:.1}s", loaded.load_seconds);
            let mut model = loaded.model;

            let text = if raw { prompt } else { lib::chat_wrap(&prompt) };
            let options = lib::GenerateOptions {
                greedy,
                temperature,
                top_p,
                sample_len,
                seed,
                speculate,
                dump_logits,
            };
            let mut generation = model.generate(&loaded.tokenizer, &text, &options)?;
            for step in &mut generation {
                let step = step?;
                if let Some(chunk) = step.chunk {
                    print!("{chunk}");
                    io::stdout().flush()?;
                }
            }
            let report = generation.finish()?;
            if let Some(rest) = &report.rest {
                print!("{rest}");
            }
            println!();
            if let Some(path) = &options.dump_logits {
                eprintln!("almost: logits dumped to {}", path.display());
            }
            let speculation_note = if report.drafted_token_count > 0 {
                format!(
                    "; drafted {} accepted {} ({:.0}%)",
                    report.drafted_token_count,
                    report.accepted_draft_token_count,
                    100.0 * report.accepted_draft_token_count as f64
                        / report.drafted_token_count as f64
                )
            } else {
                String::new()
            };
            eprintln!(
                "almost: prefill {} tokens in {:.2}s ({:.1} tok/s); decode {} tokens in {:.2}s ({:.1} tok/s){}; stopped by {}",
                report.prompt_token_count,
                report.prefill_seconds,
                report.prompt_token_count as f64 / report.prefill_seconds.max(f64::EPSILON),
                report.generated_token_count,
                report.decode_seconds,
                report.generated_token_count as f64 / report.decode_seconds.max(f64::EPSILON),
                speculation_note,
                match report.finish_reason {
                    Some(lib::FinishReason::StopToken) => "stop token",
                    Some(lib::FinishReason::SampleLen) => "sample_len",
                    None => "early stop",
                },
            );
            Ok(())
        }
        Command::Verify {
            long,
            flash,
            cross_device,
            decode_steps,
            needle,
            needle_out,
            seed,
            cpu,
            dtype,
            model_id,
            model_dir,
        } => {
            let dir = model_dir.unwrap_or_else(|| lib::default_model_dir(&model_id));
            let paths = lib::ensure_model(&model_id, &dir)?;
            let device = lib::pick_device(cpu)?;
            let dtype = lib::pick_dtype(dtype.as_deref(), &device)?;
            if needle {
                let opts = lib::NeedleOptions {
                    use_flash_attn: flash,
                    seed,
                    out: needle_out,
                };
                return lib::needle(&paths, &device, dtype, &opts);
            }
            let opts = lib::VerifyOptions {
                long,
                cross_device,
                decode_steps,
                use_flash_attn: flash,
            };
            lib::verify(&paths, &device, dtype, &opts)
        }
    }
}
