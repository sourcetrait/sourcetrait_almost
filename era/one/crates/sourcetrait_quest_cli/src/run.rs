use crate::*;

/// Binary entry: parse, dispatch, exit non-zero on error. stdout carries
/// payload only (generated text, verify table); diagnostics go to stderr.
pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    if let Err(error) = dispatch(cli) {
        eprintln!("quest: {error}");
        std::process::exit(1);
    }
}

/// Resolve the -c/-s pair into (config, settings) plus the checkpoint
/// paths the run needs.
fn resolve_run(
    config_arg: Option<&str>,
    settings_arg: Option<&str>,
) -> lib::QuestResult<(lib::Config, lib::Settings, lib::ModelPaths)> {
    let (config, profile) = lib::load_config(config_arg)?;
    let settings = lib::load_settings(settings_arg, profile.as_deref(), &config)?;
    let dir = config
        .model_dir
        .clone()
        .unwrap_or_else(|| lib::default_model_dir(&config.model_id));
    let paths = lib::ensure_model(&config.model_id, &dir)?;
    Ok((config, settings, paths))
}

fn dispatch(cli: Cli) -> lib::QuestResult<()> {
    match cli.command {
        Command::Pull { config } => {
            let (config, _profile) = lib::load_config(config.as_deref())?;
            let dir = config
                .model_dir
                .clone()
                .unwrap_or_else(|| lib::default_model_dir(&config.model_id));
            let paths = lib::ensure_model(&config.model_id, &dir)?;
            eprintln!(
                "quest: model ready at {} ({} shards)",
                dir.display(),
                paths.shards.len()
            );
            Ok(())
        }
        Command::Prompt {
            prompt,
            raw,
            config,
            settings,
            from,
            to,
            dump_logits,
        } => {
            let (config, settings, paths) = resolve_run(config.as_deref(), settings.as_deref())?;
            let device = lib::resolve_device(&config)?;
            let dtype = lib::resolve_dtype(&config, &device);
            eprintln!(
                "quest: device {device:?}, dtype {dtype:?}{}",
                if settings.use_flash_attn { ", flash" } else { "" }
            );

            let loaded = lib::load_model(&paths, &device, dtype, settings)?;
            eprintln!("quest: weights loaded in {:.1}s", loaded.load_seconds);
            let mut model = loaded.model;

            let restored = match &from {
                Some(token) => {
                    let path = lib::resolve_snapshot(token);
                    let restored = model.restore_caches(&path, &config.model_id)?;
                    eprintln!(
                        "quest: restored {} context tokens from {}",
                        restored.context_len,
                        path.display()
                    );
                    Some(restored)
                }
                None => None,
            };
            let text = if raw {
                prompt
            } else if restored.is_some() {
                lib::chat_continue(&prompt)
            } else {
                lib::chat_wrap(&prompt)
            };
            let options = lib::GenerateOptions {
                greedy: config.generation.greedy,
                temperature: config.generation.temperature,
                top_p: config.generation.top_p,
                sample_len: config.generation.sample_len,
                seed: config.generation.seed,
                speculate: config.generation.speculate,
                dump_logits,
            };
            let mut generation = match &restored {
                Some(restored) => {
                    model.generate_from(&loaded.tokenizer, restored, &text, &options)?
                }
                None => model.generate(&loaded.tokenizer, &text, &options)?,
            };
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
                eprintln!("quest: logits dumped to {}", path.display());
            }
            if let Some(token) = &to {
                let path = lib::resolve_snapshot(token);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let saved = model.snapshot_caches(&path, &config.model_id, &report.context_ids)?;
                eprintln!("quest: saved {saved} context tokens to {}", path.display());
            }
            let restored_note = {
                let restored_count = report.prompt_token_count - report.prefilled_token_count;
                if restored_count > 0 {
                    format!("{restored_count} restored + ")
                } else {
                    String::new()
                }
            };
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
                "quest: prefill {}{} tokens in {:.2}s ({:.1} tok/s); decode {} tokens in {:.2}s ({:.1} tok/s){}; stopped by {}",
                restored_note,
                report.prefilled_token_count,
                report.prefill_seconds,
                report.prefilled_token_count as f64 / report.prefill_seconds.max(f64::EPSILON),
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
        Command::Snapshot { action } => match action {
            SnapshotAction::List => {
                let dir = lib::default_snapshots_dir();
                if !dir.exists() {
                    eprintln!("quest: no snapshots home yet ({})", dir.display());
                    return Ok(());
                }
                let mut rows: Vec<(String, u64)> = Vec::new();
                for entry in std::fs::read_dir(&dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().is_some_and(|e| e == "safetensors") {
                        let name = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        rows.push((name, entry.metadata()?.len()));
                    }
                }
                rows.sort();
                for (name, bytes) in &rows {
                    println!("{name} | {:.1} GiB", *bytes as f64 / (1024.0 * 1024.0 * 1024.0));
                }
                eprintln!("quest: {} saves at {}", rows.len(), dir.display());
                Ok(())
            }
            SnapshotAction::Rm { save } => {
                let path = lib::resolve_snapshot(&save);
                snafu::ensure_whatever!(
                    path.exists(),
                    "no save at {}",
                    path.display()
                );
                std::fs::remove_file(&path)?;
                eprintln!("quest: removed {}", path.display());
                Ok(())
            }
        },
        Command::Verify {
            config,
            settings,
            long,
            cross_device,
            decode_steps,
            graph,
            needle,
            needle_out,
            profile_out,
            seed,
        } => {
            let (config, settings, paths) = resolve_run(config.as_deref(), settings.as_deref())?;
            let device = lib::resolve_device(&config)?;
            let dtype = lib::resolve_dtype(&config, &device);
            if graph {
                return lib::verify_graph(&paths);
            }
            if needle {
                let opts = lib::NeedleOptions {
                    settings,
                    seed,
                    out: needle_out,
                    profile_out,
                };
                return lib::needle(&paths, &device, dtype, &opts);
            }
            let opts = lib::VerifyOptions {
                long,
                cross_device,
                decode_steps,
                settings,
            };
            lib::verify(&paths, &device, dtype, &opts)
        }
    }
}
