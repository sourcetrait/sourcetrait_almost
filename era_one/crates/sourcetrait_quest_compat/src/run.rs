use crate::*;

/// Binary entry: parse, dispatch, exit non-zero on error.
pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    if let Err(error) = dispatch(cli) {
        eprintln!("refquest: {error}");
        std::process::exit(1);
    }
}

fn dispatch(cli: Cli) -> CompatResult<()> {
    match cli.command {
        Command::Pull { model_id, model_dir } => {
            let dir = model_dir.unwrap_or_else(|| default_model_dir(&model_id));
            let paths = ensure_model(&model_id, &dir)?;
            eprintln!(
                "refquest: model ready at {} ({} shards)",
                dir.display(),
                paths.shards.len()
            );
            Ok(())
        }
        Command::Prompt {
            prompt,
            raw,
            greedy,
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
            let dir = model_dir.unwrap_or_else(|| default_model_dir(&model_id));
            let paths = ensure_model(&model_id, &dir)?;
            let device = pick_device(cpu)?;
            let dtype = pick_dtype(dtype.as_deref(), &device)?;
            eprintln!("refquest: device {device:?}, dtype {dtype:?}");

            let config: Olmo3Config = serde_json::from_reader(std::fs::File::open(&paths.config)?)?;
            let tokenizer = match tokenizers::Tokenizer::from_file(&paths.tokenizer) {
                Ok(tokenizer) => tokenizer,
                Err(error) => snafu::whatever!("loading tokenizer.json failed: {error}"),
            };
            let load_start = Instant::now();
            let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths.shards, dtype, &device)? };
            let mut model = Model::new(&config, vb)?;
            eprintln!("refquest: weights loaded in {:.1}s", load_start.elapsed().as_secs_f32());

            let text = if raw { prompt } else { chat_wrap(&prompt) };
            let opts = GenerateOptions {
                greedy,
                temperature,
                top_p,
                sample_len,
                seed,
                dump_logits,
            };
            generate(&mut model, &tokenizer, &text, &opts, &device)
        }
        Command::Verify {
            long,
            cross_device,
            decode_steps,
            cpu,
            dtype,
            model_id,
            model_dir,
        } => {
            let dir = model_dir.unwrap_or_else(|| default_model_dir(&model_id));
            let paths = ensure_model(&model_id, &dir)?;
            let device = pick_device(cpu)?;
            let dtype = pick_dtype(dtype.as_deref(), &device)?;
            let opts = VerifyOptions {
                long,
                cross_device,
                decode_steps,
            };
            verify::verify(&paths, &device, dtype, &opts)
        }
    }
}

fn pick_device(force_cpu: bool) -> CompatResult<candle_core::Device> {
    if force_cpu {
        return Ok(candle_core::Device::Cpu);
    }
    if r::candle::cuda_is_available() {
        Ok(candle_core::Device::new_cuda(0)?)
    } else {
        eprintln!("refquest: cuda unavailable, running on cpu");
        Ok(candle_core::Device::Cpu)
    }
}

fn pick_dtype(flag: Option<&str>, device: &candle_core::Device) -> CompatResult<candle_core::DType> {
    match flag {
        Some("bf16") => Ok(candle_core::DType::BF16),
        Some("f32") => Ok(candle_core::DType::F32),
        Some(other) => snafu::whatever!("unsupported dtype {}; use bf16 or f32", other),
        None => {
            if matches!(device, candle_core::Device::Cpu) {
                Ok(candle_core::DType::F32)
            } else {
                Ok(candle_core::DType::BF16)
            }
        }
    }
}
