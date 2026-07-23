use crate::*;

/// A ready-to-run checkpoint: parsed config, tokenizer, and the model with
/// weights mounted.
pub struct LoadedModel {
    pub config: Olmo3Config,
    pub tokenizer: tokenizers::Tokenizer,
    pub model: Model,
    pub load_seconds: f64,
}

/// Parse config.json + tokenizer.json and mmap the shards into a Model on
/// the given device/dtype.
pub fn load_model(
    paths: &ModelPaths,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    settings: Settings,
) -> QuestResult<LoadedModel> {
    let config: Olmo3Config = serde_json::from_reader(std::fs::File::open(&paths.config)?)?;
    let tokenizer = match tokenizers::Tokenizer::from_file(&paths.tokenizer) {
        Ok(tokenizer) => tokenizer,
        Err(error) => snafu::whatever!("loading tokenizer.json failed: {error}"),
    };
    let load_start = Instant::now();
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths.shards, dtype, device)? };
    let model = Model::new(&config, settings, vb)?;
    Ok(LoadedModel {
        config,
        tokenizer,
        model,
        load_seconds: load_start.elapsed().as_secs_f64(),
    })
}

/// CUDA when available, unless forced off (the --cpu semantics).
pub fn pick_device(force_cpu: bool) -> QuestResult<candle_core::Device> {
    if force_cpu {
        return Ok(candle_core::Device::Cpu);
    }
    if r::candle::cuda_is_available() {
        Ok(candle_core::Device::new_cuda(0)?)
    } else {
        Ok(candle_core::Device::Cpu)
    }
}

/// bf16 or f32 when named; defaults to bf16 on cuda and f32 on cpu.
pub fn pick_dtype(flag: Option<&str>, device: &candle_core::Device) -> QuestResult<candle_core::DType> {
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
