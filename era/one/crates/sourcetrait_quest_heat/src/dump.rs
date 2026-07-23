use crate::*;

/// Which burn backend replays the ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OracleDevice {
    /// ndarray f32 - deterministic, fully isolated from the CUDA stack;
    /// the reference grade.
    Cpu,
    /// bf16 on the CUDA backend - fast, comparison-grade; shares the
    /// driver/toolkit chain with the system under test.
    Cuda,
}

/// Replay the token ids from a refquest parity dump through the burn model
/// and write a dump in the same format for diffing.
pub(crate) fn dump(model_dir: &Path, ids_from: &Path, out: &Path, device: OracleDevice) -> HeatResult<()> {
    let source_bytes = std::fs::read(ids_from)?;
    let source = safetensors::SafeTensors::deserialize(&source_bytes)?;
    let prompt_ids = tensor_io::read_u32(&source, "prompt_ids")?;
    let fed_ids = tensor_io::read_u32(&source, "fed_ids")?;
    let mut full_ids = prompt_ids.clone();
    full_ids.extend_from_slice(&fed_ids);
    eprintln!(
        "heat: replaying {} ids ({} prompt + {} fed) on {device:?}",
        full_ids.len(),
        prompt_ids.len(),
        fed_ids.len()
    );

    let config: HeatConfig =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let shards = shard_paths(model_dir)?;
    let load_start = Instant::now();
    let weights = load::Weights::load(&shards)?;
    eprintln!(
        "heat: weights read + converted to host f32 in {:.1}s",
        load_start.elapsed().as_secs_f32()
    );

    let (rows, logits) = match device {
        OracleDevice::Cpu => {
            forward_on::<CpuBack>(&config, weights, Default::default(), &full_ids)?
        }
        OracleDevice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                forward_on::<CudaBack>(&config, weights, Default::default(), &full_ids)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                snafu::whatever!("this heat build carries no cuda support (rebuild with --features cuda)");
            }
        }
    };

    let payload = tensor_io::DumpPayload::new(rows, config.vocab_size, &logits, &prompt_ids, &fed_ids);
    payload.write(out)?;
    eprintln!("heat: dump written to {}", out.display());
    Ok(())
}

fn forward_on<B: burn::tensor::backend::Backend>(
    config: &HeatConfig,
    weights: Weights,
    device: B::Device,
    ids: &[u32],
) -> HeatResult<(usize, Vec<f32>)> {
    let build_start = Instant::now();
    let model = HeatModel::<B>::new(config, weights, device)?;
    eprintln!("heat: model built in {:.1}s", build_start.elapsed().as_secs_f32());
    let forward_start = Instant::now();
    let result = model.forward_all(ids)?;
    eprintln!(
        "heat: forward over {} positions in {:.1}s",
        result.0,
        forward_start.elapsed().as_secs_f32()
    );
    Ok(result)
}

/// Shard list from the model dir's safetensors index.
pub(crate) fn shard_paths(model_dir: &Path) -> HeatResult<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let index: serde_json::Value = serde_json::from_reader(std::fs::File::open(&index_path)?)?;
    let Some(weight_map) = index.get("weight_map").and_then(|value| value.as_object()) else {
        snafu::whatever!("model.safetensors.index.json carries no weight_map");
    };
    let names: std::collections::BTreeSet<String> = weight_map
        .values()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect();
    Ok(names.into_iter().map(|name| model_dir.join(name)).collect())
}
