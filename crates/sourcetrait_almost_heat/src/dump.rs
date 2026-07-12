use crate::*;

/// Replay the token ids from an lmst parity dump through the burn model
/// and write a dump in the same format for diffing.
pub(crate) fn dump(model_dir: &Path, ids_from: &Path, out: &Path) -> HeatResult<()> {
    let source_bytes = std::fs::read(ids_from)?;
    let source = safetensors::SafeTensors::deserialize(&source_bytes)?;
    let prompt_ids = tensor_io::read_u32(&source, "prompt_ids")?;
    let fed_ids = tensor_io::read_u32(&source, "fed_ids")?;
    let mut full_ids = prompt_ids.clone();
    full_ids.extend_from_slice(&fed_ids);
    eprintln!(
        "heat: replaying {} ids ({} prompt + {} fed)",
        full_ids.len(),
        prompt_ids.len(),
        fed_ids.len()
    );

    let config: HeatConfig =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let shards = shard_paths(model_dir)?;
    let load_start = Instant::now();
    let weights = load::Weights::load(&shards)?;
    let model = HeatModel::new(&config, weights)?;
    eprintln!(
        "heat: weights loaded + converted to f32 in {:.1}s",
        load_start.elapsed().as_secs_f32()
    );

    let forward_start = Instant::now();
    let (rows, logits) = model.forward_all(&full_ids)?;
    eprintln!(
        "heat: forward over {} positions in {:.1}s",
        rows,
        forward_start.elapsed().as_secs_f32()
    );

    let payload = tensor_io::DumpPayload::new(rows, config.vocab_size, &logits, &prompt_ids, &fed_ids);
    payload.write(out)?;
    eprintln!("heat: dump written to {}", out.display());
    Ok(())
}

/// Shard list from the model dir's safetensors index.
fn shard_paths(model_dir: &Path) -> HeatResult<Vec<PathBuf>> {
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
