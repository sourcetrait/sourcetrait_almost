use crate::*;

/// Local paths of a complete pulled checkpoint.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub shards: Vec<PathBuf>,
}

/// $XDG_CACHE_HOME, with ~/.cache as the fallback when the variable is
/// unset or empty (the XDG spec's own default).
fn cache_home() -> PathBuf {
    match std::env::var("XDG_CACHE_HOME") {
        Ok(cache_home) if !cache_home.is_empty() => PathBuf::from(cache_home),
        _ => {
            let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
            PathBuf::from(home).join(".cache")
        }
    }
}

/// $XDG_CACHE_HOME/huggingface/model/<owner>--<name> (cache-home
/// fallback rules above).
pub fn default_model_dir(model_id: &str) -> PathBuf {
    cache_home()
        .join("huggingface")
        .join("model")
        .join(model_id.replace('/', "--"))
}

/// E2 saved-context home: $XDG_CACHE_HOME/sourcetrait/almost/snapshots
/// (snapshots are regenerable caches).
pub fn default_snapshots_dir() -> PathBuf {
    cache_home()
        .join("sourcetrait")
        .join("almost")
        .join("snapshots")
}

/// A --from/--to token that is purely a snake names a save under the
/// snapshots home; anything else is a filesystem path (~ expanded).
pub fn resolve_snapshot(token: &str) -> PathBuf {
    if config::is_snake(token) {
        default_snapshots_dir().join(format!("{token}.safetensors"))
    } else {
        config::expand_path(token)
    }
}

/// Ensure every checkpoint file exists under dir, downloading what is
/// missing (files already present are trusted; delete a file to re-pull it).
pub fn ensure_model(model_id: &str, dir: &Path) -> AlmostResult<ModelPaths> {
    fn fetch(
        repo: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
        dir: &Path,
        filename: &str,
    ) -> AlmostResult<PathBuf> {
        let target = dir.join(filename);
        if target.exists() {
            return Ok(target);
        }
        eprintln!("almost: pulling {filename}");
        let path = repo
            .download_file()
            .filename(filename)
            .local_dir(dir.to_path_buf())
            .send()?;
        Ok(path)
    }

    let Some((owner, name)) = model_id.split_once('/') else {
        snafu::whatever!("model id must be <owner>/<name>");
    };
    std::fs::create_dir_all(dir)?;
    let client = hf_hub::HFClientSync::new()?;
    let repo = client.model(owner, name);

    let config = fetch(&repo, dir, "config.json")?;
    let tokenizer = fetch(&repo, dir, "tokenizer.json")?;
    fetch(&repo, dir, "generation_config.json")?;
    let index_path = fetch(&repo, dir, "model.safetensors.index.json")?;

    let index: serde_json::Value = serde_json::from_reader(std::fs::File::open(&index_path)?)?;
    let Some(weight_map) = index.get("weight_map").and_then(|value| value.as_object()) else {
        snafu::whatever!("model.safetensors.index.json carries no weight_map");
    };
    let shard_names: BTreeSet<String> = weight_map
        .values()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect();
    let mut shards = Vec::with_capacity(shard_names.len());
    for shard_name in &shard_names {
        shards.push(fetch(&repo, dir, shard_name)?);
    }

    Ok(ModelPaths {
        config,
        tokenizer,
        shards,
    })
}
