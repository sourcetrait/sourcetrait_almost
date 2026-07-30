//! The saved-context file format and its three-tier token resolution.
use crate::*;

/// The era-two snapshot format version (metadata "version").
pub(crate) const SNAPSHOT_VERSION: &str = "1";

/// What restore_caches hands back: the live length and the id trail.
pub struct RestoredContext {
    pub context_len: usize,
    pub context_ids: Vec<u32>,
}

/// A bare relative token: no leading `/`, `./`, `../`, `~`, `$`.
fn is_bare_relative(token: &str) -> bool {
    !(token.starts_with('/')
        || token.starts_with("./")
        || token.starts_with("../")
        || token.starts_with('~')
        || token.starts_with('$'))
}

/// The three-tier snapshot token resolution.
pub fn snapshot_path(snapshots_dir: &Path, token: &str) -> LibQuestResult<PathBuf> {
    if config::is_profile_name(Path::new(token)) {
        return Ok(snapshots_dir.join(format!("{token}.safetensors")));
    }
    if is_bare_relative(token) {
        return Ok(snapshots_dir.join(token));
    }
    config::expand_path(token)
}

/// A parsed, format-validated snapshot ready for a model to consume.
pub(crate) struct SnapshotFile {
    tensors: HashMap<String, candle_core::Tensor>,
    pub(crate) context_ids: Vec<u32>,
    pub(crate) context_len: usize,
}

impl SnapshotFile {
    pub(crate) fn tensor(&self, name: &str) -> LibQuestResult<&candle_core::Tensor> {
        match self.tensors.get(name) {
            Some(tensor) => Ok(tensor),
            None => snafu::whatever!("snapshot is missing tensor {name}"),
        }
    }
}

/// Write one snapshot file: the cache tensors plus the id trail.
pub(crate) fn write_snapshot(
    path: &Path,
    tensors: Vec<(String, candle_core::Tensor)>,
    model_id: &str,
    context_len: usize,
    context_ids: &[u32],
) -> LibQuestResult<()> {
    snafu::ensure_whatever!(
        context_ids.len() >= context_len,
        "the id trail ({}) is shorter than the context length {context_len}",
        context_ids.len()
    );
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let ids = candle_core::Tensor::from_vec(
        context_ids.to_vec(),
        context_ids.len(),
        &candle_core::Device::Cpu,
    )?;
    let mut data = tensors;
    data.push(("context_ids".to_string(), ids));
    let mut info: HashMap<String, String> = HashMap::new();
    info.insert("version".to_string(), SNAPSHOT_VERSION.to_string());
    info.insert("model_id".to_string(), model_id.to_string());
    info.insert("context_len".to_string(), context_len.to_string());
    match safetensors::serialize_to_file(data, Some(info), path) {
        Ok(()) => Ok(()),
        Err(e) => snafu::whatever!("snapshot write failed ({}): {e}", path.display()),
    }
}

/// Read and format-validate one snapshot file onto `device`.
pub(crate) fn read_snapshot(
    path: &Path,
    expected_model_id: &str,
    device: &candle_core::Device,
) -> LibQuestResult<SnapshotFile> {
    use candle_core::safetensors::Load;

    let bytes = fs::read(path)?;
    let (_, metadata) = match safetensors::SafeTensors::read_metadata(&bytes) {
        Ok(pair) => pair,
        Err(e) => snafu::whatever!("snapshot header parse failed ({}): {e}", path.display()),
    };
    let Some(info) = metadata.metadata() else {
        snafu::whatever!("snapshot carries no metadata ({})", path.display());
    };
    let field = |key: &str| -> LibQuestResult<&String> {
        match info.get(key) {
            Some(value) => Ok(value),
            None => snafu::whatever!("snapshot metadata is missing {key}"),
        }
    };
    let version = field("version")?;
    snafu::ensure_whatever!(
        version == SNAPSHOT_VERSION,
        "snapshot version {version} is not the supported {SNAPSHOT_VERSION}"
    );
    let model_id = field("model_id")?;
    snafu::ensure_whatever!(
        model_id == expected_model_id,
        "snapshot belongs to model {model_id}, expected {expected_model_id}"
    );
    let context_len: usize = match field("context_len")?.parse() {
        Ok(len) => len,
        Err(e) => snafu::whatever!("snapshot context_len does not parse: {e}"),
    };

    let parsed = match safetensors::SafeTensors::deserialize(&bytes) {
        Ok(parsed) => parsed,
        Err(e) => snafu::whatever!("snapshot parse failed ({}): {e}", path.display()),
    };
    let mut tensors = HashMap::new();
    let mut context_ids: Option<Vec<u32>> = None;
    for (name, view) in parsed.tensors() {
        if name == "context_ids" {
            context_ids = Some(
                view.load(&candle_core::Device::Cpu)?
                    .to_vec1::<u32>()?,
            );
        } else {
            tensors.insert(name, view.load(device)?);
        }
    }
    let Some(context_ids) = context_ids else {
        snafu::whatever!("snapshot carries no context_ids trail");
    };
    snafu::ensure_whatever!(
        context_ids.len() >= context_len,
        "the id trail ({}) is shorter than the context length {context_len}",
        context_ids.len()
    );
    Ok(SnapshotFile {
        tensors,
        context_ids,
        context_len,
    })
}
