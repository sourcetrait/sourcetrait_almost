//! Local checkpoint resolution and safetensors-header ground truth.
#![allow(dead_code)]
use crate::*;

pub(crate) const DPO_MODEL_NAME: &str = "Olmo-Hybrid-Instruct-DPO-7B";
pub(crate) const BASE_MODEL_NAME: &str = "Olmo-Hybrid-7B";

/// Which local hybrid checkpoint a verb drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ModelPick {
    /// The deployed artifact (Olmo-Hybrid-Instruct-DPO-7B).
    Dpo,
    /// The pretrained base (Olmo-Hybrid-7B).
    Base,
}

impl ModelPick {
    pub(crate) fn model_name(self) -> &'static str {
        match self {
            ModelPick::Dpo => DPO_MODEL_NAME,
            ModelPick::Base => BASE_MODEL_NAME,
        }
    }
}

/// The XDG data home, honouring the spec's own fallback.
pub(crate) fn data_home() -> RefquestResult<PathBuf> {
    if let Ok(dir) = env::var("XDG_DATA_HOME")
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    let Ok(home) = env::var("HOME") else {
        snafu::whatever!("neither XDG_DATA_HOME nor HOME is set");
    };
    Ok(PathBuf::from(home).join(".local/share"))
}

/// The local directory a checkpoint lives in (the XDG data-home layout).
pub(crate) fn resolve_model_dir(pick: ModelPick) -> RefquestResult<PathBuf> {
    Ok(data_home()?
        .join("huggingface/model/allenai")
        .join(pick.model_name()))
}

/// One tensor's header entry: name, dtype, shape, byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorInfo {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) data_offsets: [u64; 2],
}

impl TensorInfo {
    pub(crate) fn byte_len(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }
}

/// Parse a safetensors file's JSON header; never reads tensor data.
pub(crate) fn read_safetensors_header(path: &Path) -> RefquestResult<Vec<TensorInfo>> {
    let mut file = fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > 64 * 1024 * 1024 {
        snafu::whatever!("implausible safetensors header length {header_len}");
    }
    let mut buf = vec![0u8; header_len as usize];
    file.read_exact(&mut buf)?;
    let value: serde_json::Value = serde_json::from_slice(&buf)?;
    let Some(map) = value.as_object() else {
        snafu::whatever!("safetensors header is not a JSON object");
    };

    let mut tensors = Vec::with_capacity(map.len());
    for (name, entry) in map {
        if name == "__metadata__" {
            continue;
        }
        let (Some(dtype), Some(shape), Some(offsets)) = (
            entry["dtype"].as_str(),
            entry["shape"].as_array(),
            entry["data_offsets"].as_array(),
        ) else {
            snafu::whatever!("malformed header entry for tensor {name}");
        };
        let shape: Vec<u64> = shape.iter().filter_map(|v| v.as_u64()).collect();
        let (Some(begin), Some(end)) = (offsets[0].as_u64(), offsets[1].as_u64()) else {
            snafu::whatever!("malformed data_offsets for tensor {name}");
        };
        tensors.push(TensorInfo {
            name: name.clone(),
            dtype: dtype.to_string(),
            shape,
            data_offsets: [begin, end],
        });
    }
    Ok(tensors)
}

/// A checkpoint's config.json as a JSON value (pin-lock reads).
pub(crate) fn read_config(model_dir: &Path) -> RefquestResult<serde_json::Value> {
    Ok(serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?)
}

/// A checkpoint's added-token table from tokenizer.json, id-sorted.
pub(crate) fn read_added_tokens(model_dir: &Path) -> RefquestResult<Vec<(u64, String, bool)>> {
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(model_dir.join("tokenizer.json"))?)?;
    let Some(entries) = value["added_tokens"].as_array() else {
        snafu::whatever!("tokenizer.json carries no added_tokens array");
    };
    let mut added = Vec::with_capacity(entries.len());
    for entry in entries {
        let (Some(id), Some(content), Some(special)) = (
            entry["id"].as_u64(),
            entry["content"].as_str(),
            entry["special"].as_bool(),
        ) else {
            snafu::whatever!("malformed added_tokens entry");
        };
        added.push((id, content.to_string(), special));
    }
    added.sort_by_key(|(id, _, _)| *id);
    Ok(added)
}
