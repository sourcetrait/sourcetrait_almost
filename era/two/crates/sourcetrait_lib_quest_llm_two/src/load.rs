//! Weight loading: safetensors header inventory and mmap VarBuilder.
use crate::*;

/// One tensor's header entry: name, dtype, shape, payload bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_len: u64,
}

/// Parse the shard's JSON header, name-sorted; no tensor data is read.
pub fn tensor_inventory(model_dir: &Path) -> LibQuestResult<Vec<TensorInfo>> {
    let mut file = fs::File::open(model_dir.join("model.safetensors"))?;
    let mut len_bytes = [0u8; 8];
    io::Read::read_exact(&mut file, &mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > 64 * 1024 * 1024 {
        snafu::whatever!("implausible safetensors header length {header_len}");
    }
    let mut buf = vec![0u8; header_len as usize];
    io::Read::read_exact(&mut file, &mut buf)?;
    let header: serde_json::Value = serde_json::from_slice(&buf)?;
    let Some(map) = header.as_object() else {
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
        if offsets.len() != 2 {
            snafu::whatever!("malformed data_offsets for tensor {name}");
        }
        let (Some(begin), Some(end)) = (offsets[0].as_u64(), offsets[1].as_u64()) else {
            snafu::whatever!("malformed data_offsets for tensor {name}");
        };
        tensors.push(TensorInfo {
            name: name.clone(),
            dtype: dtype.to_string(),
            shape: shape.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect(),
            byte_len: end - begin,
        });
    }
    Ok(tensors)
}

/// Mmap the shard into a VarBuilder; tensors materialize per get().
pub fn mmap_weights(
    model_dir: &Path,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> LibQuestResult<candle_nn::VarBuilder<'static>> {
    let shard = model_dir.join("model.safetensors");
    Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[shard], dtype, device)? })
}
