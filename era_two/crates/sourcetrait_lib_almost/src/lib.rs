pub(crate) mod chat;
pub(crate) mod checkpoint;
pub mod consts;
pub(crate) mod error;
pub(crate) mod load;
pub(crate) mod tokenizer;

#[allow(unused_imports)]
pub(crate) use std::{
    env,
    fs,
    io,
    path::{
        Path,
        PathBuf,
    },
};

pub use crate::chat::{
    chat_continue,
    chat_wrap,
    resolve_stop_ids,
};
pub use crate::checkpoint::{
    LayerKind,
    OlmoHybridConfig,
    RopeParameters,
    load_config,
    model_dir,
};
pub use crate::error::{
    LibAlmostError,
    LibAlmostResult,
};
pub use crate::load::{
    TensorInfo,
    mmap_weights,
    tensor_inventory,
};
pub use crate::tokenizer::{
    load_tokenizer,
    verify_token_map,
};

#[cfg(test)]
mod tests {
    mod chat;
    mod checkpoint;
    mod load;
    mod tokenizer;
}
