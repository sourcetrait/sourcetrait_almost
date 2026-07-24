pub(crate) mod capability;
pub(crate) mod checker_data;
pub(crate) mod cli;
pub(crate) mod convert;
pub(crate) mod doc;
pub(crate) mod error;
pub(crate) mod hybrid;
pub(crate) mod hybrid_load;
pub(crate) mod ifeval;
pub(crate) mod lora;
pub(crate) mod mix;
pub(crate) mod punkt;
pub(crate) mod pytext;
pub mod run;
pub(crate) mod score;
pub(crate) mod speculate;
pub(crate) mod tagger;
#[cfg(feature = "train")]
pub(crate) mod train;
pub(crate) mod wordtok;

#[allow(unused_imports)]
pub(crate) use std::{
    collections::{
        HashMap,
        HashSet,
        VecDeque,
    },
    fs,
    io,
    path::{
        Path,
        PathBuf,
    },
};

pub(crate) use sourcetrait_lib_quest_two as lib;

/// Reference backend for the burn hybrid oracle: deterministic host
/// f32, isolated from the CUDA stack under test.
#[allow(dead_code)]
pub(crate) type CpuBack = burn::backend::NdArray<f32>;
/// Fast-oracle and trainer cuda backend (bf16 default float; the
/// GDN mixer's gate/norm/recurrence region computes f32 regardless
/// via per-tensor casts - the reference stacks' f32-state
/// discipline).
#[cfg(any(feature = "burn-cuda", feature = "train-cuda"))]
#[allow(dead_code)]
pub(crate) type CudaBack = burn::backend::Cuda<burn::tensor::bf16>;

#[allow(unused_imports)]
pub(crate) use crate::hybrid::{
    AttnBlock,
    GdnBlock,
    HybridBlock,
    HybridModel,
    causal_mask,
};
#[allow(unused_imports)]
pub(crate) use crate::lora::{
    AttnAdapters,
    ConvDelta,
    GdnAdapters,
    LayerAdapters,
    LoraPair,
    ModelAdapters,
};
#[cfg(feature = "train")]
pub(crate) use crate::train::{
    train_cpt,
    train_gate_verb,
};
#[allow(unused_imports)]
pub(crate) use crate::hybrid_load::{
    HybridCheckpointConfig,
    HybridWeights,
    dump_read_f32_matrix,
    dump_read_u32,
};

pub(crate) use crate::capability::{
    CAPABILITY_HOME_RELATIVE,
    capability_bridge,
    capability_run,
    data_home,
};
pub(crate) use crate::cli::{
    CapabilityBridgeArgs,
    CapabilityCommand,
    CapabilityConvertArgs,
    CapabilityRunArgs,
    CapabilityScoreArgs,
    Cli,
    Command,
    DocCommand,
    MixCommand,
    MixPackArgs,
    MixRenderArgs,
    SpeculateCommand,
    SpeculateRecordArgs,
    SpeculateSimulateArgs,
    SpeculateTokensArgs,
    TrainCommand,
};
#[cfg(feature = "train")]
pub(crate) use crate::cli::TrainCptArgs;
pub(crate) use crate::mix::{
    SplitMix64,
    mix_pack,
    mix_render,
};
pub(crate) use crate::doc::doc_cli;
#[allow(unused_imports)]
pub(crate) use crate::convert::{
    capability_convert,
    collect_suffix_files,
    epoch_seconds,
    field,
    field_ids,
    field_int,
    field_rows,
    field_str,
    field_strings,
    json_to_value,
    span,
    task_token,
    v_bool,
    v_float,
    v_int,
    v_int_list,
    v_str,
    value_to_json,
};
#[allow(unused_imports)]
pub(crate) use crate::pytext::{
    PY_ASCII_WHITESPACE,
    PY_PUNCTUATION,
    nfkd_ascii,
    py_boundary_search,
    py_count,
    py_delete_chars,
    py_is_alnum,
    py_is_decimal,
    py_is_space,
    py_is_word,
    py_is_word_nondigit,
    py_lstrip_chars,
    py_lstrip_ws,
    py_rstrip_chars,
    py_split_ws,
    py_str_isdigit,
    py_str_islower,
    py_str_isupper,
    py_strip_chars,
    py_strip_ws,
    py_word_run_count,
    py_word_runs,
};
pub(crate) use crate::score::capability_score;
pub(crate) use crate::speculate::{
    speculate_record,
    speculate_simulate,
    speculate_tokens,
};
#[allow(unused_imports)]
pub(crate) use crate::error::{
    BquestError,
    BquestResult,
};

#[cfg(test)]
mod tests {
    mod convert;
    mod doc;
    mod hybrid;
    mod lora;
    mod mix;
    mod score;
    mod speculate;
    #[cfg(feature = "train")]
    mod train;
}
