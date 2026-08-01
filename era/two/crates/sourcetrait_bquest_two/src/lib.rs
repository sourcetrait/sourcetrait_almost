pub(crate) mod capability;
pub(crate) mod checker_data;
pub(crate) mod cli;
pub(crate) mod convert;
pub(crate) mod doc;
pub(crate) mod error;
pub(crate) mod example;
pub(crate) mod ifeval;
pub(crate) mod mix;
pub(crate) mod nu_sandbox;
pub(crate) mod punkt;
pub(crate) mod pytext;
pub(crate) mod rollout;
pub mod run;
pub(crate) mod score;
pub(crate) mod taskgen;
pub(crate) mod speculate;
pub(crate) mod syllabus;
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

pub(crate) use sourcetrait_lib_quest_harness_two as harness;
pub(crate) use sourcetrait_lib_quest_llm_two as llm;

/// The oracle backends and the seeded rng, from the llm lib.
#[allow(unused_imports)]
pub(crate) use llm::{
    CpuBack,
    SplitMix64,
};
#[cfg(any(feature = "burn-cuda", feature = "train-cuda"))]
#[allow(unused_imports)]
pub(crate) use llm::CudaBack;

/// The oracle surface the parity and diag verbs drive.
#[allow(unused_imports)]
pub(crate) use llm::{
    HybridCheckpointConfig,
    HybridModel,
    HybridWeights,
    dump_read_f32_matrix,
    dump_read_u32,
};
#[cfg(feature = "train")]
pub(crate) use llm::ModelAdapters;

#[cfg(feature = "train")]
pub(crate) use crate::train::{
    train_cpt,
    train_dpo,
    train_gate_verb,
    train_rlvr,
    train_sft,
};
pub(crate) use crate::example::{
    mix_instruct,
    mix_tokens,
};
pub(crate) use crate::rollout::{
    bench_run,
    rollout_run,
};
pub(crate) use crate::taskgen::taskgen_all;

pub(crate) use crate::capability::{
    CAPABILITY_HOME_RELATIVE,
    capability_bridge,
    capability_run,
    data_home,
};
pub(crate) use crate::cli::{
    BenchCommand,
    BenchRunArgs,
    CapabilityBridgeArgs,
    CapabilityCommand,
    CapabilityConvertArgs,
    CapabilityRunArgs,
    CapabilityScoreArgs,
    Cli,
    Command,
    DocCommand,
    MixCommand,
    MixInstructArgs,
    MixPackArgs,
    MixRenderArgs,
    MixRipArgs,
    MixSampleArgs,
    MixTokensArgs,
    RolloutCommand,
    RolloutRunArgs,
    SpeculateCommand,
    TaskgenAllArgs,
    TaskgenCommand,
    SpeculateRecordArgs,
    SpeculateSimulateArgs,
    SpeculateTokensArgs,
    SyllabusCommand,
    SyllabusEmitArgs,
    TrainCommand,
};
#[cfg(feature = "train")]
pub(crate) use crate::cli::{
    StageArgs,
    TrainCptArgs,
    TrainDpoArgs,
    TrainRlvrArgs,
    TrainSftArgs,
};
pub(crate) use crate::mix::{
    mix_pack,
    mix_render,
    mix_rip,
    mix_sample,
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
pub(crate) use crate::syllabus::syllabus_emit;
#[allow(unused_imports)]
pub(crate) use crate::error::{
    BquestError,
    BquestResult,
};

#[cfg(test)]
mod tests {
    mod convert;
    mod doc;
    mod mix;
    mod score;
    mod speculate;
    mod taskgen;
}
