pub(crate) mod assembler;
pub(crate) mod associations;
pub(crate) mod bucket;
pub(crate) mod cli;
pub(crate) mod corpus;
pub(crate) mod dictionary;
pub(crate) mod doc;
pub(crate) mod error;
pub(crate) mod ledger;
pub(crate) mod lexer;
pub(crate) mod matrix;
pub mod run;
pub(crate) mod trainer;
pub(crate) mod ucd;
pub(crate) mod value;
pub(crate) mod wikicorpus;
pub(crate) mod wikidoc;
pub(crate) mod wikivocab;
pub(crate) mod wikimedia;
pub(crate) mod wikitext;
pub(crate) mod wikixml;

#[allow(unused_imports)]
pub(crate) use std::{
    collections::{
        HashMap,
        HashSet,
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

pub(crate) use crate::assembler::{
    assemble_text,
    assembler_table,
    disassemble_wire,
};
pub(crate) use crate::associations::associations_build;
pub(crate) use crate::cli::{
    AssembleArgs,
    AssemblerArgs,
    AssociationsBuildArgs,
    AssociationsCommand,
    Cli,
    Command,
    DisassembleArgs,
    DocCommand,
    MatrixBuildArgs,
    MatrixCommand,
    TokenizeArgs,
    TokenizerCommand,
    TokenizerDictionaryArgs,
    TokenizerLedgerArgs,
    TrainerCommand,
    TrainerInitArgs,
    TrainerTrainArgs,
    WikimediaCommand,
    WikimediaCorpusArgs,
    WikimediaDocumentArgs,
    WikimediaNormalizeArgs,
    WikimediaPageArgs,
    WikimediaParseArgs,
};
pub(crate) use crate::dictionary::tokenizer_dictionary;
pub(crate) use crate::doc::doc_cli;
pub(crate) use crate::ledger::{
    tokenizer_ledger,
    tokenizer_ucd,
};
pub(crate) use crate::lexer::tokenize_text;
pub(crate) use crate::matrix::matrix_build;
pub(crate) use crate::trainer::{
    trainer_init,
    trainer_train,
};
pub(crate) use crate::wikicorpus::wikimedia_corpus;
pub(crate) use crate::wikimedia::{
    wikimedia_document,
    wikimedia_normalize,
    wikimedia_page,
    wikimedia_parse,
};
pub(crate) use crate::wikivocab::{
    indefinite_article,
    language_table,
    lcfirst,
    place_table,
    serial_join,
    tag_table,
    term_modifiers,
    ucfirst,
};
#[allow(unused_imports)]
pub(crate) use crate::value::{
    epoch_seconds,
    field,
    field_int,
    field_str,
    json_to_value,
    span,
    v_bool,
    v_float,
    v_int,
    v_str,
    value_to_json,
};
#[allow(unused_imports)]
pub(crate) use crate::error::{
    BiquestError,
    BiquestResult,
};

#[cfg(test)]
mod tests {
    mod assembler;
    mod associations;
    mod bucket;
    mod doc;
    mod lexer;
    mod matrix;
    mod trainer;
    mod value;
    mod wikicorpus;
    mod wikidoc;
    mod wikitext;
    mod wikivocab;
    mod wikixml;
}
