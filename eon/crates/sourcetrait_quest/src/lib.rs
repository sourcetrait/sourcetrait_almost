pub(crate) mod converse;
pub(crate) mod error;
pub(crate) mod use_harness;
pub(crate) mod plugin;
pub(crate) mod prompt;
pub mod run;

pub(crate) use std::{
    fs,
    path::PathBuf,
};

pub(crate) use sourcetrait_lib_quest_harness_two as harness_lib;
pub(crate) use sourcetrait_quest_bridge as bridge;

pub(crate) use crate::error::{
    QuestPluginError,
    QuestPluginResult,
};
pub(crate) use crate::plugin::QuestPlugin;

#[cfg(test)]
mod tests {
    mod error;
    mod prompt;
}
