//! Pinned constants: the training scheme's version.
#[allow(unused_imports)]
use crate::*;

/// The training scheme's version, pinned in the manifest metadata.
pub const TRAINING_VERSION: &str = env!("QUEST_TRAINING_VERSION");
