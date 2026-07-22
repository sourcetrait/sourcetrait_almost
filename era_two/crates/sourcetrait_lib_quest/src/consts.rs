//! Checkpoint-pinned constants for the era-two hybrid pair.
#[allow(unused_imports)]
use crate::*;

/// The deployed artifact (the era's sole checkpoint).
pub const DPO_MODEL_NAME: &str = "Olmo-Hybrid-Instruct-DPO-7B";
/// The pretrained base (loader-generality and provenance runs only).
pub const BASE_MODEL_NAME: &str = "Olmo-Hybrid-7B";
/// Checkpoint author segment under the model data home.
pub const MODEL_AUTHOR: &str = "allenai";
/// The models home relative to the XDG data home (author-qualified
/// model coordinates join beneath it).
pub const MODELS_HOME_RELATIVE: &str = "huggingface/model";
/// The suite's profile root relative to the XDG config home.
pub const SUITE_CONFIG_RELATIVE: &str = "sourcetrait/quest";

/// Stop tokens resolved by string at runtime, so tokenizer truth wins
/// over any config-side id drift (the shipped generation_config only
/// carries endoftext; chat turns end on im_end).
pub const STOP_TOKENS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

// The DPO artifact's added-token ids the engine leans on (identical to
// era-one olmo3's map; the BASE checkpoint differs at 100266-100275 -
// extra_id_1..10, no tool markers).
pub const TOKEN_EXTRA_ID_0: u32 = 100256;
pub const TOKEN_ENDOFTEXT: u32 = 100257;
pub const TOKEN_IM_START: u32 = 100264;
pub const TOKEN_IM_END: u32 = 100265;
pub const TOKEN_FUNCTIONS_OPEN: u32 = 100266;
pub const TOKEN_FUNCTIONS_CLOSE: u32 = 100267;
pub const TOKEN_FUNCTION_CALLS_OPEN: u32 = 100268;
pub const TOKEN_FUNCTION_CALLS_CLOSE: u32 = 100269;
pub const TOKEN_EXTRA_ID_1: u32 = 100270;
pub const TOKEN_EXTRA_ID_6: u32 = 100275;
/// The insufficiency tag (special: true - detect by id, never by text).
pub const TOKEN_ENDOFPROMPT: u32 = 100276;
pub const TOKEN_PAD: u32 = 100277;
