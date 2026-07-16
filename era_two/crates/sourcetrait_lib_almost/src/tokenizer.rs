//! Tokenizer loading and the added-token map lock.
use crate::*;

/// Load a checkpoint dir's tokenizer.json.
pub fn load_tokenizer(model_dir: &Path) -> LibAlmostResult<tokenizers::Tokenizer> {
    match tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")) {
        Ok(tokenizer) => Ok(tokenizer),
        Err(e) => snafu::whatever!("tokenizer load failed: {e}"),
    }
}

/// Verify the live tokenizer carries the DPO artifact's channel/tool
/// token ids the engine leans on; drift is a hard error. (The BASE
/// checkpoint intentionally fails this - its 100266-100275 range holds
/// extra_id_1..10 instead of the tool markers.)
pub fn verify_token_map(tokenizer: &tokenizers::Tokenizer) -> LibAlmostResult<()> {
    let expectations: [(&str, u32); 12] = [
        ("<|extra_id_0|>", consts::TOKEN_EXTRA_ID_0),
        ("<|endoftext|>", consts::TOKEN_ENDOFTEXT),
        ("<|im_start|>", consts::TOKEN_IM_START),
        ("<|im_end|>", consts::TOKEN_IM_END),
        ("<functions>", consts::TOKEN_FUNCTIONS_OPEN),
        ("</functions>", consts::TOKEN_FUNCTIONS_CLOSE),
        ("<function_calls>", consts::TOKEN_FUNCTION_CALLS_OPEN),
        ("</function_calls>", consts::TOKEN_FUNCTION_CALLS_CLOSE),
        ("<|extra_id_1|>", consts::TOKEN_EXTRA_ID_1),
        ("<|extra_id_6|>", consts::TOKEN_EXTRA_ID_6),
        ("<|endofprompt|>", consts::TOKEN_ENDOFPROMPT),
        ("<|pad|>", consts::TOKEN_PAD),
    ];
    for (token, expected) in expectations {
        match tokenizer.token_to_id(token) {
            Some(actual) if actual == expected => {}
            other => snafu::whatever!(
                "added-token map drift: {token} resolves to {other:?}, expected {expected}"
            ),
        }
    }
    Ok(())
}
