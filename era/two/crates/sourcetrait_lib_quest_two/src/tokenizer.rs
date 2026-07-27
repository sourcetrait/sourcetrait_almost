//! Tokenizer loading and the added-token map lock.
use crate::*;

/// Load a checkpoint dir's tokenizer.json.
pub fn load_tokenizer(model_dir: &Path) -> LibQuestResult<tokenizers::Tokenizer> {
    match tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")) {
        Ok(tokenizer) => Ok(tokenizer),
        Err(e) => snafu::whatever!("tokenizer load failed: {e}"),
    }
}

/// Lock the added-token ids the engine and the channel grammar lean on.
pub fn verify_token_map(tokenizer: &tokenizers::Tokenizer) -> LibQuestResult<()> {
    let expectations: [(&str, u32); 9] = [
        ("<|endoftext|>", consts::TOKEN_ENDOFTEXT),
        ("<|im_start|>", consts::TOKEN_IM_START),
        ("<|im_end|>", consts::TOKEN_IM_END),
        ("<functions>", consts::TOKEN_FUNCTIONS_OPEN),
        ("</functions>", consts::TOKEN_FUNCTIONS_CLOSE),
        ("<function_calls>", consts::TOKEN_FUNCTION_CALLS_OPEN),
        ("</function_calls>", consts::TOKEN_FUNCTION_CALLS_CLOSE),
        ("<|endofprompt|>", consts::TOKEN_ENDOFPROMPT),
        ("<|pad|>", consts::TOKEN_PAD),
    ];
    let channel = channel::TAGS.map(|tag| (tag.spelling(), tag.token_id()));
    for (token, expected) in expectations.into_iter().chain(channel) {
        match tokenizer.token_to_id(token) {
            Some(actual) if actual == expected => {}
            other => snafu::whatever!(
                "added-token map drift: {token} resolves to {other:?}, expected {expected}"
            ),
        }
    }
    Ok(())
}
