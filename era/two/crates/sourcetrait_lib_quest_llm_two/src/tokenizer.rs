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
    let extra_ids: [(&str, u32); 7] = [
        ("<|extra_id_0|>", consts::TOKEN_EXTRA_ID_0),
        ("<|extra_id_1|>", consts::TOKEN_EXTRA_ID_1),
        ("<|extra_id_2|>", consts::TOKEN_EXTRA_ID_1 + 1),
        ("<|extra_id_3|>", consts::TOKEN_EXTRA_ID_1 + 2),
        ("<|extra_id_4|>", consts::TOKEN_EXTRA_ID_1 + 3),
        ("<|extra_id_5|>", consts::TOKEN_EXTRA_ID_1 + 4),
        ("<|extra_id_6|>", consts::TOKEN_EXTRA_ID_6),
    ];
    for (token, expected) in expectations.into_iter().chain(extra_ids) {
        match tokenizer.token_to_id(token) {
            Some(actual) if actual == expected => {}
            other => snafu::whatever!(
                "added-token map drift: {token} resolves to {other:?}, expected {expected}"
            ),
        }
    }
    Ok(())
}
