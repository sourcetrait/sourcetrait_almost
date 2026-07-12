use crate::*;

/// Byte-exact default rendering (no system message, no tools, generation
/// prompt appended) of the checkpoint's chat_template.jinja.
pub fn chat_wrap(user_prompt: &str) -> String {
    let mut wrapped = String::new();
    wrapped.push_str("<|im_start|>system\n");
    wrapped.push_str(
        "You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions>",
    );
    wrapped.push_str("<|im_end|>\n");
    wrapped.push_str("<|im_start|>user\n");
    wrapped.push_str(user_prompt);
    wrapped.push_str("<|im_end|>\n");
    wrapped.push_str("<|im_start|>assistant\n");
    wrapped
}

/// Stop-token ids resolved by string, so tokenizer truth wins over any
/// config-side id drift.
pub fn resolve_stop_ids(tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    consts::STOP_TOKENS
        .iter()
        .filter_map(|token| tokenizer.token_to_id(token))
        .collect()
}
