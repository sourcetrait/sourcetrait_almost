//! Tokenizer locks: the added-token map, single-token markers, the
//! chat-render token count pinned by the C2 dumps.
use crate::*;

fn dpo_tokenizer() -> tokenizers::Tokenizer {
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    load_tokenizer(&dir).expect("dpo tokenizer loads")
}

#[test]
fn dpo_token_map_verifies() {
    verify_token_map(&dpo_tokenizer()).expect("dpo map matches the pins");
}

#[test]
fn base_token_map_differs_by_design() {
    let dir = model_dir(consts::BASE_MODEL_NAME).expect("base dir");
    let tokenizer = load_tokenizer(&dir).expect("base tokenizer loads");
    assert!(
        verify_token_map(&tokenizer).is_err(),
        "the base checkpoint's 100266-100275 range differs from the DPO map"
    );
}

#[test]
fn tool_markers_are_single_tokens() {
    let tokenizer = dpo_tokenizer();
    for (text, id) in [
        ("<functions>", consts::TOKEN_FUNCTIONS_OPEN),
        ("<function_calls>", consts::TOKEN_FUNCTION_CALLS_OPEN),
        ("<|extra_id_0|>", consts::TOKEN_EXTRA_ID_0),
    ] {
        let encoding = tokenizer.encode(text, false).expect("encodes");
        assert_eq!(encoding.get_ids(), [id], "{text}");
    }
}

#[test]
fn chat_wrap_renders_the_pinned_token_count() {
    // 43 tokens for this exact prompt text, pinned by the C2 dump ids
    // through the original stack.
    let tokenizer = dpo_tokenizer();
    let rendered = chat_wrap("What is the capital of France?");
    let encoding = tokenizer.encode(rendered, false).expect("encodes");
    assert_eq!(encoding.get_ids().len(), 43);
}

#[test]
fn stop_ids_resolve_by_string() {
    assert_eq!(
        resolve_stop_ids(&dpo_tokenizer()),
        vec![consts::TOKEN_IM_END, consts::TOKEN_ENDOFTEXT]
    );
}
