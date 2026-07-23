use crate::*;

#[test]
fn chat_wrap_matches_template_default_rendering() {
    let expected = "<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
    assert_eq!(chat_wrap("Hi"), expected);
}
