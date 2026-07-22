#[allow(unused_imports)]
use crate::*;

/// The channel slot map by literal form (ids pinned in the KB:
/// extra_id_0 = 100256, extra_id_1..6 = 100270-100275, all
/// non-special). Resolution stays string-first - tokenizer truth wins
/// over any id constant.
pub const HARDCODED_CONFIGURATION: &str = "<|extra_id_0|>";
pub const PROMPT_CONFIGURATION_DEFINITION: &str = "<|extra_id_1|>";
pub const PROMPT_CONFIGURATION_DATA: &str = "<|extra_id_2|>";
pub const INPUT_DEFINITION: &str = "<|extra_id_3|>";
pub const INPUT_DATA: &str = "<|extra_id_4|>";
pub const OUTPUT_DEFINITION: &str = "<|extra_id_5|>";
pub const OUTPUT_DATA: &str = "<|extra_id_6|>";

/// The insufficiency tag (100276). Special-flagged: skip-special
/// decode strips it from text, so detection is by token id at the
/// engine layer, never by text scan.
pub const INSUFFICIENCY: &str = "<|endofprompt|>";

/// One prompt-side channel span: marker + payload. Termination is the
/// next marker or the turn end; nothing closes a span explicitly.
pub fn frame_channel(marker: &str, payload: &str) -> String {
    let mut framed = String::with_capacity(marker.len() + payload.len());
    framed.push_str(marker);
    framed.push_str(payload);
    framed
}

/// The OUTPUT_DATA payload of a decoded assistant turn: after the
/// marker when present (extra_ids survive skip-special decode), the
/// whole trimmed text otherwise (a quiet-config turn is pure payload).
pub fn extract_output(text: &str) -> &str {
    match text.find(OUTPUT_DATA) {
        Some(index) => text[index + OUTPUT_DATA.len()..].trim(),
        None => text.trim(),
    }
}
