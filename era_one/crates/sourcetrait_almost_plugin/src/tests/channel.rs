use crate::*;

#[test]
fn frames_marker_plus_payload() {
    assert_eq!(frame_channel(INPUT_DATA, "{a: 1}"), "<|extra_id_4|>{a: 1}");
}

#[test]
fn extracts_after_output_marker() {
    let text = "<|extra_id_6|>{a: 1}\n";
    assert_eq!(extract_output(text), "{a: 1}");
}

#[test]
fn extracts_whole_text_without_marker() {
    assert_eq!(extract_output("  {a: 1}\n"), "{a: 1}");
}
