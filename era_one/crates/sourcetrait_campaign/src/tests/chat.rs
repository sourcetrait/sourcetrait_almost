use crate::chat::{base62, byte_index, render_log, wrap_line, Speaker, Turn};

#[test]
fn wrap_respects_width_and_words() {
    let rows = wrap_line("the quick brown fox jumps", 10);
    assert_eq!(rows, vec!["the quick", "brown fox", "jumps"]);
    assert!(rows.iter().all(|row| row.chars().count() <= 10));
}

#[test]
fn wrap_hard_splits_overlong_words() {
    let rows = wrap_line("abcdefghijkl", 5);
    assert_eq!(rows, vec!["abcde", "fghij", "kl"]);
}

#[test]
fn wrap_keeps_empty_lines() {
    assert_eq!(wrap_line("", 10), vec![""]);
}

#[test]
fn byte_index_is_utf8_safe() {
    let text = "aé€b";
    assert_eq!(byte_index(text, 0), 0);
    assert_eq!(byte_index(text, 1), 1);
    assert_eq!(byte_index(text, 2), 3);
    assert_eq!(byte_index(text, 3), 6);
    assert_eq!(byte_index(text, 4), 7);
    assert_eq!(byte_index(text, 9), 7);
}

#[test]
fn base62_renders_the_full_alphabet() {
    assert_eq!(base62(0), "0");
    assert_eq!(base62(9), "9");
    assert_eq!(base62(10), "A");
    assert_eq!(base62(35), "Z");
    assert_eq!(base62(36), "a");
    assert_eq!(base62(61), "z");
    assert_eq!(base62(62), "10");
    assert_eq!(base62(u64::MAX), "LygHa16AHYF");
}

#[test]
fn log_renders_speaker_prefixed_turns()  {
    let turns = vec![
        Turn { speaker: Speaker::You, text: String::from("hi") },
        Turn { speaker: Speaker::Quest, text: String::from("hello") },
        Turn { speaker: Speaker::Note, text: String::from("note") },
    ];
    assert_eq!(render_log(&turns), "you: hi\n\nquest: hello\n\nnote\n\n");
}
