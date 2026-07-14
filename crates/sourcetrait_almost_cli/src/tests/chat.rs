use crate::chat::{byte_index, wrap_line};

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
fn edit_round_trip_with_cursor() {
    let mut input = String::from("héllo");
    let mut cursor = 5usize;
    input.insert(byte_index(&input, cursor), '!');
    cursor += 1;
    assert_eq!(input, "héllo!");
    cursor -= 1;
    input.remove(byte_index(&input, cursor));
    assert_eq!(input, "héllo");
    let _ = cursor;
}
