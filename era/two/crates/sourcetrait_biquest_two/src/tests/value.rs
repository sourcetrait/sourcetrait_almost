//! Bridge locks: lossless JSON -> Value conversion (order, numbers,
//! nesting), the nuon round-trip, and the i64-overflow off-ramp.
use crate::*;

fn round_trips(json_text: &str) {
    let json: serde_json::Value = serde_json::from_str(json_text).expect("test JSON parses");
    let value = json_to_value(&json).expect("converts");
    let rendered = harness::nu::to_nuon_text(&value).expect("renders");
    let reparsed = harness::nu::from_nuon_text(&rendered).expect("reparses");
    assert_eq!(value, reparsed, "nuon round-trip must be lossless for {json_text}");
}

#[test]
fn scalars_and_nesting_round_trip() {
    round_trips(r#"{"a": null, "b": true, "c": -42, "d": 3.5, "e": "text"}"#);
    round_trips(r#"{"nested": {"list": [1, 2.5, "three", null, {"deep": []}]}}"#);
    round_trips(r#"{"text": "line one\nline two\ttabbed \"quoted\" \\ backslash"}"#);
    round_trips(r#"{"unicode": "emoji 😀 kanji 日本 cyrillic ж"}"#);
    round_trips(r#"{"empty_list": [], "empty_string": ""}"#);
}

#[test]
fn float_precision_round_trips() {
    round_trips(r#"{"sum_logits": -8.111950795195707}"#);
    round_trips(r#"{"tiny": -0.013073842070706073, "big": 1.7976931348623157e308}"#);
    round_trips(r#"{"int_like": 1.0, "neg": -4.986950795195706}"#);
}

#[test]
fn extreme_i64_round_trips() {
    round_trips(r#"{"max": 9223372036854775807, "min": -9223372036854775808}"#);
}

#[test]
fn object_key_order_is_preserved() {
    let json: serde_json::Value =
        serde_json::from_str(r#"{"zeta": 1, "alpha": 2, "mid": 3}"#).expect("parses");
    let value = json_to_value(&json).expect("converts");
    let record = value.as_record().expect("record");
    let columns: Vec<&String> = record.columns().collect();
    assert_eq!(columns, ["zeta", "alpha", "mid"]);
}

#[test]
fn value_to_json_inverts_the_bridge() {
    let json_text = r#"{"a": null, "b": [1, -2.5, "x"], "c": {"nested": true}, "d": 9223372036854775807}"#;
    let json: serde_json::Value = serde_json::from_str(json_text).expect("parses");
    let value = json_to_value(&json).expect("converts");
    let back = value_to_json(&value).expect("inverts");
    assert_eq!(json, back, "value_to_json must invert json_to_value");
}

#[test]
fn u64_overflow_is_refused() {
    let json: serde_json::Value =
        serde_json::from_str(r#"{"too_big": 9223372036854775808}"#).expect("parses");
    let error = json_to_value(&json).expect_err("must refuse");
    assert!(error.to_string().contains("exceeds i64"), "got: {error}");
}
