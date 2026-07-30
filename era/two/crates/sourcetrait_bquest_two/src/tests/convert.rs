//! CapabilityNuon bridge locks: lossless JSON -> Value conversion
//! (order, numbers, nesting), the nuon round-trip the convert gate
//! leans on, the i64-overflow off-ramp, task-token derivation, and
//! the harvested typedefs parsing. Checkpoint-free.
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

#[test]
fn task_tokens_strip_suffix_and_render_hash() {
    assert_eq!(
        task_token("mmlu_anatomy_7f552d-requests.jsonl", "-requests.jsonl"),
        "mmlu_anatomy"
    );
    assert_eq!(
        task_token("arc_challenge_mc_1543f3-predictions.jsonl", "-predictions.jsonl"),
        "arc_challenge_mc"
    );
    assert_eq!(task_token("gsm8k_9648bf-requests.jsonl", "-requests.jsonl"), "gsm8k");
    assert_eq!(
        task_token("ifeval_ood_705de1-requests.jsonl", "-requests.jsonl"),
        "ifeval_ood"
    );
    // No render hash: the stem stands as the token.
    assert_eq!(task_token("plain-requests.jsonl", "-requests.jsonl"), "plain");
}

#[test]
fn harvested_typedefs_parse_and_accept_real_shapes() {
    let loglik = harness::nu::parse_typedef(
        "record<request_type: string, \
         doc: record<query: string, gold_idx: int, choices: list<string>>, \
         request: record<context: string, continuations: list<string>>, \
         idx: int, task_name: string, doc_id: int, \
         native_id: oneof<int, string>, label: int>",
    );
    assert!(loglik.is_ok(), "loglikelihood typedef must parse: {loglik:?}");

    let row_json = r#"{"request_type": "loglikelihood",
        "doc": {"query": "Q", "id": 28, "gold_idx": 2, "choices": ["A", "B"]},
        "request": {"context": "ctx", "continuation": " A", "continuations": [" A", " B"]},
        "idx": 0, "task_name": "mmlu_anatomy", "doc_id": 0, "native_id": 28, "label": 2}"#;
    let json: serde_json::Value = serde_json::from_str(row_json).expect("parses");
    let value = json_to_value(&json).expect("converts");
    harness::nu::conform(&value, &loglik.expect("parsed")).expect("real shape conforms");
}
