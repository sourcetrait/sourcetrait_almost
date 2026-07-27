//! Contract locks: reading a def's channels back out of its signature,
//! and the three agreements checkable before anything executes.
use crate::channel::{
    Aliasing,
    parse_blocks,
};
use crate::contract::check_agreements;
use crate::evaluate::QuestnessEvaluator;

/// The worked example from the channel design, in wire spelling.
const WORKED: &str = "\
<|extra_id_2|> list<string>
[Var, bar, Car]
<|extra_id_6|>
<|extra_id_3|>$args<|extra_id_6|>
<|extra_id_2|> record<name: string>
{name: foo}
<|extra_id_6|>
<|extra_id_3|>$in<|extra_id_6|>";

const INTERACT: &str =
    "def --env interact [args: list<string>]: record<name: string> -> list<string> { [] }";

fn evaluator() -> QuestnessEvaluator {
    QuestnessEvaluator::new().expect("the evaluator builds")
}

#[test]
fn a_defs_channels_come_back_out_of_its_signature() {
    let contract = evaluator().contract(INTERACT).expect("contract");
    assert_eq!(contract.mode, "interact");
    assert!(contract.takes_pipeline());
    assert_eq!(contract.input.to_string(), "record<name: string>");
    assert_eq!(contract.output.to_string(), "list<string>");
    assert_eq!(
        contract.args.as_ref().map(ToString::to_string),
        Some(String::from("list<string>"))
    );
}

#[test]
fn the_mode_is_whatever_the_def_is_named() {
    let contract = evaluator()
        .contract("def evaluate []: list<string> -> list<string> { $in }")
        .expect("contract");
    assert_eq!(contract.mode, "evaluate");
    assert!(contract.args.is_none());
}

#[test]
fn a_def_with_no_pipeline_input_says_so() {
    let contract = evaluator()
        .contract("def execute []: nothing -> int { 1 }")
        .expect("contract");
    assert!(!contract.takes_pipeline());
}

#[test]
fn a_body_declaring_nothing_or_several_things_is_rejected() {
    let engine = evaluator();
    assert!(engine.contract("1 + 1").is_err());
    assert!(engine.contract("def a []: nothing -> int { 1 }; def b [] { 2 }").is_err());
}

#[test]
fn the_worked_example_agrees_with_its_def() {
    let blocks = parse_blocks(WORKED, Aliasing::Strict).expect("parses");
    let contract = evaluator().contract(INTERACT).expect("contract");
    let envelope = check_agreements(&blocks, &contract);
    assert!(envelope.is_clean(), "{:?}", envelope.errors);
}

#[test]
fn a_missing_in_binding_is_reported() {
    let text = "<|extra_id_2|> list<string>\n[a]\n<|extra_id_6|>\n<|extra_id_3|>$args<|extra_id_6|>";
    let blocks = parse_blocks(text, Aliasing::Strict).expect("parses");
    let contract = evaluator().contract(INTERACT).expect("contract");
    let envelope = check_agreements(&blocks, &contract);
    assert!(
        envelope
            .errors
            .iter()
            .any(|row| row.kind == "channel::missing_pass" && row.source.as_deref() == Some("$in"))
    );
}

#[test]
fn binding_args_against_a_def_without_one_is_reported() {
    let text = "<|extra_id_2|> list<string>\n[a]\n<|extra_id_6|>\n<|extra_id_3|>$args<|extra_id_6|>";
    let blocks = parse_blocks(text, Aliasing::Strict).expect("parses");
    let contract = evaluator()
        .contract("def evaluate []: nothing -> int { 1 }")
        .expect("contract");
    let envelope = check_agreements(&blocks, &contract);
    assert!(
        envelope
            .errors
            .iter()
            .any(|row| row.kind == "channel::unbound_pass"
                && row.source.as_deref() == Some("$args"))
    );
}

#[test]
fn a_type_disagreement_names_both_sides() {
    let text = "<|extra_id_2|> list<int>\n[1]\n<|extra_id_6|>\n<|extra_id_3|>$args<|extra_id_6|>\
                \n<|extra_id_2|> record<name: string>\n{name: foo}\n<|extra_id_6|>\
                \n<|extra_id_3|>$in<|extra_id_6|>";
    let blocks = parse_blocks(text, Aliasing::Strict).expect("parses");
    let contract = evaluator().contract(INTERACT).expect("contract");
    let envelope = check_agreements(&blocks, &contract);
    let row = envelope
        .errors
        .iter()
        .find(|row| row.kind == "channel::type_disagreement")
        .expect("the list<int> block disagrees with list<string>");
    assert!(row.message.contains("list<int>"), "{}", row.message);
    assert!(row.message.contains("list<string>"), "{}", row.message);
}

#[test]
fn the_envelope_collects_rather_than_bailing() {
    // Two independent faults: an unknown binding, and a pass with no
    // block in front of it. A first-error bail would report one.
    let text = "<|extra_id_3|>$nope<|extra_id_6|>\n<|extra_id_3|>$in<|extra_id_6|>";
    let blocks = parse_blocks(text, Aliasing::Strict).expect("parses");
    let contract = evaluator().contract(INTERACT).expect("contract");
    let envelope = check_agreements(&blocks, &contract);
    assert!(envelope.errors.len() >= 2, "{:?}", envelope.errors);
    let kinds: Vec<&str> = envelope.errors.iter().map(|row| row.kind.as_str()).collect();
    assert!(kinds.contains(&"channel::unknown_binding"), "{kinds:?}");
}

#[test]
fn a_duplicate_binding_is_reported_once() {
    let text = "<|extra_id_2|> list<string>\n[a]\n<|extra_id_6|>\n\
                <|extra_id_3|>$args<|extra_id_6|>\n\
                <|extra_id_2|> list<string>\n[b]\n<|extra_id_6|>\n\
                <|extra_id_3|>$args<|extra_id_6|>";
    let blocks = parse_blocks(text, Aliasing::Strict).expect("parses");
    let contract = evaluator().contract(INTERACT).expect("contract");
    let envelope = check_agreements(&blocks, &contract);
    assert_eq!(
        envelope
            .errors
            .iter()
            .filter(|row| row.kind == "channel::duplicate_binding")
            .count(),
        1
    );
}
