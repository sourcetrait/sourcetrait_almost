//! Shape locks: what a caller declared, and what comes back under it.
use crate::nu;
use crate::questness::shape::{
    Shape,
    ShapeMember,
};
use crate::questness::turn::Answer;

fn value(nuon: &str) -> nu::Value {
    nu::from_nuon_text(nuon).expect("fixture parses")
}

fn shape(nuon: &str) -> Shape {
    Shape::of(&value(nuon)).expect("the shape parses")
}

fn answered(rendered: &str, produced: Option<&str>, config: Option<&str>) -> Answer {
    Answer {
        value: produced.map(value),
        declared: None,
        rendered: String::from(rendered),
        config: config.map(value),
    }
}

#[test]
fn a_config_with_no_shape_is_the_base_models_own() {
    assert_eq!(shape("{env: {PWD: '/tmp'}}"), Shape::default());
    assert_eq!(
        Shape::default().request,
        vec![ShapeMember::Text],
        "text in, text out is the checkpoint-native protocol"
    );
}

#[test]
fn each_direction_falls_back_on_its_own() {
    let only_response = shape("{shape: {response: [output]}}");
    assert_eq!(only_response.request, vec![ShapeMember::Text]);
    assert_eq!(only_response.response, vec![ShapeMember::Output]);

    let only_request = shape("{shape: {request: [config, text]}}");
    assert_eq!(only_request.request, vec![ShapeMember::Config, ShapeMember::Text]);
    assert_eq!(only_request.response, vec![ShapeMember::Text]);

    assert_eq!(shape("{shape: {}}"), Shape::default(), "and both at once");
}

#[test]
fn the_declared_order_is_the_order_it_reads_back() {
    let declared = shape("{shape: {request: [config, output, text], response: [text, output]}}");
    assert_eq!(
        declared.request,
        vec![ShapeMember::Config, ShapeMember::Output, ShapeMember::Text]
    );
    assert_eq!(declared.response, vec![ShapeMember::Text, ShapeMember::Output]);
}

#[test]
fn a_member_outside_the_vocabulary_is_refused() {
    let error = Shape::of(&value("{shape: {response: [input]}}"))
        .expect_err("input is not a shape member")
        .to_string();
    assert!(error.contains("output, text or config"), "got {error}");
}

#[test]
fn a_repeated_member_is_refused() {
    assert!(Shape::of(&value("{shape: {response: [text, text]}}")).is_err());
}

#[test]
fn an_empty_direction_is_refused_rather_than_defaulted() {
    let error = Shape::of(&value("{shape: {response: []}}"))
        .expect_err("an empty list says something an absent key does not")
        .to_string();
    assert!(error.contains("omit it for the default"), "got {error}");
}

#[test]
fn a_shape_that_is_not_a_record_is_refused() {
    assert!(Shape::of(&value("{shape: [text]}")).is_err());
    assert!(Shape::of(&value("{shape: {response: text}}")).is_err());
}

#[test]
fn one_member_returns_it_bare_and_several_return_a_record() {
    let answer = answered("Hello, Roy.", Some("{n: 3}"), None);

    let text = shape("{shape: {response: [text]}}")
        .value_of(&answer)
        .expect("conforms");
    assert_eq!(text.coerce_into_string().expect("a string"), "Hello, Roy.");

    let output = shape("{shape: {response: [output]}}")
        .value_of(&answer)
        .expect("conforms");
    assert_eq!(
        output.get_data_by_key("n").and_then(|v| v.as_int().ok()),
        Some(3),
        "the typed value comes back bare rather than wrapped"
    );

    let both = shape("{shape: {response: [output, text]}}")
        .value_of(&answer)
        .expect("conforms");
    let nu::Value::Record { val, .. } = &both else {
        panic!("several members carry a record, got {both:?}");
    };
    assert_eq!(
        val.columns().map(String::as_str).collect::<Vec<&str>>(),
        vec!["output", "text"],
        "keyed by member, in the order the caller declared"
    );
}

#[test]
fn a_declared_member_the_answer_never_carried_asks_for_repair() {
    let prose_only = answered("Hello, Roy.", None, None);
    let envelope = shape("{shape: {response: [output]}}")
        .value_of(&prose_only)
        .expect_err("an answer with no value owes one");
    assert!(
        envelope.errors.iter().any(|row| row.kind == "shape::missing"),
        "got {envelope:?}"
    );
}

#[test]
fn an_undeclared_config_is_denied_rather_than_passed_on() {
    let with_config = answered("Hello.", None, Some("{tone: terse}"));
    let envelope = shape("{shape: {response: [text]}}")
        .value_of(&with_config)
        .expect_err("the shape is the contract");
    assert!(
        envelope
            .errors
            .iter()
            .any(|row| row.kind == "shape::undeclared"),
        "got {envelope:?}"
    );
}

#[test]
fn a_declared_config_is_carried() {
    let with_config = answered("Hello.", None, Some("{tone: terse}"));
    let carried = shape("{shape: {response: [config]}}")
        .value_of(&with_config)
        .expect("declared, so allowed");
    assert_eq!(
        carried
            .get_data_by_key("tone")
            .and_then(|v| v.coerce_into_string().ok()),
        Some(String::from("terse"))
    );
}
