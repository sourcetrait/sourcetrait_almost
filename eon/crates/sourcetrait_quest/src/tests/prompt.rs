//! Prompt locks: where a prompt comes from, and what rides with it.
use crate::*;
use crate::prompt::{
    CONFIG_FLAG,
    FILE_FLAG,
    FILE_PROMPT_FIELD,
    PROMPT_FIELD,
    asked,
    shaped,
};

/// A scratch file, removed when the guard drops.
struct Scratch {
    dir: std::path::PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "quest_plugin_{tag}_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("the scratch directory is made");
        Self { dir }
    }

    fn holding(&self, name: &str, text: &str) -> String {
        let path = self.dir.join(name);
        std::fs::write(&path, text).expect("the fixture is written");
        path.display().to_string()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn span() -> nu_protocol::Span {
    nu_protocol::Span::unknown()
}

fn call() -> nu_plugin::EvaluatedCall {
    nu_plugin::EvaluatedCall::new(span())
}

fn named(call: nu_plugin::EvaluatedCall, flag: &str, value: &str) -> nu_plugin::EvaluatedCall {
    use nu_protocol::IntoSpanned;
    call.with_named(
        flag.to_string().into_spanned(span()),
        nu_protocol::Value::string(value, span()),
    )
}

fn nothing() -> nu_protocol::Value {
    nu_protocol::Value::nothing(span())
}

fn record(nuon: &str) -> nu_protocol::Value {
    harness_lib::nu::from_nuon_text(nuon).expect("fixture parses")
}

#[test]
fn the_positional_is_a_prompt_and_binds_nothing() {
    let asked = asked(&call().with_positional(nu_protocol::Value::string("go", span())), &nothing())
        .expect("resolves");
    assert_eq!(asked.prompt, "go");
    assert!(asked.bindings.is_empty(), "nothing was piped");
}

#[test]
fn a_file_carries_the_prompt() {
    let scratch = Scratch::new("file");
    let path = scratch.holding("ask.liquid", "List the {{ in.count }} newest.");
    let asked = asked(&named(call(), FILE_FLAG, &path), &nothing()).expect("resolves");
    assert_eq!(asked.prompt, "List the {{ in.count }} newest.");
}

#[test]
fn a_piped_record_carries_its_prompt_and_binds_the_rest() {
    let piped = record("{prompt: 'summarise', rows: [a, b], n: 2}");
    let asked = asked(&call(), &piped).expect("resolves");
    assert_eq!(asked.prompt, "summarise");

    let (pass, input) = asked.bindings.first().expect("the rest binds");
    assert_eq!(*pass, bridge::InferPass::In);
    let bound = &input.value().0;
    let nu_protocol::Value::Record { val, .. } = bound else {
        panic!("the remainder is a record, got {bound:?}");
    };
    assert!(
        val.get(PROMPT_FIELD).is_none(),
        "the instruction is consumed rather than shown as data"
    );
    assert_eq!(val.columns().count(), 2, "everything else survives");
}

#[test]
fn a_piped_record_can_name_a_file_instead() {
    let scratch = Scratch::new("fprompt");
    let path = scratch.holding("ask.liquid", "count them");
    let piped = harness_lib::nu::from_nuon_text(&format!(
        "{{{FILE_PROMPT_FIELD}: '{path}', rows: [a]}}"
    ))
    .expect("fixture parses");
    let asked = asked(&call(), &piped).expect("resolves");
    assert_eq!(asked.prompt, "count them");
    assert_eq!(asked.bindings.len(), 1, "the rest still binds");
}

#[test]
fn a_record_that_is_only_a_prompt_binds_nothing() {
    let asked = asked(&call(), &record("{prompt: 'go'}")).expect("resolves");
    assert!(
        asked.bindings.is_empty(),
        "an emptied record is nothing rather than an empty binding"
    );
}

#[test]
fn a_piped_value_that_is_not_a_record_binds_whole() {
    let asked = asked(
        &call().with_positional(nu_protocol::Value::string("go", span())),
        &record("[a, b, c]"),
    )
    .expect("resolves");
    let (pass, input) = asked.bindings.first().expect("a list binds");
    assert_eq!(*pass, bridge::InferPass::In);
    assert!(matches!(input.value().0, nu_protocol::Value::List { .. }));
}

#[test]
fn two_sources_are_refused_rather_than_ranked() {
    let error = asked(
        &call().with_positional(nu_protocol::Value::string("go", span())),
        &record("{prompt: 'other'}"),
    )
    .expect_err("two spellings of one call must not mean different things")
    .to_string();
    assert!(error.contains("one source"), "got {error}");
    assert!(error.contains(PROMPT_FIELD), "it names them: {error}");
}

#[test]
fn no_source_is_refused() {
    let error = asked(&call(), &nothing())
        .expect_err("there is nothing to ask")
        .to_string();
    assert!(error.contains(FILE_FLAG), "it names where one goes: {error}");
}

#[test]
fn the_config_flag_is_not_a_prompt_source() {
    let call = named(call(), CONFIG_FLAG, "ignored");
    assert!(
        asked(&call, &nothing()).is_err(),
        "config carries the shape, never the question"
    );
}

fn response(
    output: Option<&str>,
    text: Option<&str>,
) -> bridge::InferResponse {
    bridge::InferResponse {
        output: output.map(|nuon| {
            bridge::InferOutput::Nuon(bridge::InferNuonOutput(bridge::InferValue(record(
                nuon,
            ))))
        }),
        text: text.map(|text| bridge::InferText(text.to_string())),
        ..bridge::InferResponse::default()
    }
}

#[test]
fn an_undeclared_response_delivers_the_models_own_choice() {
    let decide = harness_lib::Shape::default();

    let bare = shaped(&decide, &response(Some("{n: 3}"), None), span()).expect("delivers");
    assert_eq!(
        bare.get_data_by_key("n").and_then(|v| v.as_int().ok()),
        Some(3),
        "a typed value with no prose comes back bare"
    );

    let prose = shaped(&decide, &response(None, Some("done.")), span()).expect("delivers");
    assert_eq!(prose.coerce_into_string().expect("a string"), "done.");

    let both =
        shaped(&decide, &response(Some("{n: 3}"), Some("Three.")), span()).expect("delivers");
    let nu_protocol::Value::Record { val, .. } = &both else {
        panic!("value and prose together carry a record, got {both:?}");
    };
    assert_eq!(
        val.columns().map(String::as_str).collect::<Vec<&str>>(),
        vec!["output", "text"]
    );
}

#[test]
fn a_declared_response_still_pins_the_reply() {
    let pinned = harness_lib::Shape {
        request: vec![harness_lib::ShapeMember::Text],
        response: harness_lib::ShapeResponse::Declared(vec![
            harness_lib::ShapeMember::Output,
        ]),
    };
    let value =
        shaped(&pinned, &response(Some("{n: 3}"), Some("Three.")), span()).expect("delivers");
    assert_eq!(
        value.get_data_by_key("n").and_then(|v| v.as_int().ok()),
        Some(3),
        "the declared member comes back bare, the prose does not ride along"
    );
    assert!(
        shaped(&pinned, &response(None, Some("Three.")), span()).is_err(),
        "a declared member the answer never carried is owed"
    );
}
