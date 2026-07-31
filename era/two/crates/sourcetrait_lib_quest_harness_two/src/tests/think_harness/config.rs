//! Config locks: which keys reach the model, and which one switches
//! the prompt from text into a template.
use crate::think_harness::config::{
    CONVERSATION_KEY,
    Conversation,
    LIQUID_KEY,
    prepare,
    split,
};
use crate::nu;

fn bindings(nuon: &str) -> Vec<(String, nu::Value)> {
    vec![(
        String::from("$in"),
        nu::from_nuon_text(nuon).expect("nuon parses"),
    )]
}

#[test]
fn the_think_harness_keys_are_split_off_and_everything_else_stays() {
    let config = nu::from_nuon_text("{env: {PWD: '/tmp'}, liquid: {}, tone: terse}")
        .expect("nuon");
    let (think_harness, visible) = split(&config).expect("splits");
    assert!(think_harness.get(LIQUID_KEY).is_some());
    assert!(visible.get(LIQUID_KEY).is_none());
    assert!(visible.get("env").is_some(), "env is the model's own");
    assert!(visible.get("tone").is_some(), "an unfamiliar key passes through");
}

#[test]
fn a_config_must_be_a_record() {
    let error = split(&nu::from_nuon_text("[1, 2]").expect("nuon"))
        .expect_err("a list is not a config")
        .to_string();
    assert!(error.contains("a config is a record"), "got {error}");
}

#[test]
fn without_the_key_the_prompt_is_sent_verbatim() {
    let config = nu::from_nuon_text("{env: {PWD: '/tmp'}}").expect("nuon");
    let turn = prepare(&config, "What directory am I in? {{ in.n }}", &bindings("{n: 1}"))
        .expect("prepares");
    assert!(!turn.templated);
    assert_eq!(turn.prompt, "What directory am I in? {{ in.n }}");
}

#[test]
fn with_the_key_the_prompt_renders_and_the_key_does_not_travel() {
    let config = nu::from_nuon_text("{env: {PWD: '/tmp/foo'}, liquid: {}}").expect("nuon");
    let turn = prepare(
        &config,
        "List the {{ in.count }} newest files.",
        &bindings("{count: 3}"),
    )
    .expect("prepares");
    assert!(turn.templated);
    assert_eq!(turn.prompt, "List the 3 newest files.");
    let nu::Value::Record { val, .. } = &turn.visible else {
        panic!("the visible config is a record");
    };
    assert!(val.get(LIQUID_KEY).is_none(), "liquid never reaches the model");
    assert!(val.get("env").is_some(), "env does");
}

/// The infill contract: the liquid record's keys are template bindings,
/// so a literal reaches the question's own text with nothing piped.
#[test]
fn the_liquid_records_keys_bind_into_the_template() {
    let config = nu::from_nuon_text("{liquid: {train: {expression: '47 + 68'}}}")
        .expect("nuon");
    let turn = prepare(&config, "What is {{ train.expression }}?", &[])
        .expect("prepares");
    assert!(turn.templated);
    assert_eq!(turn.prompt, "What is 47 + 68?");
}

/// Both channels serve one render: the pass bindings and the liquid
/// record's keys address the same template.
#[test]
fn liquid_keys_bind_beside_the_pass_channels() {
    let config = nu::from_nuon_text("{liquid: {train: {sep: '/'}}}").expect("nuon");
    let turn = prepare(
        &config,
        "Join {{ in.count }} parts with {{ train.sep }}.",
        &bindings("{count: 3}"),
    )
    .expect("prepares");
    assert_eq!(turn.prompt, "Join 3 parts with /.");
}

/// The caller is code, so a flag where the bindings belong is refused
/// rather than read as an empty record.
#[test]
fn a_liquid_value_that_is_not_a_record_is_refused() {
    let config = nu::from_nuon_text("{liquid: true}").expect("nuon");
    let error = prepare(&config, "hi", &[]).expect_err("refused").to_string();
    assert!(error.contains("record of template bindings"), "got {error}");
}

/// A liquid key shadowing a pass channel would make one name mean two
/// values, so it is refused rather than either one silently winning.
#[test]
fn a_liquid_key_colliding_with_a_pass_channel_is_refused() {
    let config = nu::from_nuon_text("{liquid: {in: {n: 2}}}").expect("nuon");
    let error = prepare(&config, "{{ in.n }}", &bindings("{n: 1}"))
        .expect_err("refused")
        .to_string();
    assert!(error.contains("already binds"), "got {error}");
}

#[test]
fn a_template_referencing_an_unbound_channel_fails_the_turn() {
    let config = nu::from_nuon_text("{liquid: {}}").expect("nuon");
    assert!(prepare(&config, "{{ args.0 }}", &bindings("{n: 1}")).is_err());
}

/// Teardown is the default: a bare call declares nothing and gets a
/// fresh conversation; `{conversation: keep}` is the opt-in.
#[test]
fn the_conversation_tears_down_unless_kept() {
    let bare = nu::from_nuon_text("{}").expect("nuon");
    let turn = prepare(&bare, "hi", &[]).expect("prepares");
    assert_eq!(turn.conversation, Conversation::Teardown);

    let kept = nu::from_nuon_text("{conversation: keep}").expect("nuon");
    let turn = prepare(&kept, "hi", &[]).expect("prepares");
    assert_eq!(turn.conversation, Conversation::Keep);
}

/// The caller is code, so a value outside the vocabulary is refused
/// rather than silently read as either disposition.
#[test]
fn an_unknown_conversation_value_is_refused() {
    let config = nu::from_nuon_text("{conversation: forever}").expect("nuon");
    let error = prepare(&config, "hi", &[]).expect_err("refused").to_string();
    assert!(error.contains("keep"), "got {error}");
}

/// The key is ThinkHarness-addressed: interpreted, then stripped, so the
/// model never sees it.
#[test]
fn the_conversation_key_never_reaches_the_model() {
    let config = nu::from_nuon_text("{env: {PWD: '/tmp'}, conversation: keep}").expect("nuon");
    let turn = prepare(&config, "hi", &[]).expect("prepares");
    let nu::Value::Record { val, .. } = &turn.visible else {
        panic!("the visible config is a record");
    };
    assert!(val.get(CONVERSATION_KEY).is_none(), "conversation never travels");
    assert!(val.get("env").is_some(), "env does");
}
