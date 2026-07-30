//! Config locks: which keys reach the model, and which one switches
//! the prompt from text into a template.
use crate::questness::config::{
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
fn the_questness_keys_are_split_off_and_everything_else_stays() {
    let config = nu::from_nuon_text("{env: {PWD: '/tmp'}, liquid: true, tone: terse}")
        .expect("nuon");
    let (questness, visible) = split(&config).expect("splits");
    assert!(questness.get(LIQUID_KEY).is_some());
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
    let config = nu::from_nuon_text("{env: {PWD: '/tmp/foo'}, liquid: true}").expect("nuon");
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

#[test]
fn a_template_referencing_an_unbound_channel_fails_the_turn() {
    let config = nu::from_nuon_text("{liquid: true}").expect("nuon");
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

/// The key is Questness-addressed: interpreted, then stripped, so the
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
