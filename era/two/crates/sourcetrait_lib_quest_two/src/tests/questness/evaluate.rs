//! Evaluator locks: what the engine can reach, what it refuses, and
//! the one reach registration cannot close.
use crate::*;
use crate::questness::evaluate::{
    DENIED,
    PARSE_TIME_LOADERS,
    QuestnessEvaluator,
};

fn evaluator() -> QuestnessEvaluator {
    QuestnessEvaluator::new().expect("the evaluator builds")
}

#[test]
fn arithmetic_and_the_language_core_evaluate() {
    let engine = evaluator();
    let value = engine.evaluate("2 + 2 * 10", None).expect("evaluates");
    assert_eq!(value.as_int().expect("int"), 22);
    let value = engine
        .evaluate("if 3 > 2 { 'yes' } else { 'no' }", None)
        .expect("evaluates");
    assert_eq!(value.as_str().expect("string"), "yes");
}

#[test]
fn the_filter_set_is_registered_and_pipes() {
    let engine = evaluator();
    let value = engine
        .evaluate("[3, 1, 2] | sort | first 2 | length", None)
        .expect("evaluates");
    assert_eq!(value.as_int().expect("int"), 2);
}

#[test]
fn a_piped_value_arrives_as_the_input() {
    let engine = evaluator();
    let input = nu::from_nuon_text("[{n: 1}, {n: 2}, {n: 3}]").expect("nuon");
    let value = engine
        .evaluate("$in | get n | last", Some(input))
        .expect("evaluates");
    assert_eq!(value.as_int().expect("int"), 3);
}

#[test]
fn nothing_that_reaches_the_operating_system_resolves() {
    // MEASURED rather than assumed: the language core registers none
    // of these, and only add_shell_command_context brings them in.
    // This is a lock on not calling it rather than a mitigation.
    let engine = evaluator();
    for name in DENIED {
        assert!(!engine.resolves(name), "{name} must not resolve");
    }
    let error = engine
        .evaluate("ls /etc", None)
        .expect_err("an absent command is a parse error")
        .to_string();
    assert!(!error.is_empty());
}

#[test]
fn an_external_call_cannot_parse() {
    let engine = evaluator();
    // run-external is absent, so `^cmd` has nothing to bind to and
    // fails at PARSE time rather than being refused at run time.
    assert!(engine.evaluate("^whoami", None).is_err());
}

#[test]
fn the_parse_time_loaders_remain_reachable_and_that_is_the_sandbox_line() {
    // Recorded as a fact rather than closed. `use` resolves its path
    // at PARSE time, so it reads a file without any filesystem
    // command being registered. Registration cannot reach it; the
    // service-level bubblewrap is what covers it.
    let engine = evaluator();
    for name in PARSE_TIME_LOADERS {
        let head = name.split(' ').next().expect("non-empty");
        assert!(
            engine.resolves(head) || engine.resolves(name),
            "{name} is expected to be present, and is the sandbox's job"
        );
    }
}

#[test]
fn a_parse_error_is_rendered_rather_than_debug_dumped() {
    let engine = evaluator();
    let error = engine
        .evaluate("let = = =", None)
        .expect_err("garbage rejects")
        .to_string();
    assert!(!error.contains("ParseError {"), "got a debug dump: {error}");
}

#[test]
fn a_runtime_failure_returns_rather_than_ending_the_host() {
    let engine = evaluator();
    assert!(engine.evaluate("error make {msg: 'boom'}", None).is_err());
    // The host is still here and still usable.
    let value = engine.evaluate("1 + 1", None).expect("still evaluates");
    assert_eq!(value.as_int().expect("int"), 2);
}

#[test]
fn evaluations_do_not_leak_definitions_into_each_other() {
    let engine = evaluator();
    engine
        .evaluate("def leaked [] { 7 }; leaked", None)
        .expect("defines and calls within one body");
    assert!(
        engine.evaluate("leaked", None).is_err(),
        "the base engine must not carry a prior evaluation's delta"
    );
}
