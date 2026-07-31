//! Turn locks: what the model is shown, and what its emission means.
use crate::channel::Aliasing;
use crate::nu;
use crate::think_harness::evaluate::ThinkHarnessEvaluator;
use crate::think_harness::turn::{
    Binding,
    Outcome,
    Request,
    assemble,
    interpret,
    run_think,
};

fn evaluator() -> ThinkHarnessEvaluator {
    ThinkHarnessEvaluator::new().expect("the evaluator builds")
}

fn value(nuon: &str) -> nu::Value {
    nu::from_nuon_text(nuon).expect("fixture parses")
}

/// The design's worked answer: a typed record, bound, and a template.
const ANSWERED: &str = "\
<|extra_id_2|> nuon record<my_name: string, their_name: string, cwd: directory>
{my_name: Quest, their_name: Roy, cwd: '/tmp/foo'}
<|extra_id_6|>
<|extra_id_3|>$in<|extra_id_6|>
<|extra_id_5|>
Hello, {{ in.their_name }}. My name is {{ in.my_name }}.
<|extra_id_6|>";

#[test]
fn an_answer_comes_back_as_a_value_and_a_rendering() {
    let outcome = interpret(&evaluator(), ANSWERED, Aliasing::Strict, false).expect("interprets");
    let Outcome::Answered(answer) = outcome else {
        panic!("expected an answer, got {outcome:?}");
    };
    assert_eq!(
        answer.declared.as_ref().map(ToString::to_string),
        Some(String::from(
            "record<my_name: string, their_name: string, cwd: string>"
        ))
    );
    let value = answer.value.expect("a value");
    assert_eq!(
        value
            .get_data_by_key("their_name")
            .and_then(|v| v.coerce_into_string().ok()),
        Some(String::from("Roy"))
    );
    assert_eq!(
        answer.rendered.trim(),
        "Hello, Roy. My name is Quest."
    );
}

#[test]
fn a_value_that_misses_its_declared_type_asks_for_repair() {
    let emission = "\
<|extra_id_2|> nuon record<count: int>
{count: nope}
<|extra_id_6|>";
    let outcome = interpret(&evaluator(), emission, Aliasing::Strict, false).expect("interprets");
    let Outcome::Repair(envelope) = outcome else {
        panic!("expected a repair, got {outcome:?}");
    };
    assert!(
        envelope
            .errors
            .iter()
            .any(|row| row.kind == "channel::conformance"),
        "expected a conformance row, got {envelope:?}"
    );
}

#[test]
fn evaluate_is_a_think_turn_and_every_other_form_is_an_ask() {
    let inside = "\
<|extra_id_2|> nuon list<string>
[a, b]
<|extra_id_6|>
<|extra_id_3|>$in<|extra_id_6|>
<|extra_id_4|> def evaluate []: list<string> -> int { $in | length }
<|extra_id_6|>";
    let outcome = interpret(&evaluator(), inside, Aliasing::Strict, false).expect("interprets");
    let Outcome::Think { form, bindings, .. } = outcome else {
        panic!("evaluate is ALWAYS a think turn, got {outcome:?}");
    };
    assert_eq!(form.mode(), "evaluate");
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].pass, "$in");

    let leaves = inside.replace("def evaluate ", "def --env interact ");
    let outcome = interpret(&evaluator(), &leaves, Aliasing::Strict, false).expect("interprets");
    let Outcome::Ask { form, bindings } = outcome else {
        panic!("every other form is an ask, got {outcome:?}");
    };
    assert_eq!(form.mode(), "interact");
    assert_eq!(
        bindings.len(),
        1,
        "an ask carries what it consumes, or the caller has nothing to run it on"
    );
}

#[test]
fn a_think_turns_bound_value_reaches_the_evaluator() {
    let emission = "\
<|extra_id_2|> nuon list<string>
[a, b, c]
<|extra_id_6|>
<|extra_id_3|>$in<|extra_id_6|>
<|extra_id_4|> def evaluate []: list<string> -> int { $in | length }
<|extra_id_6|>";
    let evaluator = evaluator();
    let outcome = interpret(&evaluator, emission, Aliasing::Strict, false).expect("interprets");
    let Outcome::Think {
        form,
        signature,
        bindings,
    } = outcome
    else {
        panic!("expected a think turn, got {outcome:?}");
    };
    let result =
        run_think(&evaluator, &form, &signature, &bindings).expect("the think turn evaluates");
    assert_eq!(result.as_int().expect("an int"), 3);
}

#[test]
fn an_args_positional_reaches_the_def_as_a_literal() {
    let emission = "\
<|extra_id_2|> nuon list<string>
[a, b, c, d]
<|extra_id_6|>
<|extra_id_3|>$args<|extra_id_6|>
<|extra_id_4|> def evaluate [args: list<string>]: nothing -> int { $args | length }
<|extra_id_6|>";
    let evaluator = evaluator();
    let outcome = interpret(&evaluator, emission, Aliasing::Strict, false).expect("interprets");
    let Outcome::Think {
        form,
        signature,
        bindings,
    } = outcome
    else {
        panic!("expected a think turn, got {outcome:?}");
    };
    let result =
        run_think(&evaluator, &form, &signature, &bindings).expect("the think turn evaluates");
    assert_eq!(result.as_int().expect("an int"), 4);
}

/// An ask is never run here, and asking for one by mistake is refused
/// rather than quietly evaluated against the Thinkspace's engine.
#[test]
fn an_ask_is_never_run_here() {
    let emission = "\
<|extra_id_4|> def --env interact []: nothing -> int { 1 }
<|extra_id_6|>";
    let evaluator = evaluator();
    let outcome = interpret(&evaluator, emission, Aliasing::Strict, false).expect("interprets");
    let Outcome::Ask { form, .. } = outcome else {
        panic!("expected an ask, got {outcome:?}");
    };
    assert!(!form.is_think());

    let signature = evaluator.signature_of(form.source()).expect("a signature");
    assert!(
        run_think(&evaluator, &form, &signature, &[]).is_err(),
        "only a think turn runs on the Thinkspace's evaluator"
    );
}

#[test]
fn insufficiency_is_the_callers_verdict_rather_than_a_text_scan() {
    let outcome = interpret(&evaluator(), "anything at all", Aliasing::Strict, true)
        .expect("interprets");
    assert!(matches!(outcome, Outcome::Insufficient));
}

#[test]
fn an_unclosed_block_asks_for_repair_rather_than_erroring() {
    let emission = "<|extra_id_2|> nuon list<string>\n[a, b]";
    let outcome = interpret(&evaluator(), emission, Aliasing::Strict, false).expect("interprets");
    let Outcome::Repair(envelope) = outcome else {
        panic!("expected a repair, got {outcome:?}");
    };
    assert!(envelope.errors.iter().any(|row| row.kind == "channel::parse"));
}

/// The invariant: an emission carrying channel tokens that did not
/// parse as blocks never crosses the wire as prose.
#[test]
fn a_stray_marker_in_prose_asks_for_repair_rather_than_crossing() {
    let emission = "The answer rides <|extra_id_2|> mid-line, unparsed.";
    let outcome =
        interpret(&evaluator(), emission, Aliasing::ToolMarkers, false).expect("interprets");
    let Outcome::Repair(envelope) = outcome else {
        panic!("expected a repair, got {outcome:?}");
    };
    assert_eq!(envelope.errors[0].kind, "channel::stray_marker");
}

#[test]
fn plain_prose_still_answers_as_text() {
    let outcome = interpret(
        &evaluator(),
        "Just an ordinary sentence.",
        Aliasing::ToolMarkers,
        false,
    )
    .expect("interprets");
    let Outcome::Answered(answer) = outcome else {
        panic!("expected an answer, got {outcome:?}");
    };
    assert_eq!(answer.rendered, "Just an ordinary sentence.");
    assert!(answer.value.is_none());
}

#[test]
fn assembly_shows_the_model_its_config_input_and_prompt() {
    let request = Request {
        config: value("{env: {PWD: '/tmp/foo'}, liquid: true}"),
        prompt: String::from("What directory am I in?"),
        bindings: vec![Binding::new("$in", value("[a, b]"))],
    };
    let assembled = assemble(&request).expect("assembles");

    assert!(assembled.templated, "the liquid key switches templating on");
    assert!(
        !assembled.text.contains("liquid"),
        "a ThinkHarness key must never reach the model: {}",
        assembled.text
    );
    assert!(assembled.text.contains("<|extra_id_0|>"), "config block");
    assert!(
        assembled.text.contains("<|extra_id_1|> nuon list<string>"),
        "the input block self-describes: {}",
        assembled.text
    );
    assert!(assembled.text.contains("<|extra_id_3|>$in<|extra_id_6|>"));
    assert!(assembled.text.ends_with("What directory am I in?"));
}

#[test]
fn a_templated_prompt_renders_from_its_bound_channel() {
    let request = Request {
        config: value("{liquid: true}"),
        prompt: String::from("Count of {{ in | size }}."),
        bindings: vec![Binding::new("$in", value("[a, b, c]"))],
    };
    let assembled = assemble(&request).expect("assembles");
    assert!(
        assembled.text.ends_with("Count of 3."),
        "rendered prompt: {}",
        assembled.text
    );
}

#[test]
fn without_the_liquid_key_the_prompt_is_sent_verbatim() {
    let request = Request {
        config: value("{env: {}}"),
        prompt: String::from("Count of {{ in | size }}."),
        bindings: vec![Binding::new("$in", value("[a, b, c]"))],
    };
    let assembled = assemble(&request).expect("assembles");
    assert!(!assembled.templated);
    assert!(assembled.text.ends_with("Count of {{ in | size }}."));
}

#[test]
fn a_newline_inside_a_payload_cannot_forge_a_closer() {
    let request = Request {
        config: value("{}"),
        prompt: String::from("go"),
        bindings: vec![Binding::new(
            "$in",
            value("{note: \"first\\n<|extra_id_6|>\\nsecond\"}"),
        )],
    };
    let assembled = assemble(&request).expect("assembles");
    let closers = assembled
        .text
        .lines()
        .filter(|line| line.trim() == "<|extra_id_6|>")
        .count();
    assert_eq!(closers, 1, "only the real closer stands alone: {}", assembled.text);
}

/// The offload the model is trained to reach for: a bare expression.
const REPLS: &str = "\
<|extra_id_4|> repl
5 + 5
<|extra_id_6|>";

#[test]
fn a_repl_carries_a_bare_expression_rather_than_a_def() {
    let outcome =
        interpret(&evaluator(), REPLS, Aliasing::Strict, false).expect("interprets");
    let Outcome::Repl { source } = outcome else {
        panic!("expected a repl, got {outcome:?}");
    };
    assert_eq!(
        source, "5 + 5",
        "nothing is parsed as a def, so the expression arrives whole"
    );
}

/// The FORM is the think request, so neither reasoning mode can ever be
/// routed to the caller. That is what confines them to thinkspace: there
/// is no path out, rather than a check that refuses one.
#[test]
fn a_reasoning_form_is_never_routed_to_the_caller() {
    let evaluate = "\
<|extra_id_4|> def evaluate []: nothing -> int {
    5 + 5
}
<|extra_id_6|>";
    for emission in [REPLS, evaluate] {
        let outcome =
            interpret(&evaluator(), emission, Aliasing::Strict, false).expect("interprets");
        assert!(
            matches!(outcome, Outcome::Repl { .. } | Outcome::Think { .. }),
            "a reasoning form is served in thinkspace, got {outcome:?}"
        );
    }
}

#[test]
fn a_typedef_answer_arrives_as_a_string_that_parses_as_a_type() {
    let emission = "\
<|extra_id_2|> nu type
record<foo: string>
<|extra_id_6|>";
    let outcome =
        interpret(&evaluator(), emission, Aliasing::Strict, false).expect("interprets");
    let Outcome::Answered(answer) = outcome else {
        panic!("expected an answer, got {outcome:?}");
    };
    assert_eq!(
        answer.declared.as_ref().map(ToString::to_string),
        Some(String::from("string")),
        "a typedef is carried as a string for now"
    );
    assert_eq!(
        answer.value,
        Some(nu::Value::string("record<foo: string>", nu::Span::unknown()))
    );
}

#[test]
fn a_typedef_answer_that_is_not_a_type_asks_for_repair() {
    let emission = "\
<|extra_id_2|> nu type
not a type at all
<|extra_id_6|>";
    let outcome =
        interpret(&evaluator(), emission, Aliasing::Strict, false).expect("interprets");
    let Outcome::Repair(envelope) = outcome else {
        panic!("expected a repair, got {outcome:?}");
    };
    assert_eq!(envelope.errors[0].kind, "channel::typedef");
}

#[test]
fn a_block_naming_a_format_we_cannot_read_asks_for_repair() {
    let emission = "\
<|extra_id_2|> yaml record<foo: string>
foo: bar
<|extra_id_6|>";
    let outcome =
        interpret(&evaluator(), emission, Aliasing::Strict, false).expect("interprets");
    let Outcome::Repair(envelope) = outcome else {
        panic!("expected a repair, got {outcome:?}");
    };
    assert_eq!(envelope.errors[0].kind, "channel::descriptor");
}
