//! Arbitration locks: the whole turn loop, with no harness anywhere.
use crate::channel::Aliasing;
use crate::questness::arbitrate::{
    Questness,
    Step,
};

/// QUESTNESS IS A BLOCKING API ALL THE WAY DOWN, and this is the lock
/// that says the daemon can still use it. The evaluator spawns a sized
/// thread and joins it, so `step` blocks whatever else is true, and the
/// one correct way to drive it from an async daemon is `spawn_blocking`
/// - which needs the value to be `Send`.
#[test]
fn questness_can_cross_to_a_blocking_task() {
    fn assert_send<T: Send>() {}
    assert_send::<Questness>();
}

/// The model works something out for itself.
const THINKS: &str = "\
<|extra_id_2|> list<string>
[a, b, c]
<|extra_id_6|>
<|extra_id_3|>$in<|extra_id_6|>
<|extra_id_4|> def evaluate []: list<string> -> int { $in | length }
<|extra_id_6|>";

/// The model asks the CALLER to run something against the world.
const ASKS: &str = "\
<|extra_id_2|> record<cwd: string>
{cwd: ''}
<|extra_id_6|>
<|extra_id_3|>$in<|extra_id_6|>
<|extra_id_4|> def --env interact []: record<cwd: string> -> record<cwd: string> { $in }
<|extra_id_6|>";

/// The model answers.
const ANSWERS: &str = "\
<|extra_id_2|> record<cwd: string>
{cwd: '/tmp/foo'}
<|extra_id_6|>
<|extra_id_3|>$in<|extra_id_6|>
<|extra_id_5|>
We are currently in the {{ in.cwd }} directory.
<|extra_id_6|>";

#[test]
fn a_think_turn_runs_here_and_comes_back_as_a_thought() {
    let mut questness = Questness::new(Aliasing::Strict).expect("builds");
    let Step::Continue(back) = questness.step(THINKS, false).expect("steps") else {
        panic!("evaluate runs on the Thinkspace's own evaluator");
    };
    assert!(
        back.starts_with("<|im_start|>thought"),
        "the answer rides a thought turn: {back}"
    );
    assert!(
        back.contains("<|extra_id_2|> int"),
        "and it returns TYPED: {back}"
    );
    assert!(back.contains('3'), "the evaluator's own answer: {back}");
    assert!(
        back.trim_end().ends_with("<|im_end|>"),
        "the turn closes: {back}"
    );
}

/// Nothing here runs an ask. It rides back with what it consumes and
/// the turn pauses until the caller sends a result.
#[test]
fn an_ask_pauses_the_turn_rather_than_being_served() {
    let mut questness = Questness::new(Aliasing::Strict).expect("builds");
    let Step::Ask { form, bindings } = questness.step(ASKS, false).expect("steps") else {
        panic!("an ask is never serviced here");
    };
    assert_eq!(form.mode(), "interact");
    assert!(!form.is_think());
    assert_eq!(bindings.len(), 1, "the ask carries what it consumes");
    assert_eq!(bindings[0].pass, "$in");
}

#[test]
fn a_plain_answer_touches_no_engine_at_all() {
    let mut questness = Questness::new(Aliasing::Strict).expect("builds");
    let Step::Answered(answer) = questness.step(ANSWERS, false).expect("steps") else {
        panic!("expected an answer");
    };
    assert_eq!(
        answer.rendered.trim(),
        "We are currently in the /tmp/foo directory."
    );
}

/// A think that does not run is feedback rather than an error, because
/// in this grammar `<output>` means a value and the model should not
/// have to tell a result from a report of a non-result.
#[test]
fn a_failing_think_becomes_feedback() {
    let broken = "\
<|extra_id_4|> def evaluate []: nothing -> int { 1 / 0 }
<|extra_id_6|>";
    let mut questness = Questness::new(Aliasing::Strict).expect("builds");
    match questness.step(broken, false).expect("steps") {
        Step::Repair(envelope) => {
            assert!(!envelope.errors.is_empty(), "the failure is reported back");
        }
        other => panic!("a failed think is feedback, got {other:?}"),
    }
}

#[test]
fn insufficiency_is_the_callers_verdict() {
    let mut questness = Questness::new(Aliasing::Strict).expect("builds");
    assert!(matches!(
        questness.step("anything at all", true).expect("steps"),
        Step::Insufficient
    ));
}

#[test]
fn a_session_log_records_both_sides_of_the_turn() {
    let root = std::env::temp_dir().join("quest_arbitrate_log");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("scratch");
    let log =
        crate::session::SessionLog::open(&root, &["space", "session"]).expect("opens");

    let mut questness = Questness::new(Aliasing::Strict)
        .expect("builds")
        .logging_to(log.clone());
    questness.step(THINKS, false).expect("steps");

    let text = log.read().expect("reads");
    assert!(text.contains("== emission =="), "what the model wrote");
    assert!(text.contains("== thought =="), "what it was told back");
    let _ = std::fs::remove_dir_all(&root);
}
