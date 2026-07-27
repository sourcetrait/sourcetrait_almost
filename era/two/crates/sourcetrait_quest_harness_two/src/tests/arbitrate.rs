//! Arbitration locks: the whole turn loop, against a stub harness.
use sourcetrait_quest_core::channel::Aliasing;
use sourcetrait_quest_core::harness::{
    ClientHarness,
    HarnessRequest,
    HarnessResponse,
};
use sourcetrait_quest_core::nu;
use sourcetrait_quest_core::QuestCoreResult;
use crate::questness::arbitrate::{
    Questness,
    Step,
};

fn value(nuon: &str) -> nu::Value {
    nu::from_nuon_text(nuon).expect("fixture parses")
}

/// A client harness whose world is one record, modifiable between turns.
///
/// The fixture is the WORLD rather than the response: the test sets the
/// state and the answer follows from it, so a changed setup cannot leave
/// a stale answer behind still looking plausible.
struct World {
    state: nu::Value,
    served: Vec<String>,
}

impl World {
    fn new(nuon: &str) -> Self {
        Self {
            state: value(nuon),
            served: Vec::new(),
        }
    }
}

impl ClientHarness for World {
    fn serve(&mut self, request: &HarnessRequest) -> QuestCoreResult<HarnessResponse> {
        self.served.push(request.mode.clone());
        Ok(HarnessResponse::value(self.state.clone()))
    }
}

/// The model asks for something it was not given.
const ASKS: &str = "\
<|extra_id_4|> def interact []: nothing -> record<cwd: string> { {cwd: ''} }
<|extra_id_6|>";

/// The model answers, having been told.
const ANSWERS: &str = "\
<|extra_id_2|> record<cwd: string>
{cwd: '/tmp/foo'}
<|extra_id_6|>
<|extra_id_3|>$in<|extra_id_6|>
<|extra_id_5|>
We are currently in the {{ in.cwd }} directory.
<|extra_id_6|>";

#[test]
fn a_client_sub_turn_comes_back_as_a_typed_output_block() {
    let mut questness =
        Questness::new(World::new("{cwd: '/tmp/foo'}"), Aliasing::Strict).expect("builds");

    let Step::Continue(back) = questness.step(ASKS, false).expect("steps") else {
        panic!("a client mode should have been served");
    };
    assert!(
        back.starts_with("<|extra_id_2|> record<cwd: string>"),
        "the result returns TYPED, through output: {back}"
    );
    assert!(back.contains("{cwd: /tmp/foo}"));
    assert!(back.trim_end().ends_with("<|extra_id_6|>"));
}

#[test]
fn the_worked_example_differs_only_in_the_value_it_was_told() {
    for (world, expected) in [
        ("{cwd: '/tmp/foo'}", "We are currently in the /tmp/foo directory."),
        ("{cwd: '/srv/other'}", "We are currently in the /srv/other directory."),
    ] {
        let mut questness = Questness::new(World::new(world), Aliasing::Strict).expect("builds");

        let Step::Continue(back) = questness.step(ASKS, false).expect("steps") else {
            panic!("expected a sub-turn to be served");
        };
        // The model, having been told, finishes the turn. It echoes the
        // value it was handed, which is what the design's example does.
        let told = back
            .lines()
            .nth(1)
            .expect("the block carries a payload")
            .to_string();
        let finished = ANSWERS.replace("{cwd: '/tmp/foo'}", &told);

        let Step::Answered(answer) = questness.step(&finished, false).expect("steps") else {
            panic!("expected the turn to finish");
        };
        assert_eq!(answer.rendered.trim(), expected);
    }
}

#[test]
fn changing_the_world_changes_the_response() {
    let mut first = Questness::new(World::new("{cwd: '/a'}"), Aliasing::Strict).expect("builds");
    let mut second = Questness::new(World::new("{cwd: '/b'}"), Aliasing::Strict).expect("builds");

    let Step::Continue(a) = first.step(ASKS, false).expect("steps") else {
        panic!("served");
    };
    let Step::Continue(b) = second.step(ASKS, false).expect("steps") else {
        panic!("served");
    };
    assert_ne!(a, b, "the same request against two worlds must differ");
    assert!(a.contains("/a"));
    assert!(b.contains("/b"));
}

#[test]
fn an_inside_mode_never_reaches_the_client() {
    let inside = "\
<|extra_id_2|> list<string>
[a, b, c]
<|extra_id_6|>
<|extra_id_3|>$in<|extra_id_6|>
<|extra_id_4|> def evaluate []: list<string> -> int { $in | length }
<|extra_id_6|>";
    let mut questness =
        Questness::new(World::new("{cwd: '/tmp'}"), Aliasing::Strict).expect("builds");

    let Step::Continue(back) = questness.step(inside, false).expect("steps") else {
        panic!("expected the inside path to run");
    };
    assert!(back.contains('3'), "the evaluator's own answer: {back}");
    assert!(
        back.starts_with("<|extra_id_2|> int"),
        "and it returns typed too: {back}"
    );
}

#[test]
fn a_failing_client_becomes_feedback_rather_than_an_error() {
    struct Refuses;
    impl ClientHarness for Refuses {
        fn serve(&mut self, _request: &HarnessRequest) -> QuestCoreResult<HarnessResponse> {
            Ok(HarnessResponse::failed(
                "harness::denied",
                "interact is not built",
            ))
        }
    }
    let mut questness = Questness::new(Refuses, Aliasing::Strict).expect("builds");
    let Step::Repair(envelope) = questness.step(ASKS, false).expect("steps") else {
        panic!("a refusal is feedback, not a step forward");
    };
    assert_eq!(envelope.errors.len(), 1);
    assert_eq!(envelope.errors[0].kind, "harness::denied");
}

#[test]
fn a_plain_answer_never_touches_the_harness() {
    let mut questness =
        Questness::new(World::new("{cwd: '/tmp/foo'}"), Aliasing::Strict).expect("builds");
    let Step::Answered(answer) = questness.step(ANSWERS, false).expect("steps") else {
        panic!("expected an answer");
    };
    assert_eq!(
        answer.rendered.trim(),
        "We are currently in the /tmp/foo directory."
    );
}
