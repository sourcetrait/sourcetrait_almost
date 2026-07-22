//! DepthProbe replay-core locks: round mechanics mirrored from
//! generate.rs (plain cost, echo accepts, the sample_len-vs-stop
//! final-round boundary), the policy gates, cost-model split, and
//! phase attribution. Synthetic streams; checkpoint-free.
use crate::*;
use crate::speculate::{
    COST_MODELS,
    PhaseRow,
    PhaseTally,
    PolicyKind,
    RecordedTranscript,
    RecordedTurn,
    replay_transcript,
};

fn transcript(kind: &str, emitted: Vec<u32>, finish: &str) -> RecordedTranscript {
    let consumed = if finish == "sample_len" {
        emitted.len().saturating_sub(1)
    } else {
        emitted.len()
    };
    RecordedTranscript {
        name: String::from("test"),
        kind: kind.to_string(),
        turns: vec![RecordedTurn {
            suffix_ids: vec![1001, 1002, 1003],
            emitted_ids: emitted,
            consumed,
            finish: finish.to_string(),
        }],
    }
}

fn totals(tallies: &HashMap<String, PhaseTally>) -> (u64, u64, u64, u64) {
    let mut tokens = 0;
    let mut passes = 0;
    let mut drafted = 0;
    let mut accepted = 0;
    for tally in tallies.values() {
        tokens += tally.tokens;
        passes += tally.passes;
        drafted += tally.drafted;
        accepted += tally.accepted;
    }
    (tokens, passes, drafted, accepted)
}

#[test]
fn novel_streams_cost_exactly_plain() {
    // All-distinct tokens: the index never matches, every round is a
    // plain step - tokens_per_pass is exactly 1.0 under BOTH models.
    let emitted: Vec<u32> = (100..140).collect();
    let recorded = transcript("prose_task", emitted, "stop_token");
    for cost in &COST_MODELS {
        let tallies = replay_transcript(&recorded, &[], PolicyKind::V1, cost);
        let (tokens, passes, drafted, _) = totals(&tallies);
        assert_eq!(tokens, 40);
        assert_eq!(passes, 40, "plain cost under {}", cost.name);
        assert_eq!(drafted, 0);
    }
}

#[test]
fn sample_len_skips_the_final_round_and_stop_pays_it() {
    // The budget check precedes the round in generate.rs: a
    // sample_len end never forwards the final emitted token; a
    // stop-token end does (that forward sampled the stop).
    let sample = transcript("prose_task", vec![101, 102, 103, 104], "sample_len");
    let tallies = replay_transcript(&sample, &[], PolicyKind::V1, &COST_MODELS[0]);
    let (tokens, passes, _, _) = totals(&tallies);
    assert_eq!((tokens, passes), (4, 3), "sample_len end: no final round");

    let stopped = transcript("prose_task", vec![101, 102, 103, 104], "stop_token");
    let tallies = replay_transcript(&stopped, &[], PolicyKind::V1, &COST_MODELS[0]);
    let (tokens, passes, _, _) = totals(&tallies);
    assert_eq!((tokens, passes), (4, 4), "stop end pays the final round");
}

#[test]
fn echo_streams_accept_and_beat_plain() {
    // Four repeats of one 8-token block: from the second block on the
    // tail matches earlier context and whole blocks verify in single
    // passes.
    let block: Vec<u32> = (21..29).collect();
    let emitted: Vec<u32> = block.iter().cycle().take(32).copied().collect();
    let recorded = transcript("prose_task", emitted, "stop_token");
    let two_pass = replay_transcript(&recorded, &[], PolicyKind::V1, &COST_MODELS[0]);
    let (tokens, passes_two, drafted, accepted) = totals(&two_pass);
    assert_eq!(tokens, 32);
    assert!(accepted > 0, "echo must accept drafts");
    assert!(drafted >= accepted);
    assert!(
        passes_two < tokens,
        "echo beats plain ({passes_two} passes for {tokens} tokens)"
    );
    let one_pass = replay_transcript(&recorded, &[], PolicyKind::V1, &COST_MODELS[1]);
    let (_, passes_one, _, _) = totals(&one_pass);
    assert!(
        passes_one <= passes_two,
        "kernel-accept never costs more than 2-pass"
    );
}

#[test]
fn cost_models_split_exactly_by_partial_rounds() {
    // Repeating 2-gram [1,2] with always-different continuations:
    // every draft rejects fully, so the models differ by exactly one
    // pass per drafting round.
    let emitted = vec![1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6];
    let recorded = transcript("prose_task", emitted, "stop_token");
    let two_pass = replay_transcript(&recorded, &[], PolicyKind::V1, &COST_MODELS[0]);
    let (tokens, passes_two, drafted, accepted) = totals(&two_pass);
    assert_eq!(tokens, 12);
    assert!(drafted > 0, "the 2-gram must fire under v1");
    assert_eq!(accepted, 0, "every continuation differs");
    let one_pass = replay_transcript(&recorded, &[], PolicyKind::V1, &COST_MODELS[1]);
    let (_, passes_one, _, _) = totals(&one_pass);
    assert!(passes_two > passes_one, "rejections cost double under v1");
    assert_eq!(passes_one, 12, "kernel-accept rejections read plain");
}

#[test]
fn no_2gram_structurally_suppresses_the_floor() {
    // The same stream carries ONLY 2-gram matches; no_2gram must
    // never draft and read exactly plain.
    let emitted = vec![1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6];
    let recorded = transcript("channel", emitted, "stop_token");
    let tallies = replay_transcript(&recorded, &[], PolicyKind::NoTwoGram, &COST_MODELS[0]);
    let (tokens, passes, drafted, _) = totals(&tallies);
    assert_eq!(drafted, 0, "level-2 hits are structurally excluded");
    assert_eq!(passes, tokens);
}

#[test]
fn phases_partition_the_stream() {
    let emitted: Vec<u32> = (200..208).collect();
    let recorded = transcript("backfill", emitted, "stop_token");
    let phases = vec![
        PhaseRow { turn: 0, start: 0, end: 4, phase: String::from("prose") },
        PhaseRow { turn: 0, start: 4, end: -1, phase: String::from("data") },
    ];
    let tallies = replay_transcript(&recorded, &phases, PolicyKind::V1, &COST_MODELS[0]);
    assert_eq!(tallies["prose"].tokens, 4);
    assert_eq!(tallies["data"].tokens, 4);
    let (tokens, passes, _, _) = totals(&tallies);
    assert_eq!(tokens, 8);
    assert_eq!(passes, 8, "novel tokens stay plain in every phase");
}

#[test]
fn unannotated_kinds_default_by_family() {
    let emitted: Vec<u32> = (300..306).collect();
    let prose = transcript("prose_interactive", emitted.clone(), "stop_token");
    let tallies = replay_transcript(&prose, &[], PolicyKind::V1, &COST_MODELS[0]);
    assert!(tallies.contains_key("prose"), "prose kinds default to prose");
    let data = transcript("channel", emitted, "stop_token");
    let tallies = replay_transcript(&data, &[], PolicyKind::V1, &COST_MODELS[0]);
    assert!(
        tallies.contains_key("unlabeled"),
        "non-prose kinds stay unlabeled until annotated"
    );
}
