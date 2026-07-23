use crate::speculate::{DraftPolicy, LookupIndex, MAX_DRAFT};

#[test]
fn draft_finds_earlier_continuation() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2, 3, 9, 8, 7, 1, 2, 3]);
    // No 4-gram tail recurs; the 3-gram [1,2,3] first continued at
    // position 3.
    assert_eq!(index.draft(8), Some((3, vec![9, 8, 7, 1, 2, 3])));
}

#[test]
fn draft_respects_limit() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2, 3, 9, 8, 7, 1, 2, 3]);
    assert_eq!(index.draft(2), Some((3, vec![9, 8])));
    assert_eq!(index.draft(0), None);
}

#[test]
fn draft_none_when_tail_unseen() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2, 3, 4]);
    assert_eq!(index.draft(8), None);
}

#[test]
fn draft_none_below_min_gram() {
    let mut index = LookupIndex::new();
    index.extend(&[1]);
    assert_eq!(index.draft(8), None);
}

#[test]
fn draft_uses_previous_occurrence_when_tail_owns_latest() {
    let mut index = LookupIndex::new();
    // Every gram recurs; the tail's own insertion holds `latest`, so
    // drafting falls back to the previous continuation - via the
    // 4-gram level now.
    index.extend(&[5, 5, 5, 5, 5]);
    assert_eq!(index.draft(8), Some((4, vec![5])));
}

#[test]
fn draft_tracks_incremental_extends() {
    let mut index = LookupIndex::new();
    index.extend(&[10, 11, 12, 13]);
    index.extend(&[10, 11, 12]);
    // Tail [10,11,12] matches the opening gram, continuation 13
    // onward.
    assert_eq!(index.draft(8), Some((3, vec![13, 10, 11, 12])));
}

#[test]
fn four_gram_outranks_shorter_matches() {
    let mut index = LookupIndex::new();
    // The tail [6,7,8,9] recurs as a 4-gram (continuing 42, 41); a
    // decoy 3-gram [7,8,9] occurrence continues with 13 instead.
    // Longest match wins.
    index.extend(&[6, 7, 8, 9, 42, 41, 7, 8, 9, 13, 6, 7, 8, 9]);
    assert_eq!(index.draft(2), Some((4, vec![42, 41])));
}

#[test]
fn two_gram_floor_catches_short_repeats() {
    let mut index = LookupIndex::new();
    // Only the 2-gram tail [3,4] recurs; its earlier continuation
    // starts at 50.
    index.extend(&[3, 4, 50, 51, 52, 9, 3, 4]);
    assert_eq!(index.draft(8), Some((2, vec![50, 51, 52, 9, 3, 4])));
}

#[test]
fn policy_starts_at_max_shrinks_on_misses_tracks_accept_depth() {
    let mut policy = DraftPolicy::new();
    assert_eq!(policy.probe_limit(), MAX_DRAFT);
    policy.record(0);
    policy.record(0);
    assert_eq!(policy.probe_limit(), MAX_DRAFT / 4);
    // Tracking: the ceiling follows twice the observed accept depth.
    policy.record(1);
    assert_eq!(policy.probe_limit(), 2);
    policy.record(3);
    assert_eq!(policy.probe_limit(), 6);
    policy.record(MAX_DRAFT);
    assert_eq!(policy.probe_limit(), MAX_DRAFT);
}

#[test]
fn policy_pauses_after_cold_streak_and_resumes() {
    let mut policy = DraftPolicy::new();
    // A cold streak halves to the floor and trips the pause.
    let mut rounds = 0;
    while policy.probe_limit() > 0 {
        policy.record(0);
        rounds += 1;
        assert!(rounds < 32, "policy never paused");
    }
    // 64-round pause, then cheap probing resumes at the floor.
    for _ in 0..63 {
        assert_eq!(policy.probe_limit(), 0);
    }
    assert_eq!(policy.probe_limit(), 1);
}

#[test]
fn policy_seeds_length_by_matched_level() {
    let mut policy = DraftPolicy::new();
    // Hot (ceiling at MAX_DRAFT): every level drafts full.
    assert_eq!(policy.draft_limit(4), MAX_DRAFT);
    assert_eq!(policy.draft_limit(2), MAX_DRAFT);
    // Off the max: 3+ carries the ceiling, the 2-gram floor is short.
    policy.record(0);
    assert_eq!(policy.draft_limit(3), MAX_DRAFT / 2);
    assert_eq!(policy.draft_limit(2), 2);
    policy.record(2);
    assert_eq!(policy.draft_limit(3), 4);
}
