use crate::speculate::LookupIndex;

#[test]
fn draft_finds_earlier_continuation() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2, 3, 9, 8, 7, 1, 2, 3]);
    // The tail gram [1,2,3] first continued at position 3.
    assert_eq!(index.draft(8), Some(vec![9, 8, 7, 1, 2, 3]));
}

#[test]
fn draft_respects_budget() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2, 3, 9, 8, 7, 1, 2, 3]);
    assert_eq!(index.draft(2), Some(vec![9, 8]));
    assert_eq!(index.draft(0), None);
}

#[test]
fn draft_none_when_tail_unseen() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2, 3, 4]);
    assert_eq!(index.draft(8), None);
}

#[test]
fn draft_none_below_ngram() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2]);
    assert_eq!(index.draft(8), None);
}

#[test]
fn draft_uses_previous_occurrence_when_tail_owns_latest() {
    let mut index = LookupIndex::new();
    // Gram [5,5,5] recurs; the tail's own insertion holds `latest`, so
    // drafting must fall back to the previous continuation.
    index.extend(&[5, 5, 5, 5, 5]);
    assert_eq!(index.draft(8), Some(vec![5]));
}

#[test]
fn draft_tracks_incremental_extends() {
    let mut index = LookupIndex::new();
    index.extend(&[10, 11, 12, 13]);
    index.extend(&[10, 11, 12]);
    // Tail [10,11,12] matches the opening gram, continuation 13 onward.
    assert_eq!(index.draft(8), Some(vec![13, 10, 11, 12]));
}
