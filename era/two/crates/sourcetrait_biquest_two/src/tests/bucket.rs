//! Bucket locks: the canonical enumeration and longest-first matching.
use crate::bucket::BucketTable;

#[test]
fn canonical_enumeration_is_stable() {
    let table = BucketTable::new();
    assert_eq!(table.count(), 20);
    let (first_sequence, first_bucket) = table.entry(0);
    assert_eq!(first_sequence, "  ");
    assert_eq!(first_bucket.category(), "indent");
    let (last_sequence, last_bucket) = table.entry(19);
    assert_eq!(last_sequence, "*/");
    assert_eq!(last_bucket.category(), "comment");
}

#[test]
fn match_at_prefers_the_longest_sequence() {
    let table = BucketTable::new();
    let (index, length) = table.match_at("/** rest").expect("matches");
    assert_eq!(length, 3);
    assert_eq!(table.entry(index).0, "/**");
    let (index, length) = table.match_at("        x").expect("matches");
    assert_eq!(length, 4, "the four-space unit is the longest space entry");
    assert_eq!(table.entry(index).0, "    ");
    let (index, length) = table.match_at("----------").expect("matches");
    assert_eq!(length, 4);
    assert_eq!(table.entry(index).0, "----");
    assert!(table.match_at("x").is_none());
    assert!(table.match_at(" x").is_none(), "a single space is a character");
}
