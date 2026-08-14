//! Keyboard-symbol bucket locks: the canonical enumeration and the
//! (char, count) index.
use crate::bucket::BucketTable;
use crate::bucket::KEYBOARD_CHARS;

#[test]
fn canonical_enumeration_is_stable() {
    let table = BucketTable::new();
    assert_eq!(table.count(), KEYBOARD_CHARS.len() * 2);
    assert_eq!(table.count(), 68);
    // Per character ascending, double then triple: tab leads.
    let first = table.entry(0);
    assert_eq!((first.symbol, first.count), ('\t', 2));
    let second = table.entry(1);
    assert_eq!((second.symbol, second.count), ('\t', 3));
    let last = table.entry(67);
    assert_eq!((last.symbol, last.count), ('~', 3));
    assert_eq!(table.entry(2).sequence(), "  ");
    assert_eq!(table.entry(3).sequence(), "   ");
}

#[test]
fn index_for_resolves_doubles_and_triples_only() {
    let table = BucketTable::new();
    let space_double = table.index_for(' ', 2).expect("space double");
    assert_eq!(table.entry(space_double).sequence(), "  ");
    let dot_triple = table.index_for('.', 3).expect("dot triple");
    assert_eq!(table.entry(dot_triple).sequence(), "...");
    assert!(table.index_for(' ', 4).is_none(), "only 2 and 3 exist");
    assert!(table.index_for('a', 2).is_none(), "letters are not keyboard symbols");
    assert!(table.index_for('0', 2).is_none(), "digits are not keyboard symbols");
}
