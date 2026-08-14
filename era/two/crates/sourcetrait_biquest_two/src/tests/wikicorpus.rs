use crate::wikicorpus::shard_of;

#[test]
fn shards_are_two_sanitized_characters() {
    assert_eq!(shard_of("abandon"), "ab");
    assert_eq!(shard_of("Zürich"), "z_");
    assert_eq!(shard_of("don't"), "do");
    assert_eq!(shard_of("x-ray"), "x_");
    assert_eq!(shard_of("'tis"), "_t");
    assert_eq!(shard_of("ab"), "ab");
}
