# value.rs

## fn json_to_value / fn value_to_json

Carried over from bquest's convert module. Losslessness is the point:
serde_json runs with `float_roundtrip` and `preserve_order`, so floats
and key order survive both directions, and a u64 past i64 refuses
rather than truncating. The unit locks in `tests/value.rs` pin the
round-trip, and they are the reason the pair stays in the crate while
nothing in the tokenizer path calls it yet: the wiktextract parse reads
through typed serde structs instead, which skips the Value allocation
on a 3 GB stream.

## fn span / v_int / v_float / v_str / v_bool / field family

The record-building and record-reading shorthand every verb summary
and artifact uses. `span()` is `Span::unknown` because nothing here
carries source positions.
