# bucket.rs

The KeyboardSymbolBucket, TheUser's three-method repetition design's
row half. Every keyboard symbol - tab, space, and the 32 printable
ASCII symbols - gets a double and a triple row, because keyboard
characters are where repetition happens most. The rule is closed and
uniform: membership is "is a keyboard symbol", never a frequency
pick, so every row's existence is derivable and documented - the
inverse of BPE's fossil rows, preserved from the first bucket
experiment's motive.

Rows serve exact run lengths two and three only, never composing
with each other or the operators (the encoder's magnitude bands in
lexer.rs own everything longer). The (symbol, count) pair on each
entry IS the parameter association: at matrix time each row
associates to its character row, the `<|repetition|>` concept
marker, and its count - the association layer carrying the
convention the wire no longer forces.

## fn BucketTable::new

Canonical order: characters ascending (tab, space, then the symbols
by code point), double before triple. Entry position is the in-layer
id, append-only forever.
