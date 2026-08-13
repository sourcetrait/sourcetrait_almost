# bucket.rs

The bucket layer exists so every designed sequence token carries its
reason in the type system - TheUser's ruling after probing dolma2's
86 unexplainable pure-space rows: in our codebase, "where did this
come from" is answered in code. Each bucket enum is a category of
motive; each variant is one documented decision.

Buckets hold only semantic base units. Magnitude - indent depth, line
length, run width - rides the REPEAT operator (lexer.rs) instead of
depth enumerations, TheUser's correction of the first cut: enumerated
depth families are the same disease as BPE's fossil rows, just
smaller. His second calculation, ledger-confirmed: beyond the SS pair
nothing repetition-shaped earns a row except `    ` and `...` - the
quads were removed and the corpus measured very slightly BETTER
(pair-plus-REPEAT covers the 4-8 zone at one extra token, and long
rule lines are too rare to pay for rows). What stays enumerated is
what is semantic rather than magnitudinal: heading levels 2-6 are
distinct meanings, comment openers are distinct constructs. `**`
claims one id and serves both markdown bold and the comment-star
pair; `__` lives in the ascii family though markdown also uses it -
one sequence, one id, the category records the primary reason.

## fn Bucket::all

The canonical enumeration IS the in-layer id order, append-only
forever - the same discipline as the character layer's dense index.
Adding an entry appends; nothing renumbers.

## struct BucketTable

### fn match_at

First-character index, longest sequence first, so `/**` beats `/*`
and the four-space unit beats the two-space at an eight-space run.
All sequences are ASCII, which is what makes the byte-length walk in
the lexer boundary-safe.
