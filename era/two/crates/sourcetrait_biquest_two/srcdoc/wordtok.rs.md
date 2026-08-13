# wordtok.rs

nltk's `NLTKWordTokenizer` (`destructive.py`, nltk 3.10.0): the exact
regex-substitution battery, in the exact order. `nltk.word_tokenize` is a punkt
sentence split followed by this per sentence, which is why both live here.

The battery is order-dependent throughout - each rule consumes what the previous
one produced - so the sequence in `treebank_tokenize` is the specification and
reordering it for readability would change its output.

Two rules carry lookaheads, which the Rust regex engine has no support for, and
both are hand-scanned instead. Those are the two places where a divergence from
upstream would be invisible in a diff of the patterns, so they are the two worth
reading closely.

## macro_rules rules

A macro rather than a list because each rule needs its own `LazyLock` static,
and writing thirty of those out by hand is where a copy-paste error lives.

## const PYS

Python's `\s` spelled out as a regex class, because the reference patterns use
`\s` and the Rust engine's own class is not the same set. Interpolating this
rather than writing `\s` is what keeps the ported patterns faithful.

Two patterns spell the same class out inline rather than interpolating it. That
is deliberate: they are raw strings carrying other escapes, and interpolating
into them would need `format!` at every use.

## fn split_clitic_quotes

The starting-quotes rule with a negative lookahead: a quote before a
single-letter word splits, unless what follows is one of the contraction
suffixes that a later rule owns. The blocked list is why `'tis` survives to be
handled as a contraction rather than being split here.

The word-boundary check reads two characters ahead, so a single-character word
at the very end of the input satisfies it by running out of input. That matches
the regex's own behaviour at end-of-string.

## enum Tail

## fn contraction_replace

One MacIntyre contraction rule, hand-scanned because the pattern carries a
lookahead on its right edge.

The replacement always emits exactly one leading space whether or not the
pattern consumed one, which is upstream's `" \1 \2 "`. Emitting the consumed
space instead would leave contractions unseparated from the preceding word in
the leading-space variants.

The left-edge condition is a `\b`, and it applies only in the no-leading-space
form. The leading space is itself a boundary, so testing for one in that form
would reject every match.

Matching is ASCII-case-insensitive against a lowercase needle while the output
preserves the input's case, which is what `(?i)` plus captured groups does.

## fn treebank_tokenize

The rules run in upstream's order, and the padding to a space-surrounded string
before the ending rules is part of it: several of those patterns anchor on a
trailing space that the padding guarantees.

## fn nltk_word_tokenize
