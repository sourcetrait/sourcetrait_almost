# checker_data.rs

The reference checker data, loaded from the capability home's `checker_data`
directory. Every file there is a byte copy of the evaluation environment's own:
punkt_tab english, the averaged-perceptron tagger JSONs, the nltk english
stopwords, syllapy's word dictionary, and the emoji package's `emoji.json`.

Nothing is re-authored, and that is the whole point. The Rust code parses their
formats as shipped, so refreshing the data is a re-copy rather than a
translation pass, and there is no second representation to drift.

## struct CheckerData

One struct rather than five, because the ifeval verifiers reach for several of
these within a single check and loading them per call would re-read the tagger's
weights per item.

## fn load

Every file is required. A checker whose data is absent would silently pass or
silently fail depending on the check, so the load is where the absence surfaces.

The emoji map is filtered to single-codepoint keys. The package's own keys include
multi-codepoint sequences with modifiers and zero-width joiners, and the
verifier that consumes this tests one character at a time - so a sequence key
could never match and keeping it would only grow the set.

## fn syllable_count

Reproduces `syllapy.count` including its order: the dictionary first, then a
compound split on the first hyphen, then the vowel-group heuristic. The
dictionary lookup happens on the stripped and lowercased word, so the
punctuation strip is load-bearing rather than tidying.

A word containing any decimal digit returns zero rather than falling through to
the heuristic, which is upstream's behaviour and matters because the
syllable-parity verifier then counts it as even.

The compound branch reproduces the `([^-]+)-(.+)` match, so the leading part is
hyphen-free and the tail may itself compound recursively. Either part scoring
zero collapses the whole word to zero, which is again upstream rather than a
guard of ours.

## fn syllable_heuristic

`syllapy._syllables`, and the arithmetic is theirs including the quirks: a
trailing `e` subtracts, a trailing `le` after a consonant adds it back, and a
count that lands at zero is forced to one. `y` counts as a vowel throughout.
