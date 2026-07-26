# tagger.rs

nltk's averaged-perceptron part-of-speech tagger (`perceptron.py`, nltk 3.10.0),
inference for the first token only. That is all the ifeval port needs, because
the one checker that reaches for a tag reads `pos_tag(tokens)[0]`.

Four faithfulness details decide whether this agrees with the reference, and
each is a place a reasonable re-implementation would differ. The tagdict
shortcut keys on the raw word rather than the normalized one. Features
accumulate in insertion order. The weight maps iterate in JSON file order, which
is why `serde_json`'s `preserve_order` feature is load-bearing workspace-wide.
And prediction takes the maximum of the pair `(score, label)` over the class
list, exactly as Python's `max` does, so ties break on the label string.

## struct PerceptronTagger

The weights stay a `serde_json::Map` rather than becoming a typed structure,
because the file order is part of the contract and a `HashMap` would lose it.

## fn load

## fn normalize

Upstream's own normalization, and the order of its branches is significant: a
hyphenated word is `!HYPHEN` before a four-digit token can be `!YEAR`, and a
leading digit is `!DIGITS` only after both.

## fn suffix3

Codepoints rather than bytes, matching Python's `word[-3:]`. Slicing bytes here
would panic on the multibyte words this data contains.

## fn tag_first

The context vector is padded with two start sentinels and two end sentinels, so
the feature offsets index into a padded list and `i` is 2 rather than 0. Reading
those offsets as absolute positions is the way to get this subtly wrong.

Only the first token is predicted, and that is sound rather than a shortcut: the
previous-tag features for position zero are the start sentinels, so no later
prediction can feed back into it. A full tagger would thread each prediction
into the next token's features, which is exactly the part that is absent.

Features with a zero value are skipped before the lookup. Nothing here produces
one, since the accumulator only ever increments, but the reference has the guard
and a divergence in behaviour is worth less than a matching shape.

An unknown feature contributes nothing rather than erroring, because the weight
file genuinely does not carry every feature the extractor can form. A
non-numeric weight, in contrast, raises - that would be a corrupt file.
