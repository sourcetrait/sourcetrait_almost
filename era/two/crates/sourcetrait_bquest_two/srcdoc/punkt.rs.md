# punkt.rs

The nltk Punkt sentence tokenizer, inference path only, ported against nltk
3.10.0's `punkt.py` with the shipped punkt_tab english parameters. Training is
absent because we consume the shipped parameters rather than learning any.

Four faithfulness properties decide whether this agrees with the reference.
Every index is a Python string index, meaning codepoints and never bytes. The
period-context and word-tokenizer patterns are hand-scanned equivalents of the
`PunktLanguageVars` regexes, using Python's `\s` through `pytext`. The
last-whitespace scan uses ASCII `string.whitespace`, which is a different set
from `\s` and is upstream's choice at that one site. And boundary realignment
reproduces the lazy quote-run match including its multiline `$`.

It passes upstream's own doctests, the quoted-realignment example included,
which is the evidence that the hand-scanned patterns are equivalent rather than
merely similar.

## const ORTHO_BEG_UC

## const ORTHO_MID_UC

## const ORTHO_UNK_UC

## const ORTHO_BEG_LC

## const ORTHO_MID_LC

## const ORTHO_UNK_LC

## const ORTHO_UC

## const ORTHO_LC

The orthographic context flags, as upstream's bit positions. The composite
masks are sums rather than bitwise ors, which is upstream's spelling and equals
an or here because the members are disjoint.

## const NON_WORD_CHARS

## const WORD_START_EXCLUDED

## const SENT_PUNCTUATION

## const REALIGN_CHARS

Four separate character classes rather than one, because upstream treats them
separately and their memberships genuinely differ. Merging any pair changes
where a sentence ends.

## struct PunktParams

## fn load

The tabdata forms as shipped. A trailing empty line is dropped rather than
becoming an empty abbreviation, which would then match every token ending in a
period.

## struct PunktTok

## fn new

The type is the lowercased token, with a numeric token collapsing to a single
`##number##` type. That collapse is what makes the collocation and sent-starter
lookups generalise across numbers instead of memorising them.

## fn type_no_period

Slices bytes off the type string, which is safe only because the character being
removed is an ASCII period. Every other index in this file is a codepoint index.

## fn type_no_sentperiod

## fn first_char

## fn first_upper

## fn first_lower

## fn is_ellipsis_tok

## fn is_initial

## fn numeric_type_matches

The pattern is fully anchored, so this tests the whole token rather than
searching it.

The optional leading `[.,]` is consumed only when a digit follows, which
reproduces the regex's backtracking rather than approximating it. Consuming it
unconditionally would accept a bare comma as numeric.

## fn multi_char_len

The three alternatives in upstream's order, and the third needs real
backtracking: `(?:\.\s){2,}\.` matches a spaced ellipsis, greedily, and then
gives back repetitions until the trailing period is present. The loop counts the
repetitions first and then walks them down, which is what that greediness plus
backtracking does.

## fn is_non_word_char

## fn word_end_lookahead

The comma branch is a second level of lookahead rather than a repeat of the
first: a comma ends a word only when what follows it also ends a word, so
`1,000` stays one token while `foo, bar` splits.

## fn punkt_word_tokenize

## fn tokenize_words

The inference shape of `_tokenize_words`. Upstream also sets line-start and
paragraph-start flags on each token, and those feed training alone, so they are
absent here rather than carried unused.

## fn first_pass_annotation

## fn ortho_heuristic

Three outcomes rather than two, and the `None` is not a failure: it means the
orthographic evidence is inconclusive, and the caller then falls through to a
different test. Collapsing it to false would change which abbreviations end
sentences.

## fn second_pass_annotation

Takes an index into a mutable slice rather than a pair of references, because it
reads the following token while writing the current one and the borrow checker
will not have both.

## fn text_contains_sentbreak

The loop returns true only when a break is followed by another token, which is
upstream's "any non-final" reading. A break on the last token is the ordinary
end of the text rather than evidence that the context contains one.

## struct PeriodMatch

## fn period_context_matches

## fn slice_text

## fn last_ascii_whitespace_index

Returns zero both when no whitespace was found and when it sits at the start of
the range, and the caller distinguishes those by testing for zero. That
conflation is upstream's, and the caller's branch is written to match it.

## struct EndContext

## fn match_potential_end_contexts

The overlap pruning holds one match back and emits it on the next iteration, so
a match whose preceding word slice overlaps the previous one is dropped. That is
what keeps two adjacent sentence ends from each claiming the same word.

## fn slices_from_text

## fn realign_match

The realignment pattern is lazy, so it prefers the shortest quote run that can
satisfy what follows. The loop tries run lengths in increasing order for exactly
that reason, and returns on the first that works rather than the longest.

## fn realign_boundaries

Carries the realignment offset forward across iterations, which is how a closing
quote moves from the head of one sentence to the tail of the previous one.

## fn sent_tokenize
