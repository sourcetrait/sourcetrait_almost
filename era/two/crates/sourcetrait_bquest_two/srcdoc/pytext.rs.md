# pytext.rs

Python string semantics for the reference-faithful scorer ports.

The ifbench checkers and the nltk tokenizers are specified in Python, and their
behaviour leans on Python's own character classes and string methods rather than
on anything a Rust standard library offers. This module reproduces exactly those
semantics, which is what lets the ports read like their sources and the equality
gates hold.

One rule governs every function here: where Rust's own definition differs from
Python's, Python's wins, and the Rust one must not leak in. The regex crate's
`\w` is the specific trap - it is not Python's - so a ported pattern must never
use it and must reach for `py_is_word` instead.

## const PY_PUNCTUATION

## const PY_ASCII_WHITESPACE

Distinct from `py_is_space`, and both exist because upstream uses both. Punkt's
last-whitespace scan reaches for `string.whitespace`, which is ASCII-only, while
`str.isspace` and `\s` are Unicode-aware. Substituting one for the other changes
sentence boundaries on text carrying non-breaking spaces.

## fn py_is_space

## fn is_letter

## fn is_number_category

## fn py_is_word

The three number categories are all included, not just decimals, because
Python's `\w` matches letter-numbers and other-numbers too. Roman numerals and
circled digits therefore count as word characters here.

## fn py_is_alnum

## fn py_is_decimal

## fn py_is_word_nondigit

Punkt's own "alpha" notion, which is `[^\W\d]` - a word character that is not a
decimal digit. Underscore satisfies it, which is upstream's behaviour rather
than an oversight of ours.

## fn is_cased

## fn py_str_isupper

## fn py_str_islower

Both require at least one cased character, so a string of digits or punctuation
is neither upper nor lower. That is Python's rule and it is load-bearing for the
title-case checker, which relies on the empty case falling through rather than
passing.

## fn py_str_isdigit

Narrowed to decimal precision deliberately. Python's `str.isdigit` also accepts
superscript forms, which the exercised texts do not carry, and matching it
exactly would mean carrying a second character class for no measured difference.
Recorded rather than silent, because it is the one place here that is narrower
than its source.

## fn py_strip_chars

## fn py_lstrip_chars

## fn py_rstrip_chars

## fn py_strip_ws

## fn py_lstrip_ws

## fn py_split_ws

Empties are dropped, which is what `str.split()` with no argument does and what
`str.split(' ')` does not. A checker counting words depends on the difference.

## fn py_delete_chars

## fn py_count

The empty needle returns one more than the character count, reproducing
Python's `str.count("")`. Nothing here passes an empty needle, but a checker
reading a keyword out of its kwargs could, and diverging silently there would
be worse than carrying the branch.

## fn py_boundary_search

A hand-scanned `\b<literal>\b`, needed because the needles arrive as data rather
than as patterns and escaping them into a regex per call would be both slower
and easier to get wrong.

The boundary test is the subtle half. A `\b` sits between a word character and a
non-word character, so the test at each edge is that the needle's edge character
and its neighbour differ in wordness. The needles here are keywords and names,
whose edges are word characters, so in practice the neighbour must not be one -
but the condition is written in the general form because a needle carrying
punctuation at its edge would otherwise never match.

Case-insensitivity is ASCII-only, matching the reference's own use.

## fn nfkd_ascii

The checkers' `normalize('NFKD', s).encode('ASCII', 'ignore')` fold, which is
how an accented capital compares equal to its plain form. Decomposition first is
what makes the ASCII filter lossless for Latin text: the accent becomes a
separate combining character and only that is dropped.

## fn py_word_run_count

## fn py_word_runs

Two functions over one traversal because the reference has two call sites and
one of them needs the tokens rather than the count. Counting via `py_word_runs`
would allocate a string per token for a number.
