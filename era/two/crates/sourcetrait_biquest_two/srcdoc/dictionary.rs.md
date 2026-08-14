# dictionary.rs

## fn tokenizer_dictionary

One streaming pass over the wiktextract English dump (3.2 GB
decompressed, ~1M entries). The typed serde structs skip everything
but word, lang_code, and forms, which keeps the parse allocation-lean;
`json_to_value` is deliberately not on this path.

The single-piece admission filter is the load-bearing choice: a
candidate must lex as exactly one Word piece under the character
table, so multiword entries ("give up"), hyphenations ("well-being"),
and apostrophe forms ("don't") are excluded - the lexer can never
match a token across a boundary, so a row for one would be
unreachable. The filter (`single_piece_folded`) is shared with the
associations pass, which is what keeps the store's word set
identical to this artifact's. Those entries decompose at lex time instead, by design.
Inflected forms are admitted as their own words ("dogs" beside "dog"),
which is what lets the segmenter resolve whole pieces with no
morphology layer.

The artifact is plain sorted lines rather than NUON: a millions-scale
word set is a bulk intermediate on the tmp tier, and line format loads
in either language at full speed. The admitted wordlist that census
admission produces from it is the real, NUON artifact.

## const ENGLISH_WORDS / fn embedded_words

The extracted set is also vendored at `data/words/english.txt`
(10.7 MB, CC BY-SA; attribution at the repository's
docs/licenses/wiktionary/) and embedded, as the bare `tokenize`
command's default dictionary - TheUser's ruling that the test surface
must show text as it WOULD tokenize with a dictionary. Its ids (alphabetical order above the character layer)
are the test surface's own; a trained vocabulary's ids come from an
--admitted artifact instead. Re-vendor by rerunning `tokenizer
dictionary` over a fresh dump.

## fn read_words

The counterpart loader; empty is an error because an empty dictionary
would silently turn the whole corpus into character soup.
