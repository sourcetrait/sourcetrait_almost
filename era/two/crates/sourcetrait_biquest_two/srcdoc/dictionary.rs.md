# dictionary.rs

## fn tokenizer_dictionary

One streaming pass over the wiktextract English dump (3.2 GB
decompressed, ~1M entries). The typed serde structs skip everything
but word, lang_code, and forms, which keeps the parse allocation-lean;
`json_to_value` is deliberately not on this path.

The connected-candidate admission is the load-bearing choice: an
entry is admissible when its text is exactly ONE connected candidate
- word runs joined by internal hyphen and apostrophe connectors,
with at most one edge apostrophe per side - so "don't", "x-ray",
"'tis", and "fo'c's'le" are rows while multiword entries ("give up")
stay structurally unreachable, since a space is never a connector.
The only removal beyond reachability is the single-code-point rule
(TheUser: dictionary entries never match Unicode code points) - the
character layer already holds that row, a duplicate would split its
training mass, and the token count is identical either way. This is
also what keeps numerals like "2" out of the dictionary, so digit
pieces char-split. The filter (`whole_candidate_folded`) is shared
with the associations pass, which is what keeps the store's word set
identical to this artifact's. Inflected forms are admitted as their
own words ("dogs" beside "dog"), which is what lets the segmenter
resolve whole pieces with no morphology layer.

The artifact is plain sorted lines rather than NUON: a millions-scale
word set is bulk on the tmp tier, line format loads in either
language at full speed, and with the whole dictionary as the
vocabulary its line order IS the dictionary layer's id order.
`read_words_ordered` is the counterpart loader; empty is an error
because an empty dictionary would silently turn the whole corpus
into character soup.

## const ENGLISH_WORDS / fn embedded_words

The extracted set is also vendored at `data/words/english.txt`
(10.7 MB, CC BY-SA; attribution at the repository's
docs/licenses/wiktionary/) and embedded, as the bare `tokenize`
command's default dictionary - TheUser's ruling that the test surface
must show text as it WOULD tokenize with a dictionary. The embedded
set IS the vocabulary; --words swaps in another word file, file
order as id order. Re-vendor by rerunning `tokenizer dictionary`
over a fresh dump.
