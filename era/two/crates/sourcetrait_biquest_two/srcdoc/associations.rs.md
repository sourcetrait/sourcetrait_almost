# associations.rs

The ImagineQuestAssociations store builder. The foundation is
associative and DECLARED - every row's neighborhood has a citable
reason - and this store is the declaration the matrix build and
stage-zero corpus read, so neither re-derives from the dump or the
UCD.

The store is a directory, one class per file: NUON lines for the
three dump-side tables (spellings, definitions, inflections - the
millions-scale side, streamed so build memory stays flat and a
consumer can stream too), whole-value NUON for the UCD classes
(thousands-scale), one provenance record carrying every count. Code
points everywhere, never layer ids: layer ids bake the character
table's offsets into the artifact, while code points are the
version-stable address a consumer resolves against the same embedded
table.

## enum AbstractConcept

A purely conceptual association target (TheUser's class naming): an
embedding-space anchor with a keyword-page address and no wire
legality. The AbstractConceptMarker is the keyword id backing it -
for repetition, 0xFF `<|repetition|>`, which the lexer never emits
and never parses. The keyboard rows and the three repetition
operators associate to the concept: the association store is where
the convention lives once the wire no longer forces it.

## fn sanitize_gloss

Load-bearing, not cosmetic: the nuon renderer does NOT escape a raw
newline inside a string (verified by the tests), so an unsanitized
gloss would span lines and corrupt a NUON-lines file. Collapsing all
whitespace runs to single spaces makes every row single-line by
construction; `condensed_line` still guards, refusing a multi-line
rendering rather than writing it.

## struct DumpSide / fn absorb

One streamed pass, the dictionary pass's shape. The word set uses the
same `single_piece_folded` filter as `tokenizer dictionary`, which is
what makes `spellings.nuonl` exactly the vendored word set (verified
1,003,288 both). Definitions key (folded word, pos) and merge senses
across etymology-split entries; each sense keeps its LAST gloss,
because wiktextract glosses refine parent-to-child and the last is
the sense's own (the earlier elements repeat the parent sense).
Inflection links come from both directions the dump encodes: entry
`forms` (lemma-side) and sense `form_of` (form-side). A form_of lemma
can name an entry the dump never headwords, so written links are
post-filtered closed over the word set (190 dropped on the current
dump).

## fn condensed_line / fn write_lines

One hoisted EngineState for the whole build: the harness helpers
construct a fresh EngineState per render, which is fine for one
whole-value save and wrong for ~2.5M row renders. Raw (condensed)
style keeps the big files smallest.

## fn associations_build

UCD classes first (cheap, from the embedded table), then the dump
pass. ~8 s release over the 3.2 GB dump, ~185 MB written. The
provenance record carries every count the verification pass compares
against.
