# lexer.rs

The Quill content lexer. Lexer-style, never BPE: nothing merges across
a class edge, every piece is explicit, and the two-level rule is
absolute for words - dictionary hit or character split, nothing
between. The morphology middle a BPE would learn is deliberately
absent; the expansion loop (a frequent unknown earns a row) replaces
it.

## the operator block and the id layout

Each offset a constant forever: ids 0x00-0xFF are the keyword page
(reserved whole - the page belongs to Quill, not the lexer), the
character layer starts at 256 in the table's dense order, the
keyboard-symbol rows sit above it, and the dictionary above those in
admitted-wordlist order.

The page allocates from both ends (TheUser): user bindings from 0x00
up (the Syntax.nuon draft), the hardcoded operator block from 0xFF
down - 0xFC BEGIN_REPEAT, 0xFD END_REPEAT, 0xFE REPEAT, 0xFF
REPETITION. The first three are legal wire both directions - the
model may speak them (TheUser). REPETITION is the
AbstractConceptMarker backing the repetition AbstractConcept
(associations.rs): illegal in wire, never emitted, never parsed - it
exists so the association store can anchor every repetition
mechanism to one concept ("hint at convention by association").
Ones-fill corruption decodes as the wire-illegal marker repeated and
faults immediately.

Content bytes still cannot forge structure: no character sequence
maps to a keyword id through the character, bucket, or dictionary
tables - the literal text `<|FE|>` lexes as six characters. The
encoder's run pass is the one deliberate emitter of operator ids
inside content, which is the rendering path markers are allowed to
enter through.

## fn Segmenter::encode_runs

TheUser's three-method magnitude bands, deterministic and canonical:
1 = the character; 2-3 = the keyboard row (plain pair where no row
exists); 4-9 = |unit||REPEAT||one count digit|; 10+ =
|unit||BEGIN_REPEAT||digits||END_REPEAT|. Rows never compose with
operators or each other ("without the double/triples"); REPEAT runs
early where it ties. The one-digit REPEAT count keeps literal digits
after a run unambiguous (seven spaces then "2024": one count digit,
the year survives); the bracketed form's end marker does the same
for multi-digit counts, and kills run-length chaining - 128 spaces
is six tokens.

Digit-class tokens (the ten ASCII digits) are exempt from run
encoding entirely: a number is place-value content, and the pre-fix
encoder really did RLE the zeros out of "4000019". Dictionary tokens
are likewise never runs.

## fn boundary_pieces

W+ word runs and single-character Unicode pieces (TheUser-ruled: `Dog
ate. it` lexes Dog, space, ate, period, space, it - they never
merge). One rule for every character; multi-character constructs
(`//`, `**`, `...`, headings) are learned compositionally over their
explicit tokens. Digit runs ride the Word class, so an
out-of-dictionary number falls to digit-per-character; lexed numeric
literals remain an open decision recorded in the campaign.

## struct Segmenter

### fn segment / fn segment_pieces

A word piece resolves whole: fold, dictionary lookup, else its
character split. There is no partial or longest-prefix match inside a
piece - inflected forms are expected to be dictionary entries
themselves (wiktextract forms ride along). Unicode pieces go straight
to the character layer; the dictionary can never hit one because
admission (dictionary.rs) only accepts single word-piece candidates,
so the lookup is skipped rather than run dead on gigabytes.
segment_pieces is the same walk keeping each token's text (a
dictionary token's text is the FOLDED row identity, not the surface
spelling) for display surfaces.

Case is folded at lookup and the emitted id is the folded row: the
surface-case round-trip (marker token, renderer rule, or lossy) is an
open decision; token counts, and therefore the compression ledger, are
unaffected by whichever lands.

## fn tokenize_text / fn split_markers

TheUser's test surface: `biquest tokenize "string"` prints the token
table as NUON text (not a plugin, so text is the interface). The
unicode column is `U+XXXX` for a character-layer token and null for a
keyword or dictionary word (his ruling); value is the token's text.

`<|XX|>` spellings render as keyword-page tokens (his ruling: token 0,
unicode null for `<|00|>`). That recognition lives in this verb alone,
never in the segmenter: content tokenization stays collision-free by
construction, and the test surface IS the deliberate-rendering path
markers are allowed to enter through. The scan is byte-wise and
boundary-safe because every matched byte is ASCII; a malformed
spelling (`<|0|>`, `<|GG|>`) falls through to ordinary content.
