# lexer.rs

The Quill content lexer. Lexer-style, never BPE: nothing merges across
a class edge, every piece is explicit, and the two-level rule is
absolute for words - dictionary hit or character split, nothing
between. The morphology middle a BPE would learn is deliberately
absent; the expansion loop (a frequent unknown earns a row) replaces
it.

## const KEYWORD_PAGE_SIZE / CHARACTER_OFFSET / KEYWORD_REPEAT

The id layout, each offset a constant forever: ids 0x00-0xFF are the
keyword page (reserved whole regardless of how many keywords are
bound - the page belongs to Quill, not the lexer), the character layer
starts at 256 indexed by the table's dense order, and the dictionary
layer sits directly above it in admitted-wordlist order. There are no
sequence rows: TheUser's uniformity ruling retired the bucket-layer
experiment - every character acts like every other (character row
plus REPEAT), because special-cased rows are irregularity that bites
in training. The measured arc that priced it: 1.95 characters-only,
1.64 with 17 designed rows, 1.72 uniform - the ~58k-position hit
taken deliberately.

The page allocates from both ends, TheUser's ruling: user bindings
grow from 0x00 up (the Syntax.nuon draft), hardcoded
tokenizer-operator aliases grow from 0xFF down. REPEAT sits at 0xFF
with the authoring alias `<|repeat|>`; it is hardcoded because the
codec itself interprets it before any binding table exists. This
supersedes the concept archive's 0xFF-as-generic-closer note, and the
corruption property survives: a ones-filled buffer decodes as REPEAT
with no operand, faulting as loudly as NULL-fill.

Content bytes still cannot forge structure: no character sequence
maps to a keyword id through the character, bucket, or dictionary
tables - the literal text `<|FF|>` lexes as six characters. The
encoder's own collapse pass is the one deliberate emitter of a
keyword inside content, which is the rendering path markers are
allowed to enter through.

## fn Segmenter::collapse_runs

Token-level run-length encoding, TheUser's design replacing all
sequence enumeration: a run of three or more identical non-word
tokens becomes unit, REPEAT, one count digit. Ties prefer REPEAT
(ruled); a run of two stays plain because the group costs three.
Groups carry at most nine and chain greedily, a leftover of one or
two staying plain. The count is EXACTLY one digit token - that bound
is what keeps the wire unambiguous when literal digits follow a run
(eight spaces then "2024": the decoder takes one count digit and the
year survives as content). The count digit is the ordinary character
row for 2-9, spending no vocabulary. Space, newline, tab, dash - any
identical-token run collapses through this one pass with no
dedicated entries anywhere.

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
