# lexer.rs

The Quill content lexer. Lexer-style, never BPE: nothing merges across
a class edge, every piece is explicit, and the two-level rule is
absolute - dictionary hit or character split, nothing between. The
morphology middle a BPE would learn is deliberately absent; the
expansion loop (a frequent unknown earns a row) replaces it.

## const KEYWORD_PAGE_SIZE / CHARACTER_OFFSET

The id layout, each offset a constant forever: ids 0x00-0xFF are the
keyword page (reserved whole regardless of how many keywords are
bound - the page belongs to Quill, not the lexer), the character layer
starts at 256 indexed by the table's dense order, and the dictionary
layer starts at 256 + assigned_count in admitted-wordlist order. The
lexer never emits a keyword id: content tokenization cannot spell
structure, which is what makes the wire collision-free by
construction.

## fn boundary_pieces

W+ word runs and single-character symbol pieces (TheUser-ruled: `Dog
ate. it` lexes Dog, space, ate, period, space, it - symbols never
merge). Digit runs ride the Word class, so an out-of-dictionary number
falls to digit-per-character; lexed numeric literals remain an open
decision recorded in the campaign.

## struct Segmenter

### fn segment / fn segment_pieces

A word piece resolves whole: fold, dictionary lookup, else its
character split. There is no partial or longest-prefix match inside a
piece - inflected forms are expected to be dictionary entries
themselves (wiktextract forms ride along). Symbol pieces go straight
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

## fn tokenize_text

TheUser's test surface: `biquest tokenize "string"` prints the token
table as NUON text (not a plugin, so text is the interface). The
unicode column is `U+XXXX` for a character-layer token and null for a
dictionary word (his ruling); value is the token's text.
