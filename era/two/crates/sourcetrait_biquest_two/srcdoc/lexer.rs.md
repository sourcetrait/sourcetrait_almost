# lexer.rs

The Quill content lexer. Lexer-style, never BPE: nothing merges across
a class edge, every piece is explicit, and word resolution is
dictionary hit, decomposition cover, or character split. The
morphology middle a BPE would learn is deliberately absent; the
expansion loop (a frequent unknown earns a row) replaces it, and the
decomposition cover bridges the gap for compounds the dictionary has
not yet earned rows for.

## the operator block and the id layout

Each offset a constant forever: ids 0x00-0xFF are the keyword page
(reserved whole - the page belongs to Quill, not the lexer), the
character layer starts at 256 in the table's dense order, the
keyboard-symbol rows sit above it, and the dictionary above those in
admitted-wordlist order.

The page allocates from both ends (TheUser): user bindings from 0x00
up (the Syntax.nuon draft), the hardcoded operator block from 0xFF
down - 0xFC BEGIN_REPEAT, 0xFD END_REPEAT, 0xFE REPEAT, 0xFF
REPETITION, the case family at 0xF8 CASED, 0xF9 CASE, 0xFA
CAPITALIZED, 0xFB UPPERCASED, then the span pair at 0xF6 UNICODE,
0xF7 UNICODED (TheUser: forced tokenization is a tokenizer property,
not portable language syntax, so the pair is hardcoded rather than
table-bound; both ids ride the wire as encapsulation markers
bracketing the exact per-character surface, so the model sees the
tokenization change in-band). The repeat trio is legal wire both
directions - the model may speak them (TheUser). REPETITION is the
AbstractConceptMarker backing the repetition AbstractConcept
(associations.rs): illegal in wire, never emitted, never parsed - it
exists so the association store can anchor every repetition
mechanism to one concept ("hint at convention by association").
Ones-fill corruption decodes as the wire-illegal marker repeated and
faults immediately. TokenizerSyntax.nuon in the iter's reference
directory tracks the block for posterity; nothing consumes it.

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

## fn candidate_items / fn assemble_candidate

TheUser's dictionary-decides design: the walk PROPOSES connected
candidates and the dictionary disposes. A word run extends across
internal connectors - hyphen and apostrophe, each with word
characters on both sides - and takes at most one edge apostrophe per
side (hyphens never edge-extend, which is what keeps `--flag`
prefixes on their keyboard rows). U+2019 normalizes to ASCII
apostrophe in the lookup forms; every part keeps its SURFACE beside
its folded form, because a miss must fall back to exactly the
pre-design tokens - the typographic apostrophe keeps its own
character row when nothing matched it. The internal loop consumes
every word-flanked apostrophe first, so the trailing-edge check
cannot steal an internal connector.

`whole_candidate_folded` is the extraction filter the dictionary and
associations passes share: an entry is admissible when its text is
exactly one candidate of more than one code point - the
single-code-point rule is the ONLY removal, and multiword entries
stay structurally unreachable because a space is never a connector.

## struct Segmenter

### fn segment / fn segment_pieces / fn raw_tokens

Resolution is the longest-first ladder of cheap lookups: the full
candidate as one row; on a miss with edges, the core with the edge
apostrophes as character tokens; on a core miss, each word part
resolves whole, then through the decomposition cover, then
char-split, and connectors take their character rows. Nothing
regresses on a miss by construction - the cover's floor is the
character split. The segmenter consults whatever word set it was
built with (admitted in production, the embedded full set on the
test surface) - the ladder is dictionary-agnostic. segment_pieces
keeps each token's text: a dictionary token's text is the FOLDED,
apostrophe-normalized row identity, not the surface spelling.

### fn decompose_cover / fn cover_beats / fn camel_cover

TheUser's design, three criteria in order: cover the missed part
with dictionary rows; among covers, fewest tokens wins (nu + sh +
ell loses at three); ties prefer the longer match at the END,
because English wording is prefix-oriented - the tail carries the
root - so nushell covers as nu + shell (tail 5) over nus + hell
(tail 4). A position no row covers is one character token, and a
character counts as a token in the comparison, which is what makes
the cover strictly better than the char split it replaces. Dynamic
programming over suffixes: the tie-break is end-anchored, so the
suffix-optimal choice composes and the DP is sound. Each covered
span maps back onto the cased surface (the simple fold is
one-to-one per character) and carries its own postfix case
operator, so Nushell reads nu CAPITALIZED shell.

Camel casing overrides the criteria outright (TheUser): the case
transitions mark the INTENDED boundaries, so when every camel
segment resolves as a case-classified row, that segmentation is the
cover even where the criteria would pick another - AntOne reads
ant + one though an + tone carries the longer tail, LabRat reads
lab + rat, and an uppercase run breaks before its last capital
(HTTPServer reads http UPPERCASED server CAPITALIZED). Any segment
that fails to resolve abandons the camel path whole and the
criteria decide.

### fn cased_overlay / fn overlay_beats

TheUser's combination design for the CASED overlay: the overlay may
spend dictionary tokens with case operators wherever the total ties
or beats the character run - MicroSoft reads microsoft CASED micro
CAPITALIZED soft CAPITALIZED, six tokens against eleven - and a tie
prefers the words (dOg reads dog CASED d og CAPITALIZED at three
either way). The cover prices in TOKENS (an operated span costs
two), reusing the end-anchored tie-break. The decode contract
keeps the row-length bound but measures it in DECODED characters:
a character piece counts one, a word piece its own row length, its
optional operator shaping the rendered surface - which is also how
encode_runs walks past an overlay without re-encoding it.

Case is folded at lookup and the emitted id is always the folded row;
the surface rides POSTFIX case tokens (TheUser's design - the match
first, then the tokenizer token, because the word association firing
immediately is faster for the model). Classification against the UCD
simple maps: the folded surface is the bare row; the
first-char-uppercased form takes CAPITALIZED; the all-uppercased form
takes UPPERCASED; anything else (the QuILL class) takes CASED plus
exactly the row's length in surface character tokens - length-bounded
by the row itself, so no terminator exists, and the overlay is exempt
from run encoding because the length bound IS the decode contract.
The overlay carries the connector-normalized cased form, so a curly
apostrophe decodes normalized like every other surface. One row per
word against OLMo's six fossilized case-and-space variants (this/This/
THIS each twice, measured); the ours-corpus ledger prices the family
at 1.689 against the caseless 1.641.

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

A `<|unicode|>`..`<|unicoded|>` pair applies the tokenizer's span
semantics on this surface too: interior text and any other spelling
render one character row per character - the exact surface, no
dictionary, no case, no rows, no bands - and an unterminated span
faults, because the display surface shares the codec's discipline
(TheUser's fixture: `<|unicode|>Microsoft<|unicoded|>` is the two
markers bracketing ten character rows, M's surface preserved).
