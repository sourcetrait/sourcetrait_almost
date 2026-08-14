# assembler.rs

The two-level compiler between Quill assembly (the human authoring
surface) and the wire (the token stream the model sees). Assembly is
surface: indentation and raw-string wrappers exist only here, and the
wire is the truth a round trip preserves - a blank interior line and
an empty interior encode identically, and that is correct, not lossy.

## struct SyntaxTable

The draft table loads at runtime and binds by ORDER (id = row index,
NULL at 0x00), because the table is TheUser's work-in-progress and
code never hardcodes it. The four structural heads resolve BY NAME:
renaming OPEN in the draft would break existing assembly TEXT anyway
(assembly is names), so requiring the four names is not a constraint
the draft does not already carry. Bindings cap below 0xFC, where the
hardcoded operator block begins.

## fn encode / fn collect_interior

Interior collection ends ONLY at an `END <format>` line - an interior
line reading `OPEN X` or a bare keyword is content by position. The
one collision is an `END <other>` line, which faults the encode as a
noun disagreement; that is what the raw-string wrapper exists for,
and why the DECODER wraps conservatively (any structural-looking
line) - over-wrapping is safe and round-trip stable, under-wrapping
would re-encode wrong. The raw opener is honored only as the
interior's first line, matching what the decoder renders.

Content lines strip exactly their nesting depth of indentation and
keep the excess: content's own indentation (nu code) is content.

## fn decode / fn decode_interior

The decoder is the conformance oracle's display half, so it enforces
what the encoder guarantees: canonical bands (a REPEAT count below 3
or a bracketed count below 10 faults - the encoder never emits them,
so a model emission that does is non-conformant), digit-headed runs
fault (a number is place-value content), keyboard rows and operators
never compose, an operator needs a unit before it, and 0xFF anywhere
is the wire-illegality fault named as the AbstractConceptMarker.
Every fault carries its token position.

The `repeatable` gate is the character layer minus the Digit class -
Word-class characters DO run-encode (the uniformity ruling: one rule,
zero exceptions, everything acts like `$`).

## fn encode_interior / fn find_marker_spelling / escape spans

The mention mechanism (TheUser's rulings): a `<|BODY|>` spelling in
content is invalid everywhere except inside an ESCAPE..ESCAPED span,
where it converts to its REAL keyword id - syntax quoted, not used -
and prose rides along through normal content tokenization; nesting
faults; spans are legal only inside TRAIN within an INPUT
serialization, checked on both directions from the same stack
predicate. ESCAPE/ESCAPED/TRAIN/INPUT resolve by name as OPTIONS: a
draft table without them simply cannot express spans, which keeps
the existing hand tables valid. Bodies resolve as bound names first,
then two-hex page addresses; decode renders bound names (canonical),
so a hex-spelled mention of a bound id canonicalizes on round trip.
Inside a span the decoder treats every keyword id as a mention -
REPETITION included, since mention is exactly what the marker
exists to deny as USE.

## REIGN

`REIGN <human|ai|auto>` validates and produces no wire (an
authorship marker for TheUser and the agent). Decode therefore never
re-emits it: the begin.quill byte-compare runs against the file
minus its REIGN header.

## the smoke findings

Measured on TheUser's Syntax.md example against his Syntax.nuon:
byte-exact assembly round trip and wire-exact re-encode, 65 tokens.
CODE is used in the example but not yet tabled, so the assembler
rejects it until the draft table binds it - the reject is the
assembler working, not a defect.

Measured on begin.quill (the genesis document) against the 31-row
table: 411 wire tokens, 10 escape spans (ESCAPE 29, ESCAPED 30),
re-encode wire-exact, and the decode byte-compare EXACT once the
case family landed (the decoder renders postfix CAPITALIZED /
UPPERCASED / CASED-overlay forms through the UCD simple maps, and a
blank interior line stays byte-empty rather than taking the content
indent). A case token reaching the decoder with no dictionary row
before it faults, as does an overlay character that does not fold to
its row's own character - the canonicality the encoder emits.
