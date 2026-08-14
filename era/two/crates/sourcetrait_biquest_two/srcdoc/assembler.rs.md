# assembler.rs

The two-level compiler between Quill assembly (the human authoring
surface) and the wire (the token stream the model sees). Assembly is
surface: indentation and raw-string wrappers exist only here, and the
wire is the truth a round trip preserves - a blank interior line and
an empty interior encode identically, and that is correct, not lossy.

## struct SyntaxTable

The draft table binds by ORDER (id = row index, NULL at 0x00) and
never hardcodes into CODE; the current draft is vendored and
embedded as the default data (TheUser: the default should be
hardcoded - the dictionary pattern), with --syntax the override and
re-vendoring the update path. The four structural heads resolve BY
NAME:
renaming OPEN in the draft would break existing assembly TEXT anyway
(assembly is names), so requiring the four names is not a constraint
the draft does not already carry. Bindings cap below 0xF6, where the
hardcoded operator block begins, and a binding that spells a
hardcoded name refuses at load - the block's names are fixed by the
codec, and a shadowing binding would make mentions ambiguous.

## fn encode / fn collect_interior

Indentation is enforced (TheUser's ruling): structural lines sit at
exactly two spaces per level, blank lines are legal only outside any
open block (the REIGN header region), and a non-empty interior line
carries its base indent exactly - stripped on encode, everything
past it data verbatim. The precision is the point: the surface
indent strips reliably, so no structural tabs or spaces ever reach
the wire.

Interior collection ends at the `END <format>` line recognized BY
POSITION - the only line that may sit at the BEGIN's own depth - so
a data line spelling `END NUON` at the data indent stays data, and a
non-END line at the parent depth faults. The noun still checks
against the BEGIN. The raw opener is honored only as the interior's
first line, at the base indent; the DECODER still wraps
conservatively (any structural-looking or delimiter-shaped line) -
over-wrapping is safe and round-trip stable, and a raw-delimiter-
shaped content line still needs the wrapper on encode.

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

## fn encode_interior / fn find_marker_spelling / spans

The mention mechanism (TheUser's rulings): a `<|BODY|>` spelling in
content is invalid everywhere except as a span opener or inside a
span. An ESCAPE..ESCAPED span converts spellings to their REAL
keyword ids - syntax quoted, not used - with prose riding along
through normal content tokenization; nesting faults. Both span kinds
are legal only inside TRAIN within an INPUT serialization, one
shared stack predicate (spans_legal) checked in both directions.
ESCAPE/ESCAPED/TRAIN/INPUT resolve by name as OPTIONS: a draft
table without them simply cannot express escape spans, which keeps
the existing hand tables valid. Bodies resolve as bound names, then
the hardcoded block's fixed UPPERCASE names, then two-hex page
addresses - the assembly surface spells keywords uppercase, so the
lexer's lowercase alias surface is deliberately a different one.
Decode renders in the same precedence, so a hex-spelled mention of
a named id canonicalizes on round trip. Inside an escape span the
decoder treats every keyword id as a mention - REPETITION and the
span operators included, since mention is exactly what those ids
deny as USE.

## fn encode_unicode_span / the unicode span

The forced per-character span (TheUser): UNICODE/UNICODED are
hardcoded operators, and both ids ride the wire as encapsulation
markers so the model sees the tokenization change in-band. The
encoder scans marker spellings but only one resolving to UNICODED
terminates; everything else - resolvable spellings included -
char-splits as raw content, which is why the span walks characters
through character_id directly and never touches the segmenter: no
dictionary, no case modifiers, no keyboard rows, no repeat bands.
The decoder renders the span's characters verbatim between the two
mention spellings, admits character-layer tokens only, and faults
an unterminated span, a span outside the legality scope, and an
orphan UNICODED. A UNICODE spelling inside an escape span is an
inert mention (the mention branch runs first); an ESCAPE spelling
inside a unicode span is raw characters.

## REIGN

`REIGN <human|ai|auto>` validates and produces no wire (an
authorship marker for TheUser and the agent). Decode therefore never
re-emits it: the begin.quill byte-compare runs against the file
minus its REIGN header.

## the smoke findings

Measured on TheUser's Syntax.md example against his Syntax.nuon:
byte-exact assembly round trip and wire-exact re-encode, 65 tokens.

Measured on begin.quill (the rewritten tokenizer-lesson genesis)
against the 31-row table: 996 wire tokens, 14 escape spans (ESCAPE
29, ESCAPED 30), 5 unicode spans, re-encode wire-exact, decode
byte-compare EXACT against the file minus its REIGN header. The
decoder renders postfix CAPITALIZED / UPPERCASED / CASED-overlay
forms through the UCD simple maps, and a blank interior line stays
byte-empty rather than taking the content indent. A case token
reaching the decoder with no dictionary row before it faults, as
does an overlay character that does not fold to its row's own
character - the canonicality the encoder emits.
