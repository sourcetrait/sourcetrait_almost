# channel.rs

The typed-channel grammar's framing half: what a block IS on the wire, and how
one is read back. What a block MEANS - whether its declared type agrees with
the def it binds to - belongs with the evaluator, because that check needs a
parsed def signature rather than a parsed block.

THE ID IS THE BINDING FACT AND THE SPELLING IS DATA. Every tag carries both, and
they answer different questions. The reserved extra ids are real, present in the
embedding table, and atomic under BPE because added tokens never merge. The
strings attached to them are free: ai2 renamed four of these very ids to the
tool markers at their instruct stage, which is the precedent for the design's
own readable names.

That is why `spelling` and `name` are separate accessors rather than one. The
shipped tokenizer maps `<|extra_id_2|>` to 100271 today, so `spelling` is what
actually frames a turn; `name` is what the design calls the same slot and what a
diagnostic should say to a human. A tokenizer rename swaps which one frames,
and because the module is anchored on ids that is a one-line change rather than
a migration.

Nothing here can prove the spelling-to-id mapping, because this crate has no
tokenizer. That lock lives in the era library's `verify_token_map`, which reads
`TAGS` and checks each spelling against the shipped map - so the two cannot
drift without a test failing.

## const TOKEN_EXTRA_ID_0

Separated from the run because it is: the config anchor sits at 100256 while
the other six are contiguous from 100270. The gap is not tidiness, it is what
the base checkpoint's own map looks like.

## enum Tag

## const TAGS

Emission order, which is also id order, which is also the order a token dump
reads in. The closer is pinned at the far end deliberately so that the openers
stay contiguous, and that is what let `liquid` extend the run as a seventh tag
without moving anything. There is no eighth slot; the map is fully allocated,
so any future channel costs an existing one.

## enum Aliasing

The measured reason this exists: across 63 probe generations, ZERO opened with
the output marker and every structured emission routed through the trained
`function_calls` attractor instead. A strict protocol variant that explicitly
prohibited it changed nothing, so in-context teaching does not reach the routing
position at all.

`ToolMarkers` is therefore a temporary extraction-side accommodation with a
known retirement: it goes when marker routing is trained. Keeping it a named
parameter rather than a permanent leniency is what makes that retirement a
deletion rather than an archaeology exercise.

`Strict` is the grammar as locked, and it refuses unmarked content anywhere -
before, between or after blocks. That is not pedantry: v1 has no mixed prose and
data, prose belongs to chat turns, and a partial answer belongs to the
insufficiency tag rather than to loose text beside a block.

## struct Block

## fn escape_content

## fn unescape_content

THE ORDER INSIDE `escape_content` IS LOAD-BEARING and reads backwards. Backslash
is doubled FIRST, before the break characters are turned into backslash
sequences, so the backslashes this function introduces are never themselves
doubled. Swap the two and a payload containing a literal backslash-n becomes
indistinguishable from a payload containing a real newline.

The reason any of this exists: `to nuon` does not escape a newline inside a
string value, it emits the byte raw. A record carrying one therefore spans two
lines, and the line-anchored form would read that as a boundary. Escaping at
emission closes the class, and it is safe for a record or a table because the
only raw newlines a compact render can carry are inside double-quoted strings.

## fn render_block

## fn render_blocks

## fn parse_blocks

Liberal on input, canonical on output, and the asymmetry is deliberate. The
locked examples show `pass` inline and everything else block-shaped, so that is
what renders; both forms parse, because a model that emits the other one is
making a formatting choice rather than a protocol error.

THE `liquid` FORGERY HAZARD IS OPEN AND IS NOT SOLVED HERE. Every other block's
content is constrained - NUON, a nu def, a record - so a line reading exactly
the closer is vanishingly unlikely, and the escaping rule closes the rest. A
template is arbitrary text by design and can legitimately contain that line. The
NUON escape does not transfer, because there the escape is what the parser reads
back while a template's newlines are structural. This parser takes the first
closer line and stops, which is the least-committal reading available; resolving
it properly is owed before liquid emission is trained rather than after.

## fn opener_at

## fn is_closer

## fn inline_payload

## struct Diagnostic

## struct Envelope

Three properties, each answering something a repairing model needs and a
compiler-shaped error does not give it.

It addresses by CELL-PATH rather than by span, because the agent reasons in the
data model and a byte offset into a rendered value means nothing there.

It COLLECTS rather than bailing at the first fault, because first-error-only
costs one repair round trip per fault, and a round trip here is a whole
generation.

It renders as a NUON record rather than a string, because it rides back as an
environment turn and printed data is NUON. The shape is the MCP's own error
envelope - errors and warnings, each row carrying a kind, a source and a message
- reused rather than coined, so the harness and the model already agree on it.

Diagnostics are needed at all because nushell's own rejection reads `Input type
not supported.` with no element index and no field name. The pipe is the fast
gate; this is what tells anyone why it closed.

## enum Format

The vocabulary is closed on purpose, and the reason is a promise rather
than a preference: naming a format asserts the harness can read it. An
open set would let a model emit `yaml` fluently and correctly-looking
while nothing downstream could take it, which is the worst failure shape
available - confident and unreadable.

That is why it is an enum here and not a string. This type IS the
vocabulary, so the vocabulary and the processing cannot drift apart
without the compiler noticing, which is the same requirement the `<nu>`
mode set already carries.

The diagnostic lists every member rather than only reporting the miss,
because the reader who needs it is a model choosing a word.

## enum Declared

Three cases rather than an `Option<Type>`, because the absent case splits
in two and the two mean opposite things. `Untyped` says the content has
no type to check. `Typedef` says the content IS a type - the one place
the slot names what the payload is rather than what a value conforms to.
Collapsing them would lose exactly the distinction that lets a typedef be
validated as a type while text is validated as nothing.

## struct Descriptor

### fn parse

THE PAIRING IS PART OF THE VOCABULARY, which is why this is not a split
followed by two independent parses. `nuon` owes a type, `nu` takes the
word `type` alone, `string` takes nothing. A format word in the wrong
pairing is refused here rather than surviving to fail somewhere with a
less legible message.

Splitting on the first whitespace rather than tokenising is deliberate: a
type can contain spaces (`record<a: int, b: int>`), so only the FIRST
boundary is a delimiter and everything after it is the type's own text.

### fn render

The inverse of parse for every legal descriptor, which the round-trip
lock asserts. Untyped renders the format alone rather than a format and
an empty slot, so there is exactly one spelling of a text block and no
trailing space to strip.
