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

That is why `spelling` and `name` are separate accessors rather than one.
`spelling` is the `<|extra_id_N|>` token that frames a turn - what the model
sees, emits and is parsed on. `name` is the readable `<|...|>` alias the training
assets and diagnostics carry, distinct enough from content that it is not
injected by accident and never a token itself. `authoring_to_wire` bridges them
at the training-render boundary; renaming the tokenizer to the aliases was
declined, since it would touch the frozen added-token map and imply a retrain.

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

## fn escape_payload_markers

THE MARKERESCAPE CONVENTION, and it closes the arbitrary-text forgery class the
NUON escape cannot reach. A prose or template payload is raw and multi-line -
its newlines are structural - so a content line reading exactly the closer
would end the block early, and a marker spelling anywhere in content would trip
the stray guard or, worse, TOKENISE: added tokens match on the raw string, so
an unbroken `<|extra_id_6|>` in content becomes the real closer token however
it got there, and a backslash in front of the whole spelling would not stop
that match.

Breaking the introducer itself is what works on every layer at once. `<\|`
matches no added token (the tokenizer never sees the exact string), no opener
and no closer (the line predicates compare exact spellings), and no stray-guard
scan (`carries_marker` is a substring match over unbroken spellings). Markdown
renders `\|` as a literal pipe, so the displayed text reads back as the marker
being discussed - the escape is display-honest for md, and a visible wart only
in txt, which is the rarely-used format by design.

It is deliberately NEVER undone by the harness. Unescaping at parse would put
unbroken spellings back into rendered prose, where the stray guard cannot tell
restored content from a carrier-discipline fault; keeping the escape as the
carried spelling keeps the guard a one-rule scan. The uniform `<|` rule (rather
than a list of the seven spellings) covers the whole `<|...|>` added-token
family - turn markers included - in one learnable rewrite. The non-pipe marker
family (`<functions>` and kin) is deliberately outside it: those rows are
refused at conversion rather than escaped, because teaching tool-marker tokens
mid-prose is the trained-attractor hazard rather than a framing one.

## fn render_block

## fn render_blocks

## fn authoring_to_wire

The bridge that makes training and serving one grammar. The training assets carry
the readable `name()` aliases; the model only ever sees and emits the `spelling()`
tokens. Without this, a readable `<|nu|>` in an assistant turn tokenises as
ordinary BPE and the runtime's extra_id parser never sees a marker - the drift
that let a whole offload emission cross the wire as prose. Applied to message
content before tokenisation, through the one Tag map, so what the model trains on
is byte-identical to what the runtime renders and reads. A plain replace is safe
because the `<|...|>` aliases do not occur in content by accident; the distinct
form is what keeps that true.

## fn carries_marker

The prose path's guard, and it deliberately matches SPELLINGS rather
than token ids. An emitted id always decodes to its spelling, so a text
match cannot miss a real marker - and it is strictly wider: text
assembled from ordinary tokens to LOOK like a marker deceives a
downstream reader exactly as a real one would, so it repairs exactly
the same. The id-level test the design first framed would need the ids
threaded through `step`, and would catch less.

## fn parse_blocks

Liberal on input, canonical on output, and the asymmetry is deliberate. The
locked examples show `pass` inline and everything else block-shaped, so that is
what renders; both forms parse, because a model that emits the other one is
making a formatting choice rather than a protocol error.

THE ARBITRARY-TEXT FORGERY CLASS IS CLOSED BY MARKERESCAPE, not here. A prose
(md, txt) or template payload is arbitrary text whose newlines are structural,
so the NUON escape does not transfer; `escape_payload_markers` breaks the `<|`
introducer at authoring and conversion instead, so no legal content line can
read as a marker line by construction. This parser stays exactly as it was -
first closer line wins - because an UNESCAPED marker in an emission is a
carrier-discipline fault the break-early posture should refuse (a forged close
strands the trailing content as unmarked, which Strict rejects), not a case to
parse around.

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

`md` and `txt` are the no-implied-format ruling's landing: there is
always a data format in an output, and the implied one everywhere in
current practice is markdown unnamed. Naming it makes prose a formatted
payload like any other - md the lingua franca, txt the rare
disambiguation - and an entry using no markup feature is still md,
because the format designates what the payload IS rather than whether
its syntax shows. `is_prose` is the processing split those two carry:
raw, multi-line, untyped - the payloads MarkerEscape exists for.

`liquid` composes WITHIN a format rather than standing alone, which is
why a bare `liquid` refuses: a template that does not say what it
renders to would be an implied format one level up. The composition
costs no tag and no slot - it is a word in a descriptor - and it is the
direction the SystemChannelRemap debt records for the `<|liquid|>` tag
itself.

The diagnostic lists every member rather than only reporting the miss,
because the reader who needs it is a model choosing a word.

## enum Declared

Four cases rather than an `Option<Type>`, because the absent case splits
and the splits mean different things. `Untyped` says the content has
no type to check. `Typedef` says the content IS a type - the one place
the slot names what the payload is rather than what a value conforms to.
`Renders` says the content is a template and names its target, so the
second slot carries a format where every other case carries a type or
nothing. Collapsing them would lose exactly the distinctions that let a
typedef be validated as a type, a template be rendered, and text be
validated as nothing.

## struct Descriptor

### fn parse

THE PAIRING IS PART OF THE VOCABULARY, which is why this is not a split
followed by two independent parses. `nuon` owes a type, `nu` takes the
word `type` alone, `string`, `md` and `txt` take nothing, and `liquid`
owes a prose target. A format word in the wrong pairing is refused here
rather than surviving to fail somewhere with a less legible message.

Splitting on the first whitespace rather than tokenising is deliberate: a
type can contain spaces (`record<a: int, b: int>`), so only the FIRST
boundary is a delimiter and everything after it is the type's own text.

### fn render

The inverse of parse for every legal descriptor, which the round-trip
lock asserts. Untyped renders the format alone rather than a format and
an empty slot, so there is exactly one spelling of a text block and no
trailing space to strip.
