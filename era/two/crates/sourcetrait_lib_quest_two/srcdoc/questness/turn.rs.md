# turn.rs

The turn is the layer that makes Questness a model runtime rather than a
prompt wrapper: it decides what the model is shown and what its emission
obliges us to do next. Everything here is experimental in the campaign's
sense - the shape is a first draft expected to move once there is a
daemon and a plugin exercising it.

## const THINK_ROLE

## const THOUGHT_ROLE

The Thinkspace's own turn, and the reason it is a chat ROLE rather than a
tag: the seven-tag map is fully allocated, so a channel of its own would
have had to displace one, while a role costs nothing at all. A role is
plain text after `<|im_start|>`, so `think` is a few ordinary tokens
rather than an added one.

Routing is off the FORM rather than off a parsed wrapper, because an
evaluate is ALWAYS a think turn. That is why only the thought half is
read here; the think name stands because the pair is the design rather
than because anything reads it yet.

## struct Binding

Carries the nushell spelling of the channel, `$in` or `$args`, rather
than an enum. The spelling is what the wire uses, what `check_agreements`
compares against, and what a template's `binding_name` strips a sigil
from, so an enum would be converted back to this string at three
boundaries to buy nothing.

## struct Request

The caller's side before anything is rendered. `config` carries the whole
record including the Questness keys, because stripping them is
`config::prepare`'s job and doing it earlier would mean two places knew
the key list.

## struct Answer

`value` and `declared` are both optional and they are optional for the
same reason: the design admits an answer that is raw prose with no data
at all, and admits an unmarked emission under tool-marker aliasing. An
answer that carries prose only is not a degraded answer, so it is not
an error case.

`config` is REPORTED here and judged elsewhere, which is the division
the shape convention needs. Whether the model was entitled to send one
depends on what the caller declared, and this layer has never seen the
caller's declaration - it reads an emission. Deciding here would mean
threading a shape through `interpret` to answer a question `shape`
already owns.

## enum Outcome

Five arms, split by WHO ACTS NEXT rather than by whether something went
wrong. `Answered` means the caller is done. `Think` means the
Thinkspace's own evaluator runs it and the model continues inside this
same turn. `Ask` means the CALLER runs it and the turn pauses there.
`Insufficient` means the model declined, and `Repair` means the model
goes again with the envelope as feedback.

`Think` and `Ask` are two arms rather than one arm carrying a
destination, and that is the correction this module was rewritten for. A
form does not HAVE a destination that something decides for it: an
evaluate is a think turn and nothing else ever is. Carrying the two as
one shape with a flag beside it is what let a client-harness seam grow
inside Questness, which never belonged here - Questness calls nothing
out.

`Ask` carries its bindings because an ask with nothing bound to it is a
function the caller has no arguments to run on.

`Repair` deliberately returns `Ok`. A malformed emission is an ordinary
outcome of running a model that has not been trained on this grammar -
10_Channels measures marker routing as dead at zero of six - so making it
an `Err` would put the expected case on the error path and force every
caller to distinguish "the model wrote something wrong" from "the harness
broke". The `Err` channel is reserved for the second.

## fn assemble

Renders the config and input blocks, then the prompt.

The input block's header is the value's own derived type rather than
something the caller declares. That is the smart command's contract from
10_Channels - piped data self-describes - and it is why there is no
typedef parameter on `Request`.

The config block is SKIPPED when the visible record is empty. Sending
`<config>` with `{}` would teach the model that an empty record is a
thing it will routinely see, when the real signal is that the caller
supplied nothing.

The prompt goes LAST, after the blocks. The blocks are the turn's data
and the prompt is the instruction over it, and a model reading top to
bottom should have the data in hand before the question. This is a
choice rather than a constraint, and it is one of the things a later
reading may want to measure rather than assume.

## fn interpret

`insufficient` is a parameter rather than something scanned for, and this
is the one piece of the signature that looks wrong until you know why.
The insufficiency tag is `<|endofprompt|>`, which is a SPECIAL token, so a
skip-special decode strips it from the text entirely. There is nothing in
`text` to find, at any level of effort. Detection belongs to whatever
holds the token ids - the engine or the plugin - and the honest way to
express that here is to make the caller say so.

The `<nu>` block wins over `<output>` when both are present. A turn that
proposes a check and an answer is asking for the check to run, and the
answer it already wrote is what the check is about.

## fn nu_form

Runs `check_agreements` and the binding decode, then bails on a dirty
envelope. Note the order: bindings are decoded even when the agreements
already failed, so one round trip carries every diagnostic the emission
earned rather than the first class of them. That is the collect-rather-
than-bail rule from 10_Channels, applied across two checks instead of
inside one.

The form match runs LAST, after the agreements, for the same reason. A
body that fails its prototype has still bound channels worth reporting
on, and matching first would cost a round trip per class of fault.

## fn answer

A block with an EMPTY header is prose and a block with a header is typed
NUON, and that single rule is what lets raw-text answers and typed
answers share one path. It follows from the grammar rather than being
invented here: data without a definition is legal only at `extra_id_0`.

The rendering falls back to the output block's own content when no
`<liquid>` block follows. So a model that answers with prose gets that
prose as its rendering, and a model that answers with data and no
template gets its data rendered as the NUON it wrote. The second is
arguably wrong and is worth revisiting - it hands the caller a data
literal where prose was expected - but the alternative is inventing a
rendering the model did not ask for.

## fn emitted_config

A `<config>` travels OUTBOUND as the caller's curation, so one arriving
back is the model addressing the harness rather than answering the
question. Reading it is what gives the shape something to police: a block
left unparsed is indistinguishable from one never sent, and the denial
would silently never fire.

A malformed payload here is `channel::nuon` rather than a kind of its
own, because it is the same failure the bindings and the output payload
have - the model wrote something that is not NUON - and a repairing model
gains nothing from learning which block it was wrong in beyond the source
the row already names.

## fn typed_payload

Three failure classes with three distinct diagnostic kinds -
`channel::typedef`, `channel::nuon`, `channel::conformance` - because a
repairing model needs to know which of the three it got wrong, and they
call for different repairs. A bad typedef means rewrite the header, bad
NUON means rewrite the payload, and a conformance failure means the two
disagree and either could move.

## fn liquid_bindings

Binds the template to the OUTPUT value, falling back to `$in` when no
`<pass>` names it. The fallback exists because the worked example in
16_Questness always writes the pass explicitly, and a model that omits it
has still made its intent obvious - there is exactly one value in scope.

Pass diagnostics are dropped on this path, which is a known soft spot. A
malformed `<pass>` in an answer surfaces as a liquid render failure
rather than as an unknown-binding row, so the failure is visible but
mis-attributed. It is filed rather than fixed because the answer path has
no contract to check a pass against, so reporting one properly needs a
notion of agreement that does not exist without a def.

## fn run_think

The invocation is APPENDED to the body, and this is the mechanic the
whole inside path rests on. A `<nu>` block declares a def; it does not
call one. Evaluating the body alone returns `nothing`, which is exactly
what the first version of this module did and what its test caught.

`$args` renders as a NUON literal at the call site rather than being
bound some other way, which works because NUON is valid nu literal syntax
by construction - the same property that makes the data format and the
language one grammar.

`$in` rides the pipeline and the call is written `$in | <mode>` rather
than relying on the block input reaching the second pipeline. The
explicit form is the documented rule - in first position inside a block,
`$in` is the block input - and it does not depend on how a bare `def`
statement interacts with pipeline input, which is the kind of thing that
would work until it did not.

## fn thought_turn

The answer comes back on a turn of its own rather than as bare text,
which is what makes the think and thought pair a REQUEST AND RESPONSE at
the chat-role layer - the same discipline the wire already follows with a
pair per operation.

It wraps an `<output>` block rather than replacing one. The ruling that a
result from `<nu>` comes back TYPED still holds; the thought turn is what
replaces the environment turn it used to ride on, not what replaces the
block.

## fn nuon_payload

Escapes on the way out. `to nuon` does not escape a newline inside a
string value, so a record carrying one renders across two lines and the
second could read as a closer at column zero. The escape closes that,
and it is safe because the only raw newlines a compact render can carry
are inside double-quoted strings, where the escape is what nushell reads
back.

This is applied to NUON payloads only. A `<nu>` body and a `<liquid>`
template are legitimately multi-line and their newlines are structural
rather than content, so escaping them would corrupt them. The liquid case
is the one 10_Channels records as genuinely unresolved: a template is
arbitrary text and can contain a line that IS the closer, and no escape
transfers there because a template's newlines have to survive.

## fn bound_blocks

Reads the pairing out of `contract` rather than restating it. The
index-minus-one rule that decides which block a `<pass>` binds is subtle
enough that two copies would drift, and `contract` is the module that
owns the agreements, so it owns the pairing too.

It discards the envelope that call produces. On the sub-turn path those
diagnostics are already collected by `check_agreements`, so keeping them
would double every row; on the answer path see the soft spot under
`liquid_bindings`.

## const REPL_MODE

Not a form in the bridge's wire enum, and that is the point: a repl never
leaves the Thinkspace, so putting it on the wire would model a journey it
cannot take.

## fn interpret

### the thinking parameter

A parameter rather than state because the caller owns the turn and knows
its role; this layer only knows what an emission says. Passing it in
keeps the rule enforceable here and the SIGNAL where it is actually
observable.

Today no caller reads a think role out of an emission, so the bridge
passes true and the check never fires in production. That is recorded as
debt rather than hidden: the rule and its locks exist, and what is
missing is the reader.

## fn sub_turn

`repl` is matched before the parser is asked for anything. It has to be:
the signature is discovered by diffing the declaration set after a parse,
and a bare expression declares nothing, so the ordinary path would report
a missing def rather than a repl.

The deny is returned as a repair envelope rather than an error, for the
same reason every other conformance failure is - a model reaching for a
reasoning mode in the wrong turn is the ordinary untrained case, and the
repair vocabulary is what that is for.

## fn typed_payload

What is checked follows from what the descriptor says, and the three
cases check three different things. A conforming payload checks the value
against its type. A typedef checks that the CONTENT parses as a type,
which is that block's own contract and the only check available - there
is no value to conform. Text checks nothing, because a string that
round-trips is already everything the format promised.

The typedef case returns `Type::String` beside the text: nushell has no
first-class type value, so the carriage is a string until there is
something better to carry it in. The descriptor is not lying about the
payload; it names what the text MEANS while the type names how it is
held.
