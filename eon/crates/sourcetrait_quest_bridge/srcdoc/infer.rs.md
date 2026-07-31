# infer.rs

The seven channel forms as data models, and the two envelopes that carry
them. These are what both ends of a turn traffic in, which is the whole
correction this module exists for: the block grammar is the PARSE of a
token stream and belongs to ThinkHarness alone, so nothing else in the
system ever holds a tag, a header string and a content string.

They live in the bridge because both ends are eon - the daemon speaks
them through ThinkHarness, the plugin speaks them through UseHarness -
and this crate is already what carries the types both ends traffic in.

## struct InferValue

The one place a nu value's serialization is controlled, and the reason
it is a newtype rather than the value itself. A nu value's own derive
emits the engine's tagged form with a span on every node, and a span is
a byte offset into a file that existed in the SENDING process. NUON has
nowhere to put one, so spans do not travel because there is no channel
for them rather than because anything strips them; typed literals come
along for the same reason.

That property is locked with a control rather than asserted, because an
assertion that the carried form contains no span would pass equally
against a value that tags nothing.

## enum InferPass

Two accessors for one channel because two languages address it. nushell
spells it `$in`, and Liquid's grammar rejects the sigil in a variable
path - measured on 0.26 - so a template binds `in`. Rewriting the sigil
away inside a template before parsing was the alternative and is worse,
since a template is arbitrary text and a literal `$in` in prose would be
mangled by it.

## enum InferInput

## enum InferOutput

Open by ruling rather than by oversight: one variant is the model and
another format is an extension when one is given. Asking for the closed
set is reaching for a fixed schema at exactly the point where absorbing
an unenumerated one is the competence being built.

`<output>` travels in BOTH directions - outbound as the model's answer,
inbound as the result of something the caller was asked to run - which
is why the same form appears on both envelopes rather than a second
mechanism existing for the return path.

## enum InferNu

The variant IS the mode. Nothing here recovers a name from source and
nothing here decides where a body runs: only `Evaluate` is a think turn,
and that is a fact about the form rather than a routing decision this
type makes.

The payloads are `String` for now, by ruling. That is the def source as
the model wrote it, which is what a def body is; a parsed shape is what
they grow into once the prototype match produces one.

## struct InferRequest

`output` is what makes a SEQUENCE of turns work. A caller that ran an
ask sends the result back on it, and the Thinkspace's conversation
resumes rather than starting over - which is why the conversation
belongs to the Thinkspace and not to an engine.

## struct InferResponse

`inputs` is here for the same reason it is on the request, and it is not
symmetry for its own sake: an ask with nothing bound to it is a function
the caller has no arguments to run on.

`report` is optional because a turn paused on an ask has not ended, and
token counts and timings for a turn that has not ended would be fiction.
That is the same reasoning that keeps a failed turn's response separate
from a successful one rather than folding an outcome into it.
