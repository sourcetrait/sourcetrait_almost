# harness.rs

The seam between Questness and whatever runs the modes that leave it.
Experimental in the campaign's sense: this is the first draft of a
contract that has no real implementation on either side yet, and the
shape is expected to move once a daemon and a plugin are pushing traffic
through it.

## struct QuestNuValue

The newtype is not stylistic and it is not defensive. `nu_protocol::Value`
derives `Serialize`, so a value CAN be put on a wire directly - and what
comes out is the engine's internal tagged representation, carrying the
variant name and a span for every node. A span is a byte offset into a
source file that existed in the sending process, so it is meaningless to
a receiver and is exactly the kind of thing that looks harmless until
something starts trusting it.

Carrying the NUON spelling instead makes the property structural rather
than promised. NUON text has no spans to carry, so "spans do not travel"
is not a rule anyone has to remember - there is nowhere for one to sit.
It also gets typed literals for free, which is the reason NUON rather
than JSON is the target: a duration, a filesize and a cell path all
survive as themselves, where a JSON encoding would flatten them to
numbers and strings and lose which was which.

The test carries a CONTROL for this, deliberately. Asserting that the
carried form contains no `span` proves nothing on its own, because a
value that never tags anything would pass the same assertion. So the test
first serializes a bare `nu::Value` and asserts the tagged form IS there,
which is what makes the contrast evidence rather than an observation.

## struct HarnessRequest

One request type carrying the mode as a string, rather than an enum with
a variant per mode. The plan said an enum, and this deviates from it on
the design's own grounds: 10_Channels records that the modes "differ by
SIGNATURE rather than by form", and if that is true then one shape
carrying a def and its channels serves all of them. An enum would encode
a difference the design says does not exist, and would need a new variant
for every mode added, which is the closed-set property `turn` was
deliberately built without.

The `output` field carries the def's declared output type as text so the
answer can be checked against what was asked for. It is a string rather
than a `nu::Type` because a type is not serializable and reparsing it on
the far side is one call.

## enum HarnessResponse

Flat rather than carrying an `Envelope`, and the reason is the seam
rather than the shape. An envelope is the repair loop's vocabulary and it
belongs to the conversation with the model; what a harness owes back is
narrower - it either ran the thing or it did not. Keeping the failure as
a kind and a message means the seam needs no serialization for the
diagnostic types, and Questness turns a failure into an envelope row on
its own side where the rest of that vocabulary already lives.

## fn as_block

Where the response becomes what the model actually reads. This is the
half that makes the sub-turn typed: 10_Channels settled that a result
returning from `<nu>` comes back through `<output>` carrying its type,
rather than as opaque text on an environment turn, and this is that
conversion in one place.

It escapes the payload for the same reason `turn` does on the way out. A
harness answering with a record containing a newline would otherwise
render across two lines and the second could read as a closer at column
zero. The failure would be a forged block boundary in text the model then
reads as structure, which is worth closing at the one point every
response passes through.

## trait ClientHarness

Blocking rather than async, and generic rather than boxed.

BLOCKING IS DECIDED RATHER THAN PROVISIONAL, and it was carried as debt
until both halves of the seam existed to price it. Making the trait async
buys nothing: Questness is blocking all the way DOWN, since the evaluator
spawns a sized thread and joins it, so an `evaluate` sub-turn blocks its
caller whatever this signature says. An async `serve` would move the
problem onto the evaluator while forcing a runtime choice into the library
everything depends on.

So the whole of `step` is a blocking API and an async daemon's correct
call is `spawn_blocking`. That needs `Questness<H>` to be `Send`, which a
compile-time assertion locks rather than assumes.

Generic follows the design: an instance talks to one harness for its
life, so static dispatch is enough and there is no case for a trait
object.
