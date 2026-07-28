# converse.rs

The turn loop lives here, and that placement is the file's one open
question rather than a settled answer.

The design puts it in the daemon: the daemon holds the weights and calls
`step`, and Questness owns the conversation about the model rather than
the model itself. What the wire carries today is a text turn and a text
answer, so a `TurnRequest` cannot yet carry a config, a prompt and its
bindings - and until it can, the only place that can assemble a turn is
the side holding the caller's arguments. Moving the loop across is that
wire change rather than a rewrite of this file: the loop below is what
the daemon would run, unchanged.

Two consequences ride with the placement, and both are gaps rather than
choices.

A sub-turn result goes back as a USER turn, because the continuation the
era library renders is a user turn and the wire carries no role. The
design wants an environment turn, which is what the chat template renders
for a tool result and what the model's own instruct data trained. Nothing
here can fix that without the wire carrying which role a turn is.

Insufficiency can only ever be answered no. The tag is a special token,
so a skip-special decode strips it and there is nothing in the text to
find at any level of effort - detection belongs where the token ids are,
which is the engine, and the wire is what would have to carry the
verdict. `step` is therefore called with a false this side did not
measure.

## const GOODBYE

## fn ask

Takes the questness lock for the whole call, so one plugin process serves
one turn at a time. That is not a limitation being accepted quietly: the
daemon's container serialises generation anyway, so a second concurrent
turn would queue behind the first at the far end regardless. Holding the
lock makes the queueing visible at the near end instead of looking like
concurrency that is not there.

The guard is held across awaits, which is legal because `block_on` polls
on the calling thread and never requires the future to be `Send`. That is
worth knowing before anyone reaches for `spawn` here, where it would stop
compiling for a reason that reads as unrelated.

## fn converse

The close runs on EVERY path, including a failed one. A daemon reading a
dropped socket cannot tell a crashed client from a finished one, and the
transport says goodbye precisely so it can - throwing that away on the
error path would mean every failure looked like a crash.

The outcome is captured before the close rather than returned through it,
so the close cannot replace a real failure with a transport one.

## fn drive

## fn connect

The options are absent from the open because the daemon cannot honour
them: one container owns one model for the service's life. Sending a
decode budget would read as a request that was quietly ignored, which is
worse than not asking.

## fn open

## fn turn

The chunks are REASSEMBLED rather than streamed onward, and that follows
from what an answer is here rather than from convenience. The answer is a
typed value plus a rendering over it, and neither exists until the whole
emission is parsed - a rendering cannot be produced a token at a time
because the template that produces it arrives after the value. Streaming
the raw emission alongside the parsed answer is a product decision rather
than something this owes.

A plugin returns one `Value`, so there is nothing to stream into even if
there were something to stream.

## fn refuse_unexpected

A shutdown notice ends the conversation because reconnecting will not
help, and a fault does not because the session stays open. Anything else
is a message this consumer does not service - a reset it never sent - and
is passed over rather than promoted to a fault of its own. Treating an
unexpected message as an error would make every future widening of the
server's language a breaking change here.

A cancel response now IS something it asked for, and is still passed over
on the same terms: the acknowledgement says the cancel arrived, while the
thing the caller is waiting on is the turn's own answer, which comes
afterwards either way.

## The interrupt path

AN INTERRUPT IS ONLY OBSERVABLE BY ASKING FOR IT, and not asking was the
whole defect. Ctrl-C reaches the ENGINE rather than this process, so a
plugin that never consults the interface never learns one happened - and
this one then sat in an uninterruptible `block_on` until the daemon
finished on its own. A turn could not be abandoned from the terminal at
all, which is worst in exactly the cases where you most want to: a
generation running away, with minutes left to go.

THE FLAG IS POLLED RATHER THAN AWAITED, and that is forced. A signal
handler is a callback on another thread and cannot reach into the task
that is awaiting, so the two meet through an atomic flag beside the
receive. The interval bounds only how long a Ctrl-C takes to become a
`Cancel` on the wire; the daemon then stops at its next emit.

THE FAR HALF NEEDED NOTHING. `serve::generate` already reads inbound
during a turn precisely so a cancel is reachable mid-generation, and a
lock already drives a hundred thousand scripted chunks cancelled in
flight. The granularity was spent when the wire was designed and simply
never claimed here.

The cancel is sent ONCE rather than per poll, and the loop keeps reading
after it. A cancelled turn still ANSWERS - the daemon stops and reports
what it managed - so returning at the cancel would abandon a response
that is already on its way, and repeating the cancel would put a message
on the wire per poll for as long as the turn takes to wind down.

WHAT IT DOES NOT REACH is the two unbounded paths in debt. A think loop
never returns to the point where the response is awaited, and a hung
evaluator blocks the turn owner before any generation starts, so neither
is listening for a cancel when it matters.
