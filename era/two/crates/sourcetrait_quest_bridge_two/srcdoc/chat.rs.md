# chat.rs

The channel-shaped chat engine, and the reason it is shaped this way.

THE MODEL IS NOT `Send`. `OlmoHybrid` holds an `Rc` graph cache, so it cannot
cross a thread boundary at all. One named thread constructs it, holds it
privately for the session's life, and drives generations off the requests
channel - only strings and small structs ever ride the channels. That
constraint is what produced the transport-shaped seam rather than the seam
being chosen and the model made to fit.

THE THREAD NEVER PANICS FOR PROTOCOL REASONS. Every post-load failure is an
`Error` event and the session stays open, because a panic here would poison a
channel a consumer is selecting on and surface as a silent hang rather than a
message.

## const EVENTS_CAPACITY

The engine BLOCKS on a full events channel, which is deliberate backpressure: a
stalled consumer slows generation instead of letting the queue balloon. The
alternative - dropping or unbounded buffering - would either lose transcript or
grow without limit while the model kept running.

## const REQUESTS_CAPACITY

## fn open_chat

Returns as soon as the channels exist. The model loads ON the spawned thread,
so `Ready` follows the load and a load failure arrives as `Error` then
`Closed` - the obligation the API crate's mirror states, satisfied here.

## enum TurnOutcome

A turn can end for reasons that outlive it - a Reset or Close that arrived
mid-generation - so the turn reports what the LOOP should do next rather than
acting itself. That keeps the interrupt handling in one place instead of
splitting it across the turn and the loop.

## fn engine

THE TRAIL IS THE WHOLE MULTI-TURN MECHANISM, and it is worth understanding
before changing anything here. `report.context_ids` is exactly the cache
contents after a turn, so turn two onward continues from it through
`generate_from` over a `RestoredContext` CONSTRUCTED IN MEMORY - no snapshot
file is written or read. The snapshot format exists for persistence; this path
only needs the shape.

`Cancel` between turns is a no-op rather than an error, because nothing is in
flight and a consumer racing an Esc against a finishing turn should not be
punished for it.

## fn load

CHAT IS ALWAYS CAPTURE-OFF, forced here rather than left to the profile. A
chained session regrows the KV buffers, which parks captured graph buffers for
near-zero replay benefit - the era-one ruling, enforced era-side so no profile
can turn it back on for a chat.

Everything else rides the profile, which is what keeps this crate from becoming
a second settings surface.

## fn reset

## fn run_turn

The cancel poll sits BETWEEN decode steps, and dropping the iterator is the
clean early stop - the generation surface is built for exactly that, so no
teardown is owed.

FOUR THINGS END A TURN AND ALL FOUR STILL REPORT. A cancel, a closed consumer,
an engine error and a natural stop each fall through to one `finish` call, so
`TurnDone` is emitted on every path. A consumer can therefore treat it as
guaranteed rather than conditional.

The detokenizer tail is flushed as a final `Chunk` before the report, which is
what keeps a cancelled turn's last partial grapheme from being lost.

An error-broken turn reports `Cancelled` despite no cancel having arrived. That
is a deliberate flattening - the API has no fourth reason and the `Error` event
already carried the detail - and it is the one place a report's finish reason
is inferred rather than observed.
