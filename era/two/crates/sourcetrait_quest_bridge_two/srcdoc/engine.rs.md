# engine.rs

The whole of what era two owes eon. Everything above this file is generic
over the API's traits, so the crate's job is to satisfy them and add
nothing the era library does not already provide - a bridge that grew
logic of its own would be the thing to correct.

## struct BridgeTwo

A unit struct carrying no state, because an era is an identity rather than
an object. What a consumer needs from it is a name it can instantiate the
API's generic at, and everything it can do is an associated function.

## impl Era for BridgeTwo

### fn info

From constants rather than from a load, which is what lets a consumer say
what it is about to open before paying fourteen gigabytes for the answer.
The pair it reports differs from `open`'s in one way worth knowing: this
one names the checkpoint the era is built against, while `open` names the
one that actually loaded, and a configuration pointing elsewhere is
exactly the case where those two disagree.

### fn engine

The factory the container takes, and the reason it is a factory. The
era-two model holds `Rc` handles in its graph cache and its prefill
scratch, so it is not `Send` and cannot cross a thread boundary - but a
`FnOnce` returning it is `Send`. The closure crosses, the engine is built
on the thread that will own it, and it never leaves. Handing over a loaded
engine would have made this trait unimplementable by the only engine that
matters.

The options ride the FACTORY rather than the session, which is what makes
a consumer's own posture reachable at all. A consumer naming its own
`dir` reads profiles from its own root, so it decides which adapter and
which settings get loaded without writing into the suite's shared
default - where the same choice would silently reach every other consumer
of that default rather than only the one that made it.

## struct TwoEngine

Holds the trail as well as the model, because the era library's
continuation surface takes the consumed id trail rather than keeping one
of its own. A daemon session is a conversation, so something has to carry
it between turns, and the engine is the only thing that outlives one.

It also holds the checkpoint name it loaded, which is not redundant
against `Era::info`: that answers from a constant and this answers from
the configuration that was actually read.

## fn load

Called through `Era::engine` on the thread that will own the result, which
is what makes a not-`Send` model legal here at all. Nothing about this is
callable from a daemon's async side, and it must stay that way.

Graphs are off, deliberately, and for a different reason than in a
one-shot binary. Captured graphs bake buffer addresses, and a daemon
restores and clears across turns for the life of the process rather than
generating once and exiting - so the epochs that retire a capture would be
routine here rather than exceptional.

The call order is the era library's rather than ours, and it is not
arbitrary: the token map is verified before the weights are loaded, so a
checkpoint whose vocabulary does not match fails in a second rather than
after fourteen gigabytes of mmap.

THE PROFILE ROOT COMES FROM THE CALLER rather than being resolved here.
Passing nothing reads the suite's own root, which is what every consumer
got before and still gets by saying nothing; naming a root reads that one
instead.

The trap either way is what an ABSENT profile means. A missing
`config/default/lib.toml` falls through to the embedded defaults, and
those carry no adapter at all - so a consumer that has not written a
profile is running the plain base, bit-exact, and nothing anywhere in the
load path says so. It cost a daemon run measured against the base while
everyone believed it was on the supervised adapter.

## impl Engine for TwoEngine

### fn open

The options are not consulted, and that follows from the singleton rather
than being an oversight. One container owns one model for the service's
life, so a session cannot ask for a different checkpoint or a different
settings profile. What it can do is learn which one it got, which is what
the answer carries.

The same TYPE is consulted at build time, through `Era::engine`, which is
the only place its profile fields can be obeyed. What reaches here is a
session's own copy of them, arriving after the model is already loaded.

The consequence worth naming: per-session generation options are not
expressible today. A sample length arriving in `ChatOptions` would have to
be stored somewhere, and the only somewhere is the engine, which is shared
- so one session would silently change another's. Threading it per turn
rather than per session is the shape that would work, and that is a wire
change rather than something to fake here.

### fn turn

The branch on an empty trail is the only place a first turn and a
continuation differ, and it is deliberately not a stored flag: the trail
is the state, so asking whether it is empty asks the real question rather
than one kept in step with it.

The chunk callback's return value is the cancel, so the loop breaks on it
rather than treating it as an error. `listening` then decides whether the
detokeniser tail is worth emitting: a cancelled turn has nobody to emit it
to, and sending it anyway would append text after the client already
stopped reading.

An empty chunk is skipped rather than forwarded. The era library's
incremental detokeniser holds back a partial codepoint and yields nothing
that step, so forwarding it would put empty messages on the wire for no
reader.

The trail is taken from the report unconditionally, cancelled or not,
because the caches advanced either way and the next turn has to continue
from where this one actually stopped rather than from where it meant to.

## fn message_of

Library errors cross as text, and that is the seam doing its job: an eon
consumer must not have to name an era's error type to report a failure, or
the API would leak the library it exists to hide.

It is named for what it produces rather than `text`, because `turn` has a
parameter called `text` and a parameter shadows a free function of the
same name. In a higher-order position that shadowing compiles as something
else entirely rather than failing where you look - a free function and a
`&str` are both callable-looking to `map_err`, and the error names the
trait bound rather than the shadowing.
