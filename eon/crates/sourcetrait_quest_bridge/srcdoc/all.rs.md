# all.rs

The contract between eon and an era. Everything here is deliberately small:
this file is what a future era has to satisfy, so anything added becomes a
permanent obligation on every era after it.

## trait Era

Implementations live in per-era crates, and eon components reach an era through
this trait ALONE - never through era internals. That is the property the whole
bridge architecture exists to buy, and it is why camp can name `BridgeTwo` in
one line and otherwise know nothing about era two.

The trait is generic over nothing and object-safe in practice, so a consumer
holds one era for its life. A given camp process talks to one era; swapping is
a rebuild rather than a runtime dispatch, which is the honest shape given a
resident model.

### fn info

Identity WITHOUT loading anything. That separation is what lets a consumer
print what it is about to open before paying for the model, which is most of
what camp's pre-session stderr line is for.

### fn open_chat

Returns as soon as the CHANNELS exist, not when the model is live. The model
loads on the session's own engine thread and announces itself with `Ready`, so
a consumer opens in a loading state and a load FAILURE arrives as `Error` then
`Closed` rather than as an error from this call.

A consumer that treats the return as readiness will paint a usable interface
over a model that is not there yet.

## struct ChatSession

CHANNELS RATHER THAN CALLBACKS, and the choice is transport-shaped on purpose.
The in-process pair is meant to swap later for a socket and TLS boundary - the
dquest server - without touching a consumer, which a callback interface could
not do.

DROPPING `requests` CLOSES THE SESSION. That is the implicit teardown path, and
it means a consumer that panics or returns early still ends the engine rather
than orphaning it; the engine answers `Closed` and exits.

## enum ChatRequest

The turn text is RAW. An era renders its own chat protocol, because the
template belongs to the checkpoint rather than to the consumer, and a consumer
that pre-rendered would silently break on an era whose template differs.

## enum ChatEvent

`Chunk` may legitimately carry EMPTY text while a multi-token grapheme is
pending, so a consumer must not treat an empty chunk as an end or an error.

`Error` leaves the session OPEN and `Closed` does not, which is the whole
distinction between them: the first is recoverable and the second is terminal.
A consumer that collapses the two loses the ability to keep chatting after a
turn fails.

## struct EraInfo

## struct ChatOptions

The three profile tokens and a decode budget, and nothing else. ERA PROFILES
OWN THE SETTINGS, so this is not a knob farm - a sampling posture, a graph
toggle or an eviction table is configured where the rest of the suite
configures it, and adding a field here would fork that.

## struct TurnReport

## enum FinishReason

`Cancelled` is a first-class ending rather than an error, which is what lets a
consumer report a stopped turn in the same place it reports a finished one.
