# serve.rs

## fn record

A logging failure is DROPPED rather than reported, and the rule behind
that is the same one the ThinkHarness side holds to: the log is a record of
the work and never part of it, so nothing a turn does may depend on it
landing. A session with no log at all takes the same path, which is what
lets an unresolvable cache home turn logging off instead of refusing
connections.

## fn serve

ONE BAD CONNECTION IS THAT CONNECTION'S PROBLEM. A handshake failure or a
peer that never speaks ends its own task and nothing else, because a
daemon that a single malformed client can take down is worse than one that
occasionally logs nothing.

The accept loop returns rather than retrying when the LISTENER itself
fails. A listener that stopped accepting will not start again, so looping
on it would spin.

## fn session

The match is exhaustive on purpose rather than by accident of the enum
being small: every message the client language can carry is answered here,
so adding one to that language breaks this rather than being silently
dropped. That is most of the argument for splitting the language by
direction in the first place.

A `Cancel` arriving OUTSIDE a turn answers `stopped: false`. It is not an
error - a client that cancels a turn which has already finished is racing,
not misbehaving, and the answer tells it which happened.

## fn turn

THE INBOUND HALF IS READ WHILE THE TURN RUNS, and this is the whole reason
the turn has a loop of its own rather than a single await. The container
serialises work, so a `Cancel` sent as ordinary work would queue BEHIND
the turn it means to stop and be served after it finished - which is not a
cancel at all. Reading the socket during the turn is what makes the
message reachable.

CANCELLING IS CLOSING THE CHUNK CHANNEL. The engine's next emit fails, so
it stops and reports what it managed, and the turn still answers its own
request. No second channel, no shared flag, and no way for the two to
disagree about whether a cancel happened.

The loop's exit condition is the report arriving AND the chunks draining,
not either alone. The container drops its chunk sender just after it sends
the answer, so the two orders are both possible and a loop keyed on only
one of them would either truncate the tail or hang.

A second turn arriving mid-turn is REFUSED rather than queued. Queueing it
would make the daemon hold work it cannot start on behalf of a client that
has not been told, and the client already has a fault notice to read.

THE ANSWER IS REASSEMBLED ONLY TO LOG IT. Chunks still go out as they
arrive, so the copy costs memory rather than latency, and the alternative
- logging each chunk as it passes - would record the tokenizer's grapheme
boundaries rather than the answer anyone read.

EVERY RECORD IS WRITTEN BEFORE ITS RESPONSE IS SENT, which is what makes
the log testable without waiting on anything: a client that has read the
turn response has already caused the turn, the answer and the report to
land. The answer is also recorded on the path where the peer vanished
mid-turn, because a partial answer is the interesting one to have kept.

THE RECORDS ARE ROUTED BY GRAIN, and the routing is the readable half of
the split. Transport takes the connection conversation, turn takes what a
turn DID in a line or two, and emission takes the bulky pair - what the
model was fed and what it wrote. A think leaves a marker on the turn log
and its text on the emission log rather than one record on both, which is
what keeps the turn file a readable outline of a multi-generation turn
instead of a wall of thought text.

THE CHUNK SINK IS THE ONE RECORD WRITTEN WHILE THE MODEL WORKS. Every
other record here lands only once a generation has ended, so before it
existed a turn running to the 32,768 budget was indistinguishable from a
hung one for as long as it mattered - the log was correct, complete, and
silent. It is opened per generation and held, because it is written per
token rather than per turn.

That does not make the reassembly above redundant. The sink carries the
stream as it arrives, tokenizer boundaries and all; the emission record
carries the answer someone read. They answer different questions and the
cost of both is one copy.

It is also the reason the sink writes VERBATIM. Framing each piece as a
labelled record would make the file complete and useless, because the
whole point is that a `tail -f` reads as the answer forming.

TEARDOWN IS THE TURN'S DEFAULT END, in two places that guard different
failures. The terminal teardown after a delivered answer is the ruling
itself - a conversation ends with its turn unless the turn's config
kept it. The fresh-start teardown ahead of a bare turn is the second
line: it clears whatever a kept run left standing, a pending ask a
caller abandoned, or a trail a failed reset leaked - which is why a
failed reset only logs a fault rather than failing the turn. An ask
never tears down (the conversation is paused, not over), and every
failure path tears down regardless of keep, because a conversation
whose last turn died mid-step is not one a continuation can trust.

## fn teardown
