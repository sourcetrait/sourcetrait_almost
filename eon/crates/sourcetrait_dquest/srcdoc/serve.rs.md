# serve.rs

The module-level allow is temporary and named at the top of the file: the
code is built and locked but not reached from `run`, because serving needs
a container and a container needs an engine, which is the era library's.

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
