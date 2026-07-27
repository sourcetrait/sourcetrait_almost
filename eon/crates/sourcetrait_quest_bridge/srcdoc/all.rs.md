# all.rs

The vocabulary both directions of the wire share. Everything here is DATA, and
that is the whole of what this file became.

It used to carry a trait an era implemented plus a channel pair a consumer
held, which described an era owning the model inside the consumer's own
process. The daemon owns it now, so there is nothing for an era to implement
and nothing for a consumer to hold: what crosses is messages, and this file is
the types those messages are built from. Removing the trait removed the last
reason for a per-era implementation crate to exist.

The types stayed because the WIRE carries them. They are the payload of
`OpenResponse`, `TurnResponse` and `OpenRequest`, so they are shared by both
languages rather than owned by either, which is why they sit here rather than
in `wire.rs` beside the messages that carry them.

## struct EraInfo

## struct ChatOptions

THE DAEMON DOES NOT CONSULT THESE, and that is a property of the singleton
rather than an unfinished path. One container owns one model for the service's
life, so a session cannot select a checkpoint or a settings profile; what it
can do is learn which one it got, which is what `EraInfo` answers.

A decode budget is the field that reads like it should still work and does not.
It could only be stored on the shared engine, where one session would silently
change another's. Threading it per TURN is the shape that would work, and that
is a wire change rather than something to fake here.

The fields stay because the wire's `OpenRequest` carries them and because the
question they ask is the right one; what changed is who can answer it.

## struct TurnReport

## enum FinishReason

`Cancelled` is a first-class ending rather than an error, which is what lets a
consumer report a stopped turn in the same place it reports a finished one. On
the daemon side a cancel arrives as a closed chunk channel, so the engine stops
and still produces a report - the accounting is real rather than synthesised.
