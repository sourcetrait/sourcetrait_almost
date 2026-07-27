# container.rs

The trait this file is built around is the API's rather than ours, so why
it is shaped as it is lives in that crate's mirror. What is recorded here
is what consuming it costs.

## enum Work

One variant per thing a session can ask, each carrying where its answer
goes. The oneshot per unit is what makes the container an actor rather
than a queue with a shared reply path: two sessions cannot receive each
other's answers because neither can name the other's channel.

## struct ContainerHandle

Clone is on the HANDLE and not the model. Every session holds one, and
what they share is the way in rather than the thing itself.

## fn spawn

It takes a factory rather than an engine, and that is forced by the
era-two model holding `Rc` handles: it is not `Send` and cannot cross a
thread boundary, while a `FnOnce` returning it is. The closure crosses,
the engine is built here, and it never leaves.

THE SERIALISATION POINT IS THE LOOP, not something layered over it. The
thread takes one unit, serves it to completion, then takes the next - so
one generation runs at a time because that is the only shape the loop has,
and no caller has to arrange it. 03_Platform's one-resident-model rule and
roughly fourteen gigabytes of weights forbid a second copy regardless, so
there is nothing to parallelise even if the loop allowed it.

The failed-load arm is a second loop rather than an early return, and that
is the daemon's staying-up property expressed at the only place that can
express it. The thread has already been spawned and the handle already
returned by the time a load fails, so refusing each unit with the reason
is what a caller meets instead of a channel that closed.

## fn dispatch

A gone container is an ERROR rather than a hang, on both legs: the send
fails if the thread is gone, and the wait fails if the thread dropped the
work without answering. A session that hangs waiting on a dead container
would be indistinguishable from a slow model.

## fn serve_one

A dropped answer channel is IGNORED rather than reported. The session that
asked has gone, and the work is already done - there is nobody to tell and
nothing to undo.
