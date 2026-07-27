# container.rs

## trait Engine

SYNC AND NOT `Send`, and both halves are forced rather than chosen. The
era-two model holds `Rc` handles in its graph cache and its prefill
scratch, so it cannot cross a thread boundary at all, and one thread must
own it for its life.

That is why `spawn` takes a FACTORY rather than an engine. A `FnOnce`
returning the engine is `Send` even when the engine is not, so the closure
crosses to the thread and what it builds never leaves. Taking an engine by
value would have made the trait unimplementable by the only engine that
matters.

The chunk callback returns a `bool`, which is also the CANCEL mechanism
rather than a courtesy. A false return means nobody is listening any more,
so an implementation stops and reports what it managed - and closing the
receiving end is how a session cancels a turn that is already running,
without a second channel or a shared flag.

## enum Work

One variant per thing a session can ask, each carrying where its answer
goes. The oneshot per unit is what makes the container an actor rather
than a queue with a shared reply path: two sessions cannot receive each
other's answers because neither can name the other's channel.

## struct ContainerHandle

Clone is on the HANDLE and not the model. Every session holds one, and
what they share is the way in rather than the thing itself.

## fn spawn

THE SERIALISATION POINT IS THE LOOP, not something layered over it. The
thread takes one unit, serves it to completion, then takes the next - so
one generation runs at a time because that is the only shape the loop has,
and no caller has to arrange it. 03_Platform's one-resident-model rule and
roughly fourteen gigabytes of weights forbid a second copy regardless, so
there is nothing to parallelise even if the loop allowed it.

## fn dispatch

A gone container is an ERROR rather than a hang, on both legs: the send
fails if the thread is gone, and the wait fails if the thread dropped the
work without answering. A session that hangs waiting on a dead container
would be indistinguishable from a slow model.

## fn serve_one

A dropped answer channel is IGNORED rather than reported. The session that
asked has gone, and the work is already done - there is nobody to tell and
nothing to undo.
