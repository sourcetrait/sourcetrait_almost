# arbitrate.rs

Questness as an object rather than a set of functions: it holds the
evaluator and the client harness, and it decides which of them a sub-turn
belongs to. This is the module that closes the loop the other three open.

Experimental, and the shape here is the least settled in the crate,
because it is the one a daemon will actually drive and no daemon exists
yet.

## enum Step

The states a turn can be in AFTER an emission, which is deliberately not
the same set as `turn::Outcome`. `Outcome` says what the model's text
meant; `Step` says what the caller does next. The difference is exactly
one arm: a sub-turn is an outcome but never a step, because by the time
`step` returns, the sub-turn has already run and what is left is text to
send back.

That is the whole reason this type exists rather than re-exporting
`Outcome`. A caller holding a `Step` cannot forget to service a sub-turn,
because there is no variant that would let it.

## struct Questness

Owns the evaluator rather than borrowing one, because the evaluator's
whole design is build-once-clone-per-evaluation and the owner of the turn
is the natural place for that "once" to live.

The model is NOT here, and that is the significant structural choice.
Under 17_End_Use the model lives in the daemon's singleton container, so
Questness cannot own it without dragging the engine into every consumer.
What Questness owns instead is the conversation ABOUT the model: it
renders what goes in, reads what comes out, and services what that asks
for. The daemon holds the model and calls `step`.

The consequence worth noting is that this makes Questness testable with
no model at all, which is why the arbitration locks run in milliseconds
against a stub. That was not the reason for the design, but it is a good
sign about it.

## fn step

One emission in, one state out. The sub-turn arm is the only one that
does work rather than translating.

`insufficient` is threaded through rather than detected here for the
reason `turn::interpret` records: the tag is special and a skip-special
decode removes it, so only a caller holding token ids can know.

## fn serve

Where the boundary from 16_Questness becomes two lines of code. An
`Inside` destination runs on the evaluator with no client involved; a
`Client` destination crosses the seam. Everything else about the two
paths is identical, including that both results come back through the
same `as_block` conversion - which is what makes 10_Channels' ruling true
in practice rather than in intent: a result returning from `<nu>` comes
back TYPED through `<output>`, and it does so whichever side ran it.

An evaluator failure is turned into a `HarnessResponse::failed` rather
than propagated as an `Err`. That keeps one shape for "the thing did not
run" regardless of which side it did not run on, and it puts a nu error
in front of the model as feedback it can repair from, rather than
collapsing the turn. The `Err` channel stays reserved for the harness
itself breaking.

A failing response becomes `Repair` rather than `Continue`. The
alternative - handing the model a block describing the failure - was
rejected because the model would then have to distinguish a result from a
report of a non-result, in a grammar where `<output>` means a value. The
envelope already exists for exactly this.

## fn request_for

The only place a `SubTurn` becomes a `HarnessRequest`, so the mapping
lives once. It is public because the daemon will want to inspect a
request before serving it, and because a harness implementation's tests
want to build one the same way the real path does.
