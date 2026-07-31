# arbitrate.rs

ThinkHarness as an object rather than a set of functions. It holds the
Thinkspace's evaluator and nothing else, and the one thing it RUNS is a
think turn. This is the module that closes the loop the other three open.

Experimental, and the shape here is the least settled in the crate,
because it is the one a daemon will actually drive.

## enum Step

The states a turn can be in AFTER an emission, which is deliberately not
the same set as `turn::Outcome`. `Outcome` says what the model's text
meant; `Step` says what the caller does next.

`Continue` is the ONLY arm that keeps the turn going, and it carries a
thought turn. Everything else hands control back: an answer, an ask, an
insufficiency, or an envelope to repair from.

`Think` is an outcome but never a step, for the reason that shape exists:
by the time `step` returns, the think has already run and what is left is
text to send back. A caller holding a `Step` cannot forget to service one,
because no variant would let it.

`Ask` is the opposite and it is the correction this module was rewritten
for. It IS a step, because ThinkHarness does not service it and must not:
the caller's own engine runs it, and the turn pauses until a later
request carries the result. An earlier version of this file described a
`serve` that crossed a seam to a client harness, and that seam was never
part of the design - it was mine.

## struct ThinkHarness

Owns the evaluator rather than borrowing one, because the evaluator's
whole design is build-once-clone-per-evaluation and the owner of the turn
is the natural place for that "once" to live. Being one per Thinkspace is
what makes that ownership right rather than incidental.

The model is NOT here, and neither is a harness. The model lives in the
daemon's singleton container, which is about WEIGHTS; the conversation
belongs to the Thinkspace's ThinkHarness. What ThinkHarness owns is the
conversation about the model: it renders what goes in, reads what comes
out, works out what it can for itself, and hands back anything it cannot.

The consequence worth noting is that this makes ThinkHarness testable with
no model and no harness at all, which is why the arbitration locks run in
milliseconds.

## fn step

One emission in, one state out. The think arm is the only one that does
work rather than translating.

`insufficient` is threaded through rather than detected here for the
reason `turn::interpret` records: the tag is special and a skip-special
decode removes it, so only a caller holding token ids can know.

## fn think

Runs the form on the Thinkspace's evaluator and renders the answer back
as a thought turn carrying a typed `<output>` block. That is 10_Channels'
ruling made true in practice rather than in intent - a result from `<nu>`
comes back TYPED rather than as opaque text.

A failure becomes `Repair` rather than a block describing the failure.
The model would otherwise have to distinguish a result from a report of a
non-result, in a grammar where `<output>` means a value, and the envelope
already exists for exactly this. The `Err` channel stays reserved for the
harness itself breaking rather than for a mode that did not run.

## fn repl

Nothing binds and nothing is checked, and both absences are the mode
rather than an omission. A repl declares no contract, so there is no
signature to agree with and no `<pass>` to pair; what it offers is an
exact answer where the model would otherwise infer one.

The rendering is NUON text rather than nushell's own table renderer,
which is a real limitation and worth knowing before trusting it on
structured output. For the case the mode exists to serve it is exactly
right - a scalar's NUON text IS what a command line shows - and a box
renderer is not in this library to reach for.

A failure becomes a repair envelope for the same reason a think's does:
in this grammar a returned block means a RESULT, and the model should not
have to tell a result from a report of a non-result.
