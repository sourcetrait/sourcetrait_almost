# nu_sandbox.rs

Running nushell that we did not write.

Three of the task families are verified by execution - the answer is a pipeline,
and whether it is right is whether it produces the right value. That means
running model output, so it runs under bubblewrap with the root read-only, no
network, no session, and dying with its parent.

In-process evaluation was never an option here even though the library already
carries the nushell crates. A verifier runs generated code, and generated code is
arbitrary: an in-process eval would hand it the trainer's own address space,
filesystem and network. Bubblewrap is unprivileged, already present because
Flatpak ships it, and alters nothing.

## const SANDBOX_BIN

Absent is a hard error rather than a silent fallback to running unsandboxed. The
whole point is that model output never runs unconfined, so the degraded mode does
not exist.

## const NU_BIN

Resolved on PATH rather than pinned to a path, so the estate's pinned nushell is
what grades an answer. That couples the verifier to the estate's version on
purpose: a pipeline is correct against a version, and the corpus, the batteries
and this grader are meant to move together.

## const ROOT

## const DEV

## const PROC

Platform paths rather than ours, so they are required to exist rather than
created. Creating one would turn a failed assertion about the platform into a
fabricated success.

## const DEFAULT_TIMEOUT

## const POLL

## struct NuOutcome

`timed_out` is carried beside `ok` rather than folded into it, because the two
mean different things to a generator: a failure is an answer that is wrong, while
a timeout is a template that hangs, and the second is a defect in the fixture
rather than in the answer.

## fn require_sandbox

Called once before a verification pass rather than per item, so a missing
sandbox fails once and loudly instead of once per item.

## fn run_nu

The wait loop polls rather than blocking, which is what makes the timeout
reachable at all: a blocking wait on a non-terminating pipeline never returns,
and a generated pipeline can loop forever. The child is killed on the deadline
and `wait_with_output` then reaps it.

`--no-config-file` matters for reproducibility rather than for safety. A user's
config could define commands, aliases or environment that change what a pipeline
means, so the grader runs against the bare language.

## fn pipeline_value

The comparison is on values, never on stdout text. Two correct pipelines can
render the same value differently - a table prints as a bordered grid and as a
NUON literal - so this appends `to nuon` and the caller compares parsed values.
Comparing the rendered text would score formatting and call it correctness.

The pipeline is parenthesised rather than concatenated, because a trailing
semicolon or comment in the model's answer would otherwise swallow the appended
stage and the run would return nothing rather than a value.

A non-NUON result reads as `None` alongside a failure and a timeout. The caller
grades all three the same way, and that is correct for a verifier: an answer that
does not produce a comparable value has not answered.
