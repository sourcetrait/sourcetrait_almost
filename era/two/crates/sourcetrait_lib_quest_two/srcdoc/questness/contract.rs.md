# contract.rs

The half of the channel grammar that is about MEANING rather than framing: what
a `<nu>` block's def declares, and whether the blocks around it agree with it.

Everything here runs BEFORE anything executes, which is the whole point. Piping
a block into its def is itself a deep conformance check, for free, because
nushell enforces a def's infix INPUT type at runtime to full depth. But that
check happens on the way in, tells you nothing about the output side, and
reports `Input type not supported.` with no field name when it fires. These
agreements are what can be settled while the turn is still text.

## const ARGS_POSITIONAL

## struct NuSignature

## fn takes_pipeline

The distinction is `nothing` against everything else, not presence against
absence. A def with no infix annotation at all reads as `nothing -> any` here,
which is the same answer.

## fn signature_of

THE HEAD IS DISCOVERED BY DIFFING THE DECLARATION SET, not by scanning the
source for `def`, so the answer comes from the parser rather than from a regex
and a def spelled across lines or carrying flags is found exactly as a plain
one is.

Requiring exactly one fresh declaration is the deliberate reading of "a nu block
carries a definition". Zero and several are both rejected, and several matters:
a body declaring two defs has no single form, so there is nothing to bind blocks
against.

`--env` IS NOT REACHABLE FROM THE SIGNATURE, which is the trap here. `def --env`
sets `redirect_env` on the BLOCK rather than adding a named flag, so a
signature's `named` list never carries it and a check written against that list
reads false for every def and fails silently. It is read off the decl's block
instead.

## fn infer_nu

The form is what a signature MATCHES rather than what the def happens to be
called. An earlier revision of this file argued the opposite - that the mode is
whatever the def is NAMED, with no list of known modes anywhere - and that is
superseded: the forms are an enum, and a prototype per form is what selects a
variant.

## fn check_prototype

One prototype fact is stated by the design so far and it is enforced rather than
assumed: `--env` is interact's contract, since env and `cd` changes surviving
into the caller is what interact MEANS. A body carrying it that is not interact,
and an interact without it, are both refused however they are spelled.

The other forms' prototypes are owed. Until they land the head is all that
separates them, and this function is where each one attaches.

## fn check_agreements

Returns an envelope rather than a result, and that is the design rather than
leniency. A repairing model gets one round trip per generation, so reporting the
first fault and stopping costs a whole generation per fault. Every check below
runs even when an earlier one failed.

The three agreements, in the order the design states them.

A `<pass>$in</>` must be present exactly when the infix input is not `nothing`.
Both directions are errors: a missing binding for a def that consumes a pipeline,
and a binding for a def that does not.

A `<pass>$args</>` must be present exactly when a positional named `args`
exists, on the same both-directions rule.

And a bound block's declared type must be a subtype of the channel it binds to.
Subtype rather than equality is what makes `table<...>` and
`list<record<...>>` interchangeable here, which they are everywhere else in the
program, and what lets an empty list bind a concrete list type. The check runs
through nushell's own `CompareTypes`, so the answer matches what a pipeline
would decide rather than what a checker of ours would.

## fn pass_bindings

A pass binds the block IMMEDIATELY BEFORE it, which is what the worked example
shows and is the only reading the emission order supports.

WHAT IS DELIBERATELY NOT DECIDED HERE: the design lists as open what orders two
blocks binding the SAME channel, noting that positional order is the obvious
answer but has not been ruled. So a second binding of a channel is reported as a
duplicate rather than silently ordered, and it is reported once rather than
once per later occurrence. When the ordering is ruled, this is the function that
changes and the diagnostic kind that retires.
