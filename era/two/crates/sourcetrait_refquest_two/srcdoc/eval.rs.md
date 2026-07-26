# eval.rs

The evaluation batteries, routed to one of two python drivers by backend.

One suite exists so far, and the subcommand nesting is there because the suites
differ in their arguments rather than in a parameter. A second battery arrives as
a sibling variant rather than as a flag on this one.

## fn eval

## fn eval_needle

The two backends are two different drivers rather than one driver with a switch,
which follows the pinned stacks: the transformers path and vllm agree about
almost nothing operationally, and their arguments barely overlap.

The transformers branch carries the CPU-grade handling - the fla blocker forced
on CPU, and the GPU hidden - exactly as the generation verb does.

The vllm branch instead carries its two capacity levers. The utilisation fraction
is how much of the card the engine may claim, and its ceiling beside a live
desktop is about 0.86 rather than something near one. The long-context escape
sets the allow-long environment variable, which is mechanically safe on this
model because it has no positional encoding at all - the config's length is
post-training-stage metadata rather than a mechanism.

One process runs one length batch, which is the caller's discipline rather than
this function's. Beyond 24K the transformers path needs the expandable-segments
allocator option and still runs out of memory at 32K; the reliable vllm ceiling
is the 32K row, and the zone above it dies unreliably inside an unguarded prefill
transient.
