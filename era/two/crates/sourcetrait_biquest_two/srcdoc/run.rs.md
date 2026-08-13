# run.rs

The whole dispatch is one nested match over the parsed command tree, and it is
deliberately the only place that shape exists. Every verb is a plain function
taking the parsed arguments, so the tree can be re-categorised without touching
a verb.

The exit path prints to stderr and exits nonzero rather than returning a result
to `main`. That keeps the process's failure visible to a shell driving it, and
it is why `main` can stay two lines.

## fn run

The global flagset is passed as `&cli` to the verbs that resolve a profile from
it, and withheld from the verbs that do not touch the model. Reading the call
sites is therefore how you tell which verbs load a checkpoint.

## fn train_dispatch

Two definitions under opposite feature gates, which is what makes a non-train
build fail with a sentence rather than with an absent subcommand. The verbs stay
in the parser either way, so `bquest doc cli` renders one tree per era rather
than one per build.

That gate is also the seam behind a real constraint: the trainer's burn-cuda
feature set and the engine's candle-cuda set cannot combine on this box, so one
source tree ships as two separately-built artifacts. Pointing a stage at the
wrong build reaches this arm.
