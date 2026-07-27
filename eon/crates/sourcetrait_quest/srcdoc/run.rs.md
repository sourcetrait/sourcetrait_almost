# run.rs

## fn run

Stdout is the msgpack protocol channel, and that decides everything about
this function. A diagnostic printed there is not a message anybody reads;
it is a corrupt frame in the middle of a protocol, so the failure it
reports is replaced by a stranger one. The startup failure therefore goes
to stderr, which nushell surfaces, and the exit code carries the rest.

There is no fallback to serving without a runtime or without an
evaluator. A plugin that came up and then failed every call would be
harder to diagnose than one that did not come up.
