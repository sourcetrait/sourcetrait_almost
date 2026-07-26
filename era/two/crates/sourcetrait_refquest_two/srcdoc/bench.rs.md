# bench.rs

One timed performance row per invocation, and the two backends reach it
differently.

The transformers row rides the generation verb rather than a driver of its own,
because that driver's timing and VRAM events already are the row. Building a
second path would mean a second measurement of the same thing, which is how two
numbers that should agree stop agreeing.

The vllm row has its own driver, since a serving engine's timing is not a
generation's.

## fn bench

The transformers branch constructs the generation arguments explicitly rather
than sharing a struct or defaulting, and three of those values are the bench
posture rather than a copy: raw prompting so no template intervenes, greedy
decode, and stops ignored so the row decodes exactly the requested count. That
last is what makes the token rate a rate rather than an average over a reply that
ended early.

The sampling values are set even though sampling is off, which is harmless
because the generation verb only passes them when it is on. Filling them keeps
the struct construction total rather than partial.
