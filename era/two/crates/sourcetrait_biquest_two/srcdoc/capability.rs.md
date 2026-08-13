# capability.rs

The capability runner: drive the engine over rendered olmo-eval fixture requests
and emit predictions in their own shape.

This file's whole job is boundary fidelity. A reading is only trustworthy if the
boundary matches the reference exactly, so the contracts below are mirrored from
olmo-eval's source rather than reimplemented from its documentation, and each one
is a place where a reasonable implementation would differ and produce numbers that
look plausible.

The encode boundary: trailing whitespace moves from the context to every
continuation before tokenization, continuation ids are what remains after the
context's own encoding, and the fed ids are the context ids followed by the
continuation ids. That last is deliberately not a re-encoding of the whole string,
because the two differ across a BPE boundary.

The field discipline: the logprob-derived keys appear only on log-likelihood rows,
while generation rows carry text, extracted answer, character count and the greedy
flag alone. The reference's own prediction builder gates on that, so an extra key
is not merely untidy.

## const PREFILL_CHUNK

## const CAPABILITY_HOME_RELATIVE

## struct RequestRow

## struct RequestBody

## struct GenerationKwargs

Fields this runner does not consume are ignored rather than denied, which is the
right posture for a foreign artifact whose shape upstream owns. The `context` field
stays a raw JSON value because it is genuinely two shapes - a string for
completions, a message list for chat - and the branch that decides is in the
runner.

## struct ModelOutput

## struct PredictionRow

The optional fields serialize only when present, which is what reproduces the
reference builder's field discipline in one place rather than in two code paths.

## fn data_home

Honours the XDG spec's own fallback rather than requiring the variable, so the
tool works on a machine that does not set it.

## fn encode

## fn ids_tensor

## fn row_pick

A log-softmax pick on one host row, returning the log-probability and whether the
token was the row's argmax. It is computed host-side rather than on the device
because the caller needs one value per position and the transfer of a whole logits
row is already paid.

The maximum is subtracted before exponentiating, which is the standard stability
form and matters here because these are raw logits rather than a normalised
distribution.

## fn capability_run

Input format is detected at source rather than flagged: a requests tree carrying
NUON files runs the NUON path, and JSONL otherwise. That freezes the original
behaviour for the original trees while letting the mirrored ones run through the
same verb.

The NUON path reuses the typed runner internals verbatim through the serde seam -
a NUON row converts to JSON, deserialises into the request type, runs the same task
functions, and converts back. There is therefore zero logic drift between the two
paths by construction, which is a stronger guarantee than a test could give, and a
full-grid rerun reads exactly value-equal to the conversion of the JSONL run.

The output directory is derived from the settings token, so two postures cannot
overwrite each other's predictions.

## fn capability_bridge

The reverse adapter, for comparisons against the reference scorer. It exists
because that scorer is comparison-only now that the Rust scorers exist, and it
consumes JSONL.

## fn collect_request_files

## fn read_rows

## fn prediction_path

## fn run_loglikelihood_task

The shared context prefills once, all but its final token, in chunks. Each
continuation then forwards the last context token followed by its own ids and
rewinds through the library's mark and rollback pair, so row `j` of the block
predicts continuation token `j` under the all-position logits contract.

That structure is what makes a multiple-choice item one prefill rather than one per
choice, which is most of why a sixty-task grid finishes in half an hour.

The rollback happens before every continuation but the first, rather than after
each, so the mark is taken once and the cache is left holding the last
continuation - which nothing reads.

## fn run_generation_task

Greedy only, and a sampled request raises rather than being silently run greedily.
A sampled reading compared against a greedy reference would be a different
measurement wearing the same name.

Stop sequences truncate client-side at the earliest occurrence, which is the
reference's own behaviour rather than the engine's stop handling. Chat contexts
render through the byte-identical template and string contexts feed verbatim.

The single-user-message restriction is a genuine limit of this version rather than
a property of the fixtures, and it raises rather than rendering only the first
message.
