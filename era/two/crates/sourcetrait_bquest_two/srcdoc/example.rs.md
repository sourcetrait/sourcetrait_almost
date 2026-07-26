# example.rs

Instruction-shaped training examples: the on-disk shapes, their ingest, and the
packing that turns them into trainable rows. The verifiers live here too, because
a verifiable prompt carries the name of the one that grades it.

Packing one example per row is forced rather than chosen. The continued-
pretraining path packs a document stream - join everything, cut fixed windows,
shuffle - and that is wrong for an instruction example, because the cut falls
wherever it falls and would split a prompt from its response into two
independently shuffled rows. That is a defect with no symptom: the run completes
and the loss descends.

So an example is rendered, encoded, masked at its assistant positions, padded to
the window, and kept whole or dropped. The loss mask rides a second tensor beside
the ids, which is why the pack loader had to start rejecting tensors it does not
recognise - it previously read one named tensor and ignored the rest, so a mask
written beside the ids would have been skipped in silence.

## const SFT_TYPEDEF

## const DPO_TYPEDEF

## const RLVR_TYPEDEF

Three shapes for three stages, and they are deliberately not one shape with
optional fields. Each stage's method, objective and content differ, so a row that
could serve two stages would be a row that fits neither well.

## const TENSOR_IDS

## const TENSOR_LOSS_MASK

## struct DpoExample

## struct RlvrExample

## struct PackedRow

## fn read_messages

An unknown role raises rather than being dropped. A dropped turn changes the
conversation the model is trained on, silently, and the renderer's role set is
the checkpoint's own.

## fn load_table

## fn as_record

## fn load_sft

An example with no assistant turn raises. Its mask would be empty, so it would
train nothing while looking like an example.

## fn load_dpo

## fn load_rlvr

The verifier name is checked at load rather than at grading time, so an unknown
verifier fails before a run rather than partway through one.

## fn encode_supervised

The span boundaries come from the renderer rather than from re-measuring a
rendered prefix, so they are exact by construction.

That is a deliberate departure from the reference tooling, and the reason is
measured. open-instruct locates a mask boundary by re-rendering the conversation
prefix and taking its token count, and this checkpoint's template renders an
assistant turn differently by position - end-of-sequence when last, `<|im_end|>`
plus a newline when interior. The offset error is real, exactly one token, and
scales linearly with the number of interior assistant turns. Returning spans
means one render per example rather than n+1, no positional invariant to
maintain, and boundaries that cannot drift.

A span covers its turn's terminator and stops there, so the newline that opens
the next turn is excluded. That exclusion is a decision rather than the inherited
off-by-one: our harness re-renders the whole conversation each turn, so the model
is never asked to produce that newline.

## fn encode_prompt

## fn pack_row

Truncation is deliberately not offered. A clipped response teaches the model to
stop mid-answer, and dropping is visible in a report where clipping is not.

## fn write_pack

Rows are `width` wide, and the trainer takes `ids[..width-1]` as inputs against
`ids[1..]` as targets under `mask[1..]`. The shift is the reader's job rather than
the writer's, so the artifact stays a plain aligned pair.

## fn mix_instruct

An example rendering with no supervised position raises rather than being dropped,
which is a different case from being too long: too long is a data-shape fact,
while an empty mask means the render produced no assistant tokens and something is
wrong with the renderer or the row.

## const VERIFIER_NUON

## const VERIFIER_EXACT

## const VERIFIER_NU_VALUE

## const KNOWN_VERIFIERS

Every verifier is mechanical. Nothing here asks a model whether an answer is
good, which is what keeps judge output out of the training path entirely.

## fn verifier_known

## fn verifier_executes

The caller needs this because execution needs the sandbox present, and the
sandbox check is once-per-pass rather than per item.

## fn verify

Two of the three compare values rather than text, and that is the load-bearing
choice. NUON renders the same value differently depending on its content, and two
correct pipelines can render one result as a bordered table and as a literal - so
a text comparison would score formatting and call it correctness.

`exact` is the exception by design, since it grades the formatting families where
the bytes are the answer. It is also the criterion that a later reading found was
aimed at the wrong object for the broadening stages: it grades the rendering,
which is the half that legitimately varies, and the value comparison grades what
the model actually produced.
