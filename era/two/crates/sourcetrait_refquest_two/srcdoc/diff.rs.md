# diff.rs

The dump comparison: normalised mean-squared error plus per-row argmax
agreement, over two all-position logits dumps.

It is valid only on same-ids dumps, and the id tensors are asserted byte-equal
before any row arithmetic. That is the era-one lesson made structural: diff a
same-ids replay, never two independently generated runs, because two runs that
diverge at one token then disagree about everything after it and the number that
comes back measures the divergence rather than the implementations.

This is the crate's other genuinely local computation, and the only verb that
spawns nothing.

## fn diff

Two accumulators rather than one, and the distinction is what makes the reading
comparable across lengths. The whole-file error sums the numerator and
denominator over every row before dividing, so a long dump is not dominated by
its noisiest row, while the per-row values exist only to rank the worst ones for
reporting.

The reference is the denominator, which makes the measure asymmetric on purpose:
this answers how far the candidate sits from the reference rather than how far
apart they are.

Argmax agreement is counted alongside, and the first divergent row is reported,
because at short lengths agreement is tie-dominated and carries no statistical
power - so a reader needs to see where the flips are rather than only how many.

The payload is one JSON line on stdout in the same event shape the python drivers
emit, so a diff row files beside a generation row in one record.

## struct DumpView

## fn row

Decodes a row on demand rather than converting the whole tensor, which keeps a
gigabyte-class dump at its file size in memory rather than at a multiple of it.
Reading the buffer as little-endian explicitly is what makes that safe against a
foreign-endian host, which safetensors pins.

## fn load_dump

The re-slice at the end is the one subtle thing in this file. A safetensors
`TensorView` borrows from the buffer passed to `deserialize`, and holding it
would tie the view's lifetime to the parser rather than to the file bytes - so
the offset is recovered by pointer arithmetic against the buffer's own base and
the data is re-sliced from the bytes directly. That yields a plain borrow of the
file with no parser in the way, which is what lets the view outlive the parse.

The dtype and rank are asserted rather than assumed, since a dump written by
something other than the contract would otherwise be read as though it complied.

## fn id_bytes

Re-parses the header rather than threading the parser through, which is cheap
against the row arithmetic and keeps the same-ids assertion readable as its own
step.
