# fused.rs

The fused gated-delta DECODE family: the per-token recurrence as one raw kernel
over the f32 state IN PLACE, replacing a roughly seven-kernel candle chain per
GDN layer per step, plus the norm, convolution and head-prep kernels around it.

Raw pointer arguments on the model's own stream, operands addressed at their
LAYOUT OFFSETS - the offset-leak lock in `graph.rs` explains why that is not
optional - which is what makes every kernel here capture-legal.

NUMERICS: the same per-element math as the classic chain with named
reassociations, so cuda readings move within the bf16 envelope class while the
cpu path never dispatches here and stays bit-stable by construction. The
reassociations are deliberate and locked, not incidental: the decay fold in the
recurrence, serial in-thread reductions where candle sums tree-wise, one
rounding at the store where the classic chain rounds before the weight
multiply, f32 accumulation where the classic convolution rounds its tail
through bf16 per operation, and a reciprocal-multiply for the l2 divide.

Each of those moves the result TOWARD f32 truth rather than away, which is why
the gate readings tightened monotonically across the kernel eras.

## const KERNEL_SRC

The CUDA source carries its own commentary inline, which is left in place: it
documents a program in another language, and the srcdoc headings here address
Rust items rather than positions inside a string literal.

## static MODULE

## fn kernel

## const NORM_BLOCK

A power of two because the tree reduction requires it; shared memory is one f32
per thread.

## struct GdnFusedStep

One block per head, one thread per value column. The third operand is an
OUTPUT despite the in-place-op signature: candle tensors mutate through shared
storage by design, the same mechanism `slice_set` uses, so writing y through it
is ordinary rather than a trick.

## struct RmsNormFused

## struct ConvStepFused

The tail is READ AND ROTATED IN PLACE inside the launch, so the second operand
is mutated. That is safe for the same shared-storage reason, and the tail
buffer's address stability is already the graph-capture contract.

## struct RmsNormGatedFused

Consumes the fused step's RAW f32 output. The classic path rounds y through
bf16 before the variance is taken, so this form has one fewer rounding on a
value the variance is computed from.

## struct HeadPrepFused

Collapses the entire single-token head-prep chain into one launch per GDN layer
per step: three upcasts, about ten l2 launches, about thirteen gate launches
and the packing concatenation all disappear.

GEOMETRY-LOCKED to a key dimension of 96 and a value dimension of 192. That is
a deliberate trade - the kernel indexes with compile-time constants - and the
guard rejects anything else rather than computing quietly with wrong strides.
