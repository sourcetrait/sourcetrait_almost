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

The embedded kernels keep their one-line summaries at their own items; their
rationale is here, one sub-heading per kernel.

### gdn_fused_step_f32

One block per head, one thread per value column. Pass one reduces the key-value
memory for that column; pass two updates the state column in place and reduces
the y readout in the same sweep, which is why the state write and the readout
share a loop rather than being two passes.

The packed operand's rows are query, key, value, decay and beta - two key
dimensions, one value dimension, and two scalars per head.

### rms_norm_fused_bf16

One block per row: a strided f32 sum-of-squares reduction over the columns,
then normalize and weight, all in f32 with ONE rounding to bf16 at the store.
The classic chain rounds the normalized value to bf16 BEFORE the weight
multiply, so this is a one-rounding reassociation and the envelope gates
arbitrate it.

### conv_step_fused_bf16

The four-tap causal convolution decode step, its silu, and the tail shift, one
thread per channel. History is the f32 tail's three rows plus the packed
current row.

Taps and accumulation run in f32 where the classic chain rounds the tail
through bf16 before multiplying - envelope-class, and toward truth. The tail
rotates IN PLACE inside the launch, which is what removes the separate write.

### rms_norm_gated_fused_bf16

The gated form over per-head rows. The readout and the gate ride ONE packed
tensor, the gate pre-upcast, so the kernel takes a single operand where the
classic chain takes two tensors and a cast between them. Norm over the value
dimension, then silu on the gate in f32, then one rounding out.

### head_prep_f32

The single-token head-prep chain in one launch: the convolution tap's output
row splits per head, query and key take their f32 l2 norms with the query
scale folded in, the value upcasts, and the gating scalars compute from the
dynamic row - writing the operand the recurrence step consumes DIRECTLY. The
classic chain's casts, norm chains, gate soup and packing concatenation all
collapse into it.

One block per head with a power-of-two thread count for the two sum-of-squares
reductions, zero-padded lanes. The math is f32 end to end from bf16 inputs -
the classic chain's own formulas, reassociated by a reciprocal-multiply for the
l2 divide, which the component lock pins.

The softplus guard at 20 appears here in its kernel form; `gdn.rs` carries why
it matters.

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
