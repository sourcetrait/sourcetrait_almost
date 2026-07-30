# gdn.rs

## const O_NORM_EPS

1e-5, the fla `FusedRMSNormGated` default the pinned upstream hardcodes, and
deliberately NOT the model's `rms_norm_eps`. It is the only place in the model
where the two differ.

## fn gdn_gates

Computed in f32 regardless of model dtype, matching the upstream torch fallback
which upcasts the gates whatever the model is. The decay is handed onward in
LOG space rather than exponentiated here, because the chunked rule needs the
cumulative sum of logs and exponentiating early would lose that.

Doubling beta under the negative-eigenvalue variant is this checkpoint's
posture rather than an option we chose; the config field decides it.

## fn softplus

The guard at 20 is the same threshold the upstream fused gating kernel uses.
The FORWARD is value-safe without it, because softplus saturates and the decay
underflows to zero, so both forms agree to floating-point precision wherever
they differ - which is exactly why its absence is invisible in a forward-only
stack.

It matters where a BACKWARD exists: the naive form's exponential overflows to
infinity on real gating inputs, and differentiating through a retained infinity
computes infinity over infinity, which is NaN. That surfaced on the burn side
as a finite loss at step one and NaN at every step after.

## fn l2_norm

A SUM under the root, not RMSNorm's MEAN. The two differ by a factor of the
square root of the dimension, so reading this as an RMSNorm produces a q and k
scale that is wrong by about ten at this geometry - the single easiest
misreading in the mixer.

## fn depthwise_causal

The kernel unrolls into kernel-many shifted broadcast multiply-adds, computing
the same values in the same per-element accumulation order as a grouped
`conv1d` would.

It is here because candle 0.11's CPU grouped convolution chunks its input into
`groups` single-group convolutions and maps them SERIALLY. At the 2880 to 5760
depthwise groups this model carries, that is roughly 11.5 thousand convolutions
per GDN layer and measured a fixed 61.5 seconds per decode step with under
three of thirty-two cores busy. The unroll is milliseconds.

The replacement was sub-ulp rather than merely close, because candle fuses the
multiply and add: the stateless gate moved from 9.471e-13 to 9.487e-13, the
same pin class.

## fn causal_conv_silu

Prepending kernel-1 zeros equals the upstream's pad-both-sides-then-truncate
form, and the consequence worth holding is that the LAST weight column
multiplies the CURRENT token. Getting that orientation backwards produces a
model that reads the future, and it is locked by a unit test for that reason.

## const CHUNK

## fn conv_with_tail

A zero tail equals the stateless form BY CONSTRUCTION, which is what makes cold
starts and chunk chains agree without a separate code path to keep in sync.

The tail holds RAW pre-convolution rows in f32, which is exact for a bf16 model
and is what lets the carried and stateless forms match to the bit.

This engine therefore does not inherit the transformers multi-token
convolution gap, where an existing conv cache is ignored on a multi-token
forward: tails carry across chunk boundaries here and a unit test locks it.

## fn recurrent_step

Decay is applied BEFORE the delta correction. That order is the recurrence's
definition rather than an implementation detail, and reversing it is a change
that still runs and still produces plausible output.

## fn lower_triangle

## fn chunk_rule

The upstream `torch_chunk_gated_delta_rule` re-expressed in candle ops, holding
the same contract as `recurrent_step`: q and k already l2-normed, q already
scaled. It matches the sequential form to f32 rounding, which is the upstream's
own 1.2e-13 pin rather than a bar we set.

THE INVERSION LOOP IS THE SHAPE THAT MATTERS. Forward substitution runs IN
PLACE on one working matrix through `slice_set`. The prior vector-and-append
shape re-copied every prior row per iteration - about two thousand copy kernels
per call, which profiled at 39 percent of ALL 32K prefill GPU time and was the
single largest prefill lever in the engine. The in-place form is about five
kernels per row and every gate re-read identical digits, so it was pure shape.

Two mechanics keep that correct. Reads materialize through `contiguous()` or
`cat` before the row write, so no view observes its own mutation. And `cat` on
a non-zero dimension returns a TRANSPOSED VIEW while `slice_set` demands
contiguity - the era-one contiguity lesson, and the reason the explicit
`contiguous()` calls here are not redundant.

The pad rows carry zero keys, which makes them inert rather than merely
ignored, so they can be dropped from the output without a mask.
