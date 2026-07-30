# oracle.rs

The stateless Olmo-Hybrid forward on burn: the oracle triangle's third stack,
beside the original python one and this crate's own candle engine.

It is implemented from the pinned checkpoint semantics rather than translated
from the candle implementation, and that is the entire reason it exists. Era one's
third stack turned out to be bit-identical to its second because both rode one
candle graph, so a shared misreading would have passed every comparison. An
independent reading is the only thing that closes that.

There are no caches and no decode path. Every call is one full-sequence replay
over all positions, which makes it useless for generation and ideal as a
reference: there is no cache, mask or offset state for a bug to hide in.

The f32 discipline is not a grade-specific choice. The gate, norm and recurrence
region computes in f32 on every backend through per-tensor casts, which is a no-op
on the CPU reference and a real cast on the bf16 CUDA grade. That matches the
reference stacks, which upcast the same region regardless of model dtype - and the
1024-deep training backward made it mandatory rather than merely faithful, because
a bf16 recurrence NaNs there.

Adapters ride the same block forwards as an `Option`, with the absent path
operation-identical rather than merely numerically close. That is what lets the
parity gates and the trainer share one model, and it is re-verified after any
change to injection: the CPU f32 gates re-read their exact standing digits.

## type FloatTensor

## struct HybridModel

The embedding table is host-resident rather than a device tensor, since the
forward gathers rows by id for one sequence at a time.

## enum HybridBlock

## struct GdnBlock

## struct AttnBlock

Two block structs rather than one with optional fields, because the two layer
kinds genuinely differ in their weight sets and their norm conventions. The
hybrid is one model with two block conventions, and the types say so.

## const O_NORM_EPS

The gated output norm's epsilon is the fla `FusedRMSNormGated` default and is
deliberately not the model's own `rms_norm_eps`. Using `rms_norm_eps` here is the
plausible mistake, and it moves every GDN layer's output slightly.

## const L2_EPS

## fn new

The embedding shape is checked against the config rather than trusted, since a
mismatch there is the signature of a wrong checkpoint and would otherwise surface
as a confusing failure much later.

The tied-embedding branch exists for loader generality. These checkpoints ship
untied, so it is the untied path that runs.

## fn embed

## fn forward_all

Returns a flat row-major host buffer rather than a tensor, because the consumer is
a comparison against a reference dump. Row `r` predicts token `r+1`, which is the
dump contract.

The causal mask is built once for the whole sequence and shared across the eight
attention layers, rather than per layer.

## fn new

## fn forward

The GDN block is fully pre-norm on both halves: the mixer sees a normed input and
the residual is the raw one, then the same shape again for the MLP. The attention
block is the opposite. Getting either wrong is a real semantic error rather than a
style difference, and it is the first thing to check if a parity gate moves.

The dataflow, in the order it must happen: the three projections each through
their own causal depthwise convolution with silu, then the f32 upcast, then the L2
norms on query and key, then the query scaling, then the gating scalars, then the
recurrence, then the gated output norm, then the cast back for the output
projection. The gate projection takes no convolution, which is easy to miss beside
the three that do.

Decay is applied to the state before the delta correction, and that order is the
recurrence rather than an implementation detail. Reversing it produces a plausible
recurrence that is not this one.

The gating scalars are `-exp(A_log) * softplus(a + dt_bias)` for decay and
`2 * sigmoid(b)` for beta, the latter being the negative-eigenvalue variant.

The gated output norm normalizes before gating, with the gate being
`silu(g_proj(x))`, and all of it in f32 with one cast back before the output
projection.

Adapters add on the sanctioned surface only - query, its convolution, the gate and
output projections, and the MLP - so the key, value and gating path never adapts.
The `Option` is matched rather than folded into a default, so the absent path
compiles to the same operations the frozen oracle runs.

The recurrence loop narrows and reshapes per token, which is what makes it a
correctness grade rather than a fast one. The chunked form is the engine's job.

## fn new

## fn forward

Raw hidden state enters attention with no input norm, and the norms sit on the
sublayer outputs. This is the Olmo 2 and 3 post-norm arrangement, and the export
names say so.

There is no rotation code anywhere, by construction rather than by configuration.
The checkpoints ship a null rope theta, so positions reach the computation only
through the causal mask.

The query and key norms are full-projection-width and run before the head reshape,
which is the family's convention and not interchangeable with a per-head norm.

## fn causal_mask

An additive mask of zeros and negative infinity rather than a boolean, because the
scores add it before the softmax. Built on the host and uploaded once.

## fn rms_norm

## fn rms_norm_last

Two ranks rather than one generic function, because burn's dimension parameter is
a const generic and the mean reduction names its axis. The rank-3 form is the
gated output norm's per-head shape.

## fn l2_norm_last

Sum-based rather than RMSNorm's mean-based form, which is a real difference of a
factor of the square root of the width and not a normalization variant. This is
the one upstream calls an L2 norm, and it is unit-locked separately for that
reason.

## fn softplus

The overflow-stable form, and the reason is the backward. The naive form's
exponential overflows to infinity on large gating inputs; the forward survives
that because softplus saturates and the decay underflows to zero, but the backward
computes infinity divided by infinity and yields NaN.

That was live-fired: the first real-input trainer smoke read a finite loss at step
one and NaN at every step after. The reference gating kernels carry exactly this
guard, and the candle engine's unguarded f32 forward is value-safe only because it
never runs a backward.

## fn causal_conv_silu

Four shifted broadcast multiply-adds rather than a grouped convolution call. The
engine's own history is the reason it is written this way there; here it is simply
what burn expresses cleanly for a depthwise kernel over time.

Tap `k-1` multiplies the current token and earlier taps reach back, with missing
history reading as zero. That is the stateless left-pad form, and it is the same
orientation the tap loader pins.
