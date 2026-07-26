# norms.rs

The order of casts here is SEMANTICS, not style, and it is pinned from the
upstream modeling code rather than chosen. Variance and normalize run in f32,
the weight multiplies AFTER the downcast to the input dtype, and the gated form
applies `silu(gate)` in f32 after the weight. Reordering any of those changes
digits the parity gates arbitrate.

## fn rms_norm

## fn rms_norm_auto

The CONTIGUITY GATE is the load-bearing part of the dispatch, and it is a
correctness edge rather than an optimization threshold. The attention query and
key norms feed multi-row NARROWS of the fused projection, which are contiguous
only at one row, so those legs stay on the classic chain rather than paying a
pack copy to reach the kernel.

Stateless and parity callers pass `fused = false` by construction, which is why
arming fusion cannot move their digits at all.

## fn rms_norm_gated

The epsilon here is 1e-5, the fla `FusedRMSNormGated` default the pinned
upstream hardcodes, and NOT the model's own `rms_norm_eps`. It is the single
eps exception in the whole model, so a reader who assumes one epsilon
everywhere is wrong exactly here.
