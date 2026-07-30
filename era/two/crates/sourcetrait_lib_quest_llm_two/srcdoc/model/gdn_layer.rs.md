# gdn_layer.rs

The linear-attention decoder layer, FULLY PRE-NORM - the opposite convention to
the attention layer in the same model. It carries the cross-chunk cache: the
f32 recurrent state plus the three raw pre-convolution tails.

THE CARRIED CACHES UPDATE IN PLACE VIA `slice_set` AND NEVER REBIND. That is
not a style preference: their device addresses must stay stable for the
lifetime of the layer, because captured decode graphs bake them, and a rebind
anywhere - this path, a clear, a restore - would leave every cached graph
reading dead memory.

## struct GdnLayer

`a_proj` and `b_proj` deliberately stay OUT of the fused projection. Folding
them in changes the gating scalars' accumulation order, and that drift
compounds through the recurrent state - vllm measured this on their own stack
and declined the same fusion for the same reason.

The fused `qkvg` row order is chosen so the first three spans are exactly the
batched convolution's channel order, which dissolves what would otherwise be an
explicit concatenation into a narrow.

Three prestored static tensors exist purely as kernel operands - the row-major
convolution weights, the prefill static rows with the gate parameters riding
the pad columns, and the decode static. Each is `Some` only when its fusion is
armed, and `fused_gdn` and `fused_prefill` arm independently by design.

## impl GdnLayer

### fn new

### fn install_prefill_scratch

### fn project_fused

### fn split_conv

### fn forward

### fn forward_chunk

### fn close_halves

### fn heads_and_gates

The f32 upcast discipline lives here rather than in the callers, so every path
into the recurrence gets it. That matches the upstream torch fallback, which
upcasts regardless of model dtype.

### fn finish_mixer

The fused gated norm consumes the recurrence's RAW f32 output directly. The
classic path rounds that output through bf16 before the variance is taken, so
the fused form is one rounding closer to truth - envelope-class, and the gates
arbitrate it.

### fn mixer_stateless

The parity instrument: a zero-seeded per-token recurrence that keeps no state.
It exists so the carried path has something to be compared against that shares
no incremental logic with it.

### fn mixer_carried

Decode rides the recurrent step and larger chunks ride the chunked rule, which
is the upstream's own path split rather than an optimization of ours.

### fn mixer_carried_fused_chunk

### fn fused_decode_step

### fn carried_chunk

### fn carried_decode_step

### fn conv_carried

On the fused decode path the tail rotates IN PLACE inside the kernel, so there
is no `slice_set` at all - the address stability requirement is satisfied by
never moving the buffer rather than by writing carefully to it.

### fn cache_snapshot

Cheap clones sharing storage. Safe only because the writer serializes them
immediately; holding one across a forward would observe the mutation.

### fn restore_cache

Shape and dtype validation lives here because the LAYER owns its geometry - the
snapshot reader deliberately validates format only.

### fn clear_cache

### fn shadow_save

The shadow buffers allocate lazily and are FRESH storage rather than clones,
because `slice_set` refuses shared storage. A clone would alias the very cache
it is meant to preserve.

### fn shadow_restore

### fn spec_release

### fn spec_readvance

Pure recurrence math over CACHED post-projection operands: no projections and
no weight traffic at all. That is what makes a partial accept cost one weight
pass per round rather than two.

Capturing the layer INPUT instead of these post-projection rows would have
re-streamed the GDN projection weights on re-advance and killed the point. The
captured set is about 0.6 MB per layer, roughly 14 MB of transient at the
largest span.

A readvance span is at most about 17 rows and therefore never a full span, so
it deliberately bypasses the scratch pool.
