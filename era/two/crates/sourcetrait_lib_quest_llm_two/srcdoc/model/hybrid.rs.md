# hybrid.rs

The model driver over the 32 layers: the stateless whole-sequence parity
instrument, the carried chunk path that prefill and decode share, the eviction
compaction epochs, and the staged decode path graphs capture.

The recurrence and its gates ride f32 regardless of model dtype - the pinned
upstream discipline - and attention softmax computes in f32 and casts back.

## enum Layer

## struct SpecMark

Attention rows are prefix-correct, so acceptance is a LENGTH RESET recorded
here. The GDN caches are cumulative and cannot be reset that way, so the mark
shadows them per layer and arms a rule-input capture instead. That asymmetry is
the whole design of the accept path.

## struct ContextMark

## const RESERVE_DECODE_MARGIN

The margin exists because a graph arm would otherwise pre-reserve against the
full sample budget: the 32768 default would pre-grow gigabytes a typical decode
never touches.

## struct OlmoHybrid

## impl OlmoHybrid

### fn new

The shared prefill scratch pool is installed AFTER construction because layers
run sequentially, so one set serves all 24 and there is nothing to size until
the layers exist.

### fn settings

### fn forward_all

Stateless, and it never touches the carried caches. It is the parity instrument
the oracle comparisons run against, which is why it must stay free of cache
machinery even though the carried path could produce the same numbers.

### fn advance_chunk

### fn needs_prefill_mask

The flash dispatch takes causality as a KERNEL PARAMETER and never reads the
mask tensor, so an all-flash forward can skip the build entirely - roughly 2 GB
of host fills, uploads and casts across a 32K prefill.

Every ineligible grade keeps it, and a profile-armed pass forces it back on,
because the profiled path is eager. The eager path then hard-errors on a
missing multi-token mask rather than attending with full visibility, which is
what makes a leak in this predicate loud rather than silent.

### fn forward_chunk

### fn forward_chunk_last

Per-row math: RMSNorm and the head are row-independent, so the last row here is
value-identical at f32 to `forward_chunk`'s last row. The bf16 kernel SHAPES
differ - a one-row gemm against a t-row one - so kin-class deltas ride the
gates rather than being a defect.

### fn forward_chunk_carry

Where the skipped head traffic actually lives: the caches advance and no logits
are computed at all.

### fn contain_parked_grows

One parked buffer set per model lifetime stays the bound, which is why this
retires capture rather than merely flushing the cache.

### fn clear_cache

### fn reserve_for_generation

ONE allocation at exactly prompt plus margin, instead of about one regrow per
reserve step of prefill. Each regrow holds old and new buffers during its copy
and frees an odd-sized block into the raw allocator - that fragmentation is
what inflated the 32K peak, and it is why the grain rounding was dropped here:
with no regrows to amortize it was pure padding, roughly 170 to 235 MiB at 32K.

### fn context_len

### fn snapshot_caches

### fn restore_caches

A restore is a CLEAR-CLASS epoch, so an armed graph stage disarms and the next
arm revalidates. Treating it as anything less would leave graphs captured
against a cache state that no longer exists.

### fn mark_context

The public mark exists for shared-prefix scoring loops - the capability
runner's multiple-choice continuations - and carries the same classic-path-only
constraints as speculation.

### fn rollback_context

### fn spec_mark

### fn spec_rollback

### fn spec_accept

THE ACCEPTED-PREFIX ROWS OF A VERIFY CHUNK COMPUTE IDENTICALLY TO A PLAIN
ADVANCE, under causal attention and the recurrence order. So the attention KV
rows written during the verify ARE the accepted rows, and acceptance costs a
length reset with zero recompute, while only the cumulative GDN caches restore
and re-advance.

### fn spec_release

### fn evict_keep_sets

### fn evict_post_prefill

Ranking uses the FINAL scoring pass, which by construction is the question
window. The capacity chosen here is exact rather than grain-rounded, because
overflow epochs bound decode length at cap plus slack and a graph arm may
pre-grow it once more.

### fn evict_overflow

Packs IN PLACE so every baked address stays alive, then re-points an armed
stage's mask at the shrunk length. The bucket is unchanged by construction, so
the same captured graph replays across the epoch.

### fn arm_attn_profile

### fn take_attn_profile

### fn device

### fn graph_kv_state

Uniform by construction - every layer sees every token - and asserted rather
than assumed, because the staged path depends on one slot and one mask serving
all eight layers.

### fn arm_graph_decode

Pre-grows once to the run's bucket ceiling so later bucket crossings never
reallocate under a captured graph. The ceiling clamps the sample budget's
contribution to the reserve margin; a decode that outruns it falls to the
classic path at the capacity epoch rather than failing.

### fn graph_armed

### fn set_graph_capture

### fn graph_disarm_when_full

The BUCKET the next step needs is the binding width rather than the raw length,
because it rounds up by the grain and can therefore overshoot a capacity the
grain does not divide.

### fn flush_captured_graphs

### fn graph_decode_step

The stage is taken out and ALWAYS put back, even on error. An error must not
destroy the persistent staging, since the buffers behind it are what captured
graphs reference.

### fn graph_step_with

### fn capture_decode_graph

THE RECIPE IS PINNED AND ORDER-SENSITIVE. Hold candle's parameter-cache guard;
WARMUP-run the sequence uncaptured, which both performs this step's real work
and populates the content-keyed dimension and stride cache whose miss during an
active capture is a designed hard error; then record, which performs no work.

NEVER LAUNCH THE FRESHLY CAPTURED GRAPH: the warmup already computed that step,
so launching would execute it twice.

Thread-local capture mode fails loudly on a capture-illegal call from this
thread. Auto-free-on-launch is the relaunch semantic for in-graph memory nodes.
The executable is uploaded explicitly because the alternative flag is
parameters-only.

### fn graph_forward_sequence

GDN layers run their ORDINARY carried step here rather than a special one -
fixed shapes and in-place state make the classic form already capturable, which
is why only attention needed a staged variant.
