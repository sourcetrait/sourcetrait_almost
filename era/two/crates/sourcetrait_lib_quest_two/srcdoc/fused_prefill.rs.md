# fused_prefill.rs

The fused chunked-GDN PREFILL family. Four ideas stack here, and each has a
measured reason.

PREP COLLAPSES THE CHAIN. One launch per layer-call performs the carried
convolution and silu, the q and k l2 norms and scale, and the gating scalars
with their per-chunk decay cumulative sum - writing straight into the rule's
bundle. No transpose, pad or concatenate pipeline exists at all.

THE BUNDLE PACKS BF16 PAIRS. The q, k, w and u columns ride two bf16 values per
f32 cell, so one four-byte load carries two elements and the rule family's
bandwidth-bound bundle traffic halves. The gating scalars and the recurrence
state stay EXACT f32, because those compound through the state - the vllm
accumulation-order lesson, which is the same reason the a and b projections
never fuse.

THE STATE ADVANCE SPLITS IN TWO. One pass runs the serial state math alone,
materializing each chunk's initial state and its vnew rows to a scratch; a
second computes every chunk's output rows CHUNK-PARALLEL from that scratch.

That split also REFUTED ITS OWN HYPOTHESIS, which is worth keeping. It moved
the pair from 21.4 to 19.3 percent of GPU time for 0.8 percent of wall clock:
the eight-fold-occupancy output pass recovered only about a quarter of its own
half, and the serial pass alone still costs 527 microseconds per call. The
recurrence is bandwidth-intrinsic at low occupancy rather than
dependency-latency-bound, and chunk-parallelism structurally cannot reach the
serial half. Prefill kernel surgery is close to mined out.

THE MODULE COMPILES ONCE PER TILE. Tile 64 is the prefill default and tile 32
serves small spans, where a speculation verify or readvance span of at most
about 17 rows computes a quarter of the tile-64 pad waste. Both variants build
from the SAME source with the tile prepended, so tile-64 code generation is
byte-for-byte what it was before the variant existed.

Numerics: the prep kernel runs the convolution, silu and l2 chain in f32 where
the classic chain rounds through bf16 per operation, and the rule kernels run
the same f32 math as the classic rule with reassociation-class deltas. The cpu
path, the stateless parity form and decode never dispatch here - only the
carried cuda bf16 multi-token branch does.

## const KERNEL_SRC

The CUDA source carries its own commentary inline and keeps it, for the reason
given in `fused.rs`.

## const BUNDLE_CELLS

## const CELL_Q

## const CELL_K

## const CELL_W

## const CELL_U

## const CELL_G

## const CELL_B

## fn tile_for

## static MODULE_TILE_64

## static MODULE_TILE_32

## fn kernel

## struct TriSolve

Folds the intra-chunk decay-mask build, the beta-scaled matmul and the
triangular inversion into one launch reading the bundle.

The inversion uses BLOCKED SUBSTITUTION: the diagonal blocks solve
concurrently, then each block row's off-diagonal tiles update as shared-memory
block products. That is the same bilinear form as the row-serial loop with a
regrouped accumulation - reassociation-class, and the lock re-pins it.

In-block upper triangles stay zero, so the full sixteen-wide sums add exact
zeros rather than needing a bound.

## struct PrepChunk

Pad rows past the token count ride the decay scan as zeros - which keeps the
last real cumulative sum intact - and write only their gate cell.

The q and k passes recompute the convolution rather than holding it, because a
96-wide register array would spill. Two passes over cheap arithmetic beat one
pass over spilled registers here.

## struct PrepKbg

## struct StateStage

Pass one: the serial inter-chunk recurrence ALONE. Grid is heads by four
stripes with one thread per value column of its stripe, which is the landed
register contract rather than an arbitrary shape.

## struct StateOut

Pass two: chunk-parallel output. Every tile reads its chunk's initial state and
vnew rows from the scratch and writes its output rows independently, so no
serial dependency remains in the heavy math.

## struct PackPairs

Row-flat and therefore tile-agnostic, which is why it can borrow either
compiled module.

## fn rule_over_bundle

The pooled set is value-identical to fresh zeros BY CONSTRUCTION on a full
span: every cell of every tensor is rewritten each pass, with the solve and the
writers covering their whole outputs. That is why no re-zeroing exists
anywhere and why only full spans may reuse.

## struct PrefillScratch

Six per-layer-call tensors for ONE standing chunk shape, allocated once and
shared by all 24 GDN layers - safe because they run sequentially, the same
single-thread contract the graph cache relies on.

Ragged and small spans keep per-call allocation, so the pad-row discipline
stays the allocator's problem rather than becoming a correctness question here.

Prefill tensors are never graph-captured, so a shape change frees the replaced
set legally - unlike the KV buffers, which must park.

## struct ScratchShape

## type PooledSet

## fn allocate

## fn checkout

The clones share the pool's storage and the kernels write in place, so no
`RefCell` borrow outlives the call.

## type SharedPrefillScratch

## struct PrepInputs

## fn prep_chunk_rule

The new tail is taken from the dyn rows rather than recomputed, which makes it
classic-exact: it is the same bf16 round-trip the classic carried convolution's
history takes.

## fn rule_from_parts

The lock rig's harness. It fills the bundle exactly the way the prep kernel
does - q and k as bf16 pairs, gates and beta as exact f32 - so the locks
exercise the real back half rather than a parallel construction of it.
