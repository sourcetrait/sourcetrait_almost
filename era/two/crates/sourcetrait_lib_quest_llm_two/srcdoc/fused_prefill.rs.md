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

The embedded kernels keep their one-line summaries at their own items; their
rationale is here, one sub-heading per kernel.

The geometry the whole family compiles against: one tile-row chunk per block,
where the tile is prepended per compile, and a substitution block count derived
from it at sixteen rows per block. A bundle row is a fixed number of f32 cells
- query, key, w and u as bf16 pairs, then exact-f32 cumulative gate and beta.
Beta rides the bundle for the solve and the k-cumdecay prep; the state-advance
pair ignores it. The family is geometry-locked to a key dimension of 96 and a
value dimension of 192.

### pack_pair and unpack_pair

The bf16 pair packing: the query, key, w and u columns ride two bf16 values per
f32 cell, so one four-byte load carries two elements and the rule family's
bandwidth-bound bundle traffic halves. The gate and beta cells stay EXACT f32,
because those compound through the recurrent state - the same accumulation-order
lesson that keeps the a and b projections unfused. The even column is the low
half.

### tri_solve_f32

The intra-chunk solve: build the strictly-lower matrix as the negated
beta-scaled key product times the decay, run the blocked forward substitution,
then add the identity. One block per head-and-chunk tile, one thread per row,
with key, beta and the cumulative gates read from the bundle rows.

THE BLOCKED SUBSTITUTION is the part worth understanding. The sixteen-wide
diagonal blocks solve CONCURRENTLY, since they are independent regions each
following the row-serial snapshot shape; then each block row's off-diagonal
tiles update by shared-memory block products - a sum over the intervening
blocks, with the in-block inverse applied LAST. That is the same bilinear form
as the row-serial loop with a regrouped accumulation, so it is
reassociation-class and the lock re-pins it.

Two details make it cheap. The freed key staging region stages the intermediate
tiles, being dead once the initial matrix is built. And in-block upper triangles
stay zero, so the full sixteen-wide sums add exact zeros rather than needing a
bound check.

### prep_chunk_f32

The per-chunk prep chain in one launch: the carried causal convolution and silu
in f32 where the classic chain rounds through bf16 per operation, the query and
key l2 norms with the query scale, the gating scalars, and the per-chunk decay
cumulative sum - written bundle-direct, plus the beta-scaled value operand for
the gemm.

The cumulative sum runs SERIALLY on thread zero, deliberately: that reproduces
candle's own association for the same quantity, and a parallel scan would not.

The dynamic rows arrive in one tensor with a fixed layout - the convolution
weight taps first, with the gate parameters riding the first tap row's pad
columns, then the carried tail, then the token rows.

PAD ROWS PAST THE TOKEN COUNT ride the scan as zeros, which keeps the last real
cumulative sum intact at the tile's end, and write only their gate cell.

The query and key passes RECOMPUTE the convolution rather than holding it -
two passes over cheap arithmetic - because a 96-wide register array would
spill, and spilling costs more than the recompute.

### prep_kbg_f32

The k-cumdecay gemm operand, read straight off the bundle. The multiply order
is the classic one: key times beta, then the exponential.

### state_stage_f32

Pass one of the advance: the serial inter-chunk recurrence ALONE. Per chunk it
writes that chunk's INITIAL state and its vnew rows to the scratch, then
advances the carried state. No attention tile and no output rows - pass two
computes those chunk-parallel.

The grid is heads by four stripes with one thread per value column of its
stripe, which is the landed register contract rather than an arbitrary shape.
The vnew and update math is the earlier single-pass form verbatim, same
per-element accumulation orders, so the split moved no digits.

The state update is the decayed carry plus the decay-scaled key transposed
against vnew, in the landed row-major order.

### state_out_f32

Pass two: the chunk-PARALLEL output. Every tile reads its chunk's initial state
and vnew rows from the scratch, builds the decayed local attention against its
own bundle rows - the query-key product times the gate difference, on and below
the diagonal and zero above - and writes its output rows independently.

The output row is the gate-scaled inter-chunk term plus the local attention
against vnew, unrolled full-width because the upper triangle adds exact zeros.

### pack_pairs_bf16

Packs a gemm's f32 output rows into the bundle's pair cells at a fixed offset -
the w and u columns, and the lock harness's query and key. One thread per pair,
rounding round-to-nearest-even like every bf16 store in the family.

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
