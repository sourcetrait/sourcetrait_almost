# attn_layer.rs

The full-attention decoder layer, FULLY POST-NORM in the Olmo 2 and 3 style:
raw hidden state enters attention with no input norm, and the norm sits on each
sublayer's output before the residual add. That is a real family feature rather
than naming, and it is the opposite convention to the GDN layer in the same
model - one model, two block conventions.

## const KV_RESERVE_STEP

Growth happens in whole steps and reads narrow to the live length, so the
per-append concatenation dies. That concatenation copied the entire KV cache
per token, which measured 12.6 percent of 32K prefill GPU time and, worse,
scaled with context - era one's allocator-churn lesson verbatim.

Clear keeps the capacity, which is the era-one semantic and what makes a second
generation cheap.

## struct AttnKv

## struct AttnLayer

Four of these fields exist for the capture lifecycle rather than for the math.
`buffers_captured` records that a graph baked the buffer addresses, which makes
their eventual replacement a PARK rather than a free - ever-captured buffers
refuse asynchronous free, so `mem::forget` is the correct operation and the
leak is deliberate. `parked_graphs` latches that it happened so the model can
collect it and retire capture. The score buffer rides the identical lifecycle
because captured graphs bake its address too.

`score_zero` is a persistent zero row rather than a fresh allocation per use,
for the same address-stability reason.

## impl AttnLayer

### fn new

### fn kv_len

### fn kv_capacity

### fn advance_len

### fn take_parked

### fn forward

### fn forward_chunk

### fn score_pass

Prefill chunks re-score the WHOLE store from their tail queries, so by the
final chunk the scores are the question window - that overwrite is the
stage-2 mechanism rather than an accumulation.

A decode row zeroes its OWN slot after writing. A fresh token's
self-attention score is large by construction, and the recent-window
protection already covers it, so letting that score persist would keep the
token alive long after the window moved past it.

### fn compact

`shrink_to` re-homes survivors into fresh capacity-sized buffers, which is
where the residency win actually cashes - but it is legal ONLY while nothing
captured the buffers. Otherwise the gather packs IN PLACE, keeping every baked
address alive, which is also what an overflow epoch always does.

### fn reserve

The capacity check happens BEFORE the grain rounding, and the order is
load-bearing. An exact pre-reserve is not a grain multiple, so rounding first
would demand a spurious regrow at the last grain boundary under it - which
lands mid-final-prefill-chunk, at the KV-maxed moment, exactly where a regrow
is least affordable.

### fn reserve_capacity

### fn close_halves

### fn qkv

The full-projection-width q and k RMSNorm runs BEFORE the head reshape, which
is the family's convention and not interchangeable with a per-head norm.

### fn attend

Decode stays eager even on a flash build, because candle-flash-attn 0.11 has no
split-kv kernel - the era-one measured gate rather than an oversight.

The eager path HARD-ERRORS on a missing multi-token mask rather than attending
with full visibility. That guard exists because the mask build is skippable
when flash covers every layer, and a leak in that logic would otherwise be
silent and catastrophic: attention that sees the future produces plausible
output.

### fn attend_profiled

Slicing is value-identical because softmax and the weighted sum are
row-independent. It exists to bound the f32 softmax transient so that 32K
profiling fits at all.

An armed profile forces this path regardless of flash settings, because
observation needs materialized weights and flash never produces them.

### fn attend_flash

Q PACKS AND K/V STAY VIEWS, deliberately. The flash kernels take strided rows
as long as the last dimension is contiguous, and a `contiguous()` on the
carried span would copy the WHOLE prefix per chunk per layer - gigabyte-class
transients late in a 32K prefill, and the top copy share on the prefill path.

Bottom-right alignment gives absolute-position causality over the carried
prefix, which is what makes chunked and cached prefill safe under a kernel that
knows nothing about the offset.

### fn restore_kv

Stale rows beyond the restored length stay hidden rather than being zeroed,
because every read narrows to the live length.

### fn clear_cache

### fn forward_graph

The same math as the classic mask-free decode, over a STATIC bucket width. Pad
columns carry exactly zero post-softmax mass, so the widening is free in
correctness terms and costs at most one grain of padded compute.

The in-graph score write is bucket-static and its operands are address-stable,
which is what makes it capturable at all. Pad columns write near-zero scores to
slots beyond the live length, and those are overwritten when the slots fill.
