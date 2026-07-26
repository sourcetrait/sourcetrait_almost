# evict.rs

Stage-2-only KV eviction: the scores are the LAST scoring pass alone rather
than an accumulation. Each prefill chunk overwrites them from its tail queries,
so by the final chunk they are the question window; each decode row overwrites
them again, so they become response-conditioned. That overwrite discipline IS
the mechanism, and an accumulating variant would be a different one.

NoPE is what makes eviction position-free here. Nothing but the causal
structure of PREFILL depends on store order, and compaction only ever runs
post-prefill or between decode steps, so a compacted store is semantically
identical to an uncompacted one at decode.

## const SCORE_TAIL

16 is the ADOPTED value, not the inherited one. The era-one carry was 64; a
sweep to 32 and then 16 lifted 32K prefill from 2,474 to 2,782 tokens per
second while the full dual needle grid stayed per-cell identical to the ratchet
reference, so the question window carries its ranking signal at a quarter of
the standing tail on this checkpoint's dense middle. That shrank the re-score
tax from about 16 percent to about 6.

The live value rides the settings field, so reverting is configuration rather
than a rebuild.

## const SCORE_SLICE

The peak-transient lever. A full-width f32 softmax chain beside a prefill-sized
KV store breached era one's peak contract, which is why the scoring pass slices
at all.

## const OVERFLOW_SLACK

## fn keep_indices

Ties break toward the OLDER row so the set is deterministic across runs, which
matters because the ratchet compares per-cell outcomes and a nondeterministic
keep-set would show as retrieval noise.

Indices come back ascending. Order preservation matters less here than it did
in era one - with no positional encoding the store order is semantically free
at decode - but prefill causality still depends on position, so keeping the
compaction order-preserving costs nothing and removes a question.

## fn gather_rows

Returns fresh storage rather than a view, which is what makes it safe to
`slice_set` the result back into the source buffer.

## fn last_pass_scores

The matmul rides the MODEL dtype over a transposed VIEW - the eager-attention
shape - so there is no store-sized copy and no upcast; only the sliced softmax
runs f32. That is the difference between a scoring pass that costs a few
percent of prefill and one that doubles its peak.

The causal tail mask is built ARITHMETICALLY on device from a row-position
vector rather than uploaded, so the host carries only per-row thresholds. A
blocked column rides a large negative additive term, which underflows to
exactly zero mass through the softmax.

Row j of the tail sits at absolute position `len - tail + j`, which is the
relation the mask encodes and the one thing to get right when changing the
window.
