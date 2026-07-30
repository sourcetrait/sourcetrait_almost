# mask.rs

## fn causal_mask

## fn offset_causal_mask

The mask is ADDITIVE with negative infinity above the diagonal rather than a
multiplicative keep-mask, which is what lets the softmax saturate a blocked
column to exactly zero mass instead of a small non-zero one.

Both forms are the same function - the plain mask is the offset form at zero -
and only the carried prefill path needs the offset, because a decode row
attends everything and runs mask-free.

The build is host-side, which is what makes it worth SKIPPING rather than
optimizing: the flash dispatch takes causality as a kernel parameter and never
reads this tensor, so an all-flash forward can avoid the fill, the upload and
the cast entirely. See `hybrid.rs`'s `needs_prefill_mask`.
