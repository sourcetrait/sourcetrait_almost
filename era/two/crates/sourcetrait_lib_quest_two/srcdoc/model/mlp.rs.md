# mlp.rs

## struct Mlp

The gate and up projections ride ONE row-fused matrix while down stays
separate, and the asymmetry is structural rather than an unfinished fusion:
gate and up consume the same input `x`, so they share a gemm, while down
consumes their product and cannot.

The fusion happens at load by concatenating the two checkpoint tensors, so the
shard's own tensor names are untouched and the inventory locks stay green.

## fn new

## fn forward
