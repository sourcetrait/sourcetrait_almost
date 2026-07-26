# dump.rs

The dump verb: all-position f32 logits, in the contract the comparison side
depends on.

What the artifact carries is the thing to hold: `logits` as f32 at rows by vocab,
one row per fed position of the prompt ids concatenated with the fed ids, where
row `r` holds the logits predicting token `r+1`. The id tensors are u32. That
shape is why a diff can be exact rather than approximate.

## fn dump

Two modes, and they exercise different code in the reference stack rather than
different settings. Single drives the chunked GDN prefill path; incremental
drives the recurrent decode path. Comparing the two against each other is how the
reference's own internal agreement is established, which is the floor every
cross-stack bar is measured against.

Replaying another dump's ids verbatim is what makes a comparison valid at all,
since differences are only ever computed over identical ids. That is the reason
the ids-from argument exists beside a prompt file rather than as an alternative
spelling of one.

The parent directory is created for the output because a dump lands in offload
rather than in the tree, and those paths are per-campaign rather than
pre-existing.

The CPU grade forces the fla blocker and hides the GPU for the same reason
generation does. Note the asymmetry against generation's defaults: dump defaults
to eager attention where the batteries default to sdpa, because eager is the
numerically honest path and a long dump is where it runs out of memory - so the
default is the correct one and the operator overrides it at length.
