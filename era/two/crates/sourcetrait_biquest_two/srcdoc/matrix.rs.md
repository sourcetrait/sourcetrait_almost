# matrix.rs

The ImagineQuestMatrix builder: the foundation is authored, and this
is the authoring - every row's neighborhood has a citable reason,
which here is literally the seeding rule that produced it.

## the geometry

MATRIX_HEADS is 32 by TheUser's ruling (hidden dimension is
associative capacity, heads are correlation channels; the hybrid's 30
is ai2's comparison artifact). Width is arithmetic and settled at the
organism default head_dim 32, hidden 1024. The vocabulary is the
WHOLE dictionary (TheUser: census admission can never work off a
corpus as small as ours) - the word file's line order is the
dictionary layer's id order, ~1.24M rows total with the reserve, so
the embedding is ~1.27B parameters and the model is a lookup
structure by weight to a degree the admitted-subset design never
reached. The full 4096 geometry stays one `--head-dim 128` away. The artifact
carries the embedding matrix alone (tied embeddings are the rule;
the trainer materializes the tied head), so nothing here decides
depth, GDN dims, or intermediate width.

## the seeding rules

- Keyword and reserve rows: plain gaussians at ROW_SIGMA (0.02, the
  transformer convention), seeded per row through disjoint kind tags.
- Character rows: plain gaussians seeded by CODE POINT, not table
  index, so rows shared across a Unicode version bump keep their
  values.
- Keyboard rows: composed from exactly their declared associations -
  the symbol's character row, the count's digit row, and the
  repetition AbstractConceptMarker row (0xFF).
- Word rows: composed from their component character rows.
- Composition is sum over sqrt(k), the variance-preserving mean:
  a plain mean would shrink a k-component row's scale by sqrt(k),
  encoding word length as magnitude - noise, not signal.
- Composed rows add a JITTER_SIGMA (0.004, a fifth of the base
  scale) identity offset, seeded by the word's own bytes (FNV-1a)
  rather than its admitted position: anagrams ("dog"/"god") and
  equal-multiset spellings compose identically and must start
  distinct, and keying by word keeps a shared word's row identical
  across differently-ordered admitted lists.

## fn matrix_build

bf16 on disk (the era is bf16 end to end; the oracle loader converts
to f32 on read). The tensor is `model.embed_tokens.weight` - the name
the model tree reads - with the whole layout in the safetensors
metadata and a provenance sidecar beside the artifact. The reserve
rows exist for the expansion loop: a mint overwrites a seeded row in
place, no repack.
