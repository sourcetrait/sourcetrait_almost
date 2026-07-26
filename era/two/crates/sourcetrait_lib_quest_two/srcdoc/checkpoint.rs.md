# checkpoint.rs

## struct OlmoHybridConfig

The transformers 5.x flat form for `model_type` `olmo_hybrid`: explicit
`layer_types`, the `linear_*` GDN fields, and a `rope_parameters` block whose
null theta is the NoPE gate.

## fn num_kv_heads

## fn head_dim

## fn key_dim

## fn value_dim

## fn is_nope

Mirrors the pinned upstream gate EXACTLY: rope exists only when
`rope_parameters.rope_theta` is non-null. Both era-two checkpoints ship null,
so no rotary object is ever constructed, attention receives no position
embeddings, and `position_ids` reach only the mask machinery.

The DPO checkpoint's `rope_type` of "default" is dead metadata that never
reaches code, which is worth knowing before anyone reads it as evidence that
rope is active.

## fn layer_kind

## fn validate

The sole-checkpoint policy, enforced at load rather than trusted. Each check
buys a specific absence downstream: equal query and key-value head counts mean
no repeat-interleave machinery exists anywhere in the engine, and equal GDN key
and value head counts mean the same for the mixer. Both layer kinds being
present is what makes "hybrid" true of the file rather than of the name.

Failing here is the intended outcome for a foreign checkpoint. The alternative
is a model that constructs and then computes something subtly wrong.

## enum LayerKind

## struct RopeParameters

Kept whole even though both era-two checkpoints null the theta, because the
NULL IS THE SIGNAL that `is_nope` reads. Collapsing the block to a bool would
throw away the ability to see a future non-null theta arrive.

## fn data_home

## fn model_dir

Models resolve under the XDG DATA home rather than the cache home: a checkpoint
is not regenerable, and the two homes encode exactly that difference. Snapshots
take the opposite call.

## fn load_config

Validation rides the load rather than sitting beside it, so there is no way to
hold an unvalidated config.
