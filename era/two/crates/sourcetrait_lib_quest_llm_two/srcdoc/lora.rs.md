# lora.rs

The adapter framework: frozen-base additive adapters, with the placement rule
enforced by construction rather than by validation.

That construction is the whole design. The only adapter types that exist cover
the sanctioned readout surface - the GDN query projection, its convolution, the
gate and output projections, attention query and output, and the MLP everywhere -
so the state-carrying weights have no adapter type to attach. The GDN key and
value projections and their convolutions, the two gating projections, and
attention key and value are not excluded by a check that could be forgotten; there
is simply nothing to attach to them.

Why those in particular: cached key-value state and prefix snapshots must stay
expert-invariant, so an adapter that moved a key or value projection would make
every saved context wrong for every posture but the one that saved it.

Zero-init keeps adapter-off bit-exact, which is the zero-regression fallback the
whole program leans on. The low-rank pairs draw `a` from a normal and leave `b` at
zero, so an untrained pair contributes exactly nothing rather than nearly nothing.

## struct LoraPair

## fn normal_draw

Box-Muller over the crate's own SplitMix64 rather than an external generator,
which is what makes an adapter's initialization reproducible from its seed alone.
The seed rides the artifact's metadata for the same reason.

The clamp on the first uniform draw guards the logarithm: a draw of exactly zero
would produce infinity.

## fn init

The `a` values are drawn on the host and uploaded rather than generated on the
device, because a device generator's stream is not reproducible across backends
and the CPU gate and the CUDA run must agree.

Both tensors are marked as requiring gradients, which is a no-op on a non-autodiff
backend and is why the same type serves the oracle and the trainer.

## fn contribution

## fn merged_delta

The consumer is the engine's own adapter loading, which merges a delta into a base
weight. The orientation transpose is left to the caller because this crate stores
the pair over the transposed weight while the checkpoint stores the weight
untransposed, and only the caller knows which side it is holding.

## struct ConvDelta

A full delta rather than a low-rank pair, and the arithmetic is the reason: the
query convolution is a four-tap depthwise kernel, so its whole weight is
`channels` by 4. A rank decomposition of that buys nothing - it is already
smaller than the pair that would approximate it.

## fn init

## fn effective_taps

Added at forward rather than merged at load, so the base taps stay shared and
unmodified. A zero delta therefore yields the base convolution bit-exact.

## enum LayerAdapters

Two variants rather than one struct with optional fields, which is what makes the
placement rule structural: a GDN layer's adapter set cannot carry an attention
projection because the type has no field for it.

## fn params

## fn params_mut

Two nearly identical walks, and the duplication is deliberate rather than
awkward. The optimizer needs mutable access while the gradient chain needs shared
access, and a single generic walk over both would need interior mutability - which
would cost more clarity than the twenty duplicated lines do.

The names are the checkpoint-aligned prefix plus the target, so a gradient, an
optimizer state and a saved tensor are all keyed identically. That is what lets
the chain hand gradients to the optimizer as a name-keyed map rather than as a
positional list.

The conv taps are keyed with a synthetic per-tap suffix, since they train as
`kernel` separate rows but persist as one tensor.

## fn pair_refs

## fn pair_muts

## struct ModelAdapters

Rank and alpha are carried on the whole model rather than per pair, because rank
is fixed library-wide: an adapter swap is meant to be a pointer swap, which it
cannot be if two adapters disagree about geometry.

## fn init

Placement is decided by the config's layer kinds, read through the same
`is_gdn_layer` the model itself uses. That shared read is the invariant behind the
`unreachable!` arm in the trainer's block dispatch.

## fn load

What makes a staged sequence possible. Without it every stage would restart from
the base and the chain would not exist.

Rank and alpha come from the artifact rather than from the caller's flags, because
a resumed adapter's geometry is already decided and taking them from a flag would
let a mismatched rank load as garbage rather than error.

The structure is built at the artifact's geometry and then every trainable tensor
is overwritten from the file, rather than being constructed from the file
directly. That reuses `init`'s placement logic, so a resumed adapter cannot
disagree with a fresh one about which projections adapt.

The conv delta needs the inverse of the save-time assembly: it persists as one
`(channels, 1, kernel)` tensor and trains as `kernel` rows, so each tap is a
stride through the stored values rather than a contiguous slice.

Validation is strict and loud throughout - version, model identity, rank against
every pair, dtype, and shape - because the failure it prevents is a wrong-model
adapter loading successfully and training nonsense.

## fn params

## fn params_mut

## fn save

One safetensors file, with the low-rank pairs stored over the transposed weight
and the conv delta reassembled into the checkpoint's own layout. Metadata carries
the version, the model identity, rank, alpha, seed, the target list and the tool
version.

The buffers are sorted by name before serialization, which makes the artifact
byte-reproducible from the same training run rather than dependent on a hash map's
iteration order.

The target list is written as a string rather than derived at load, so an artifact
records what surface it was trained over even if the sanctioned surface later
changes. A future reader can then tell an old adapter from an incompatible one.

## fn tensor_f32_values

## fn tensor_f32_bytes

Adapters persist as f32 regardless of the training backend's default float. The
values are small against the base weights, and f32 is what makes an adapter
loadable at any grade the engine runs.
