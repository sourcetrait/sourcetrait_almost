# adapter.rs

Engine-side adapter consumption. A trained adapter merges into the RESIDENT
weights at load, so the decode path changes not at all and adapter-off is a
plain load that is bit-exact by omission rather than by a zero delta.

THE MERGE RIDES THE VARBUILDER SEAM, and that placement is what makes the rest
free. A wrapping backend patches each unfused checkpoint tensor as the model's
constructors `get()` it, and because those constructors fetch the unfused
tensors BEFORE their fusion concatenations, the fused rows downstream are
merged rows by construction. No span arithmetic, no constructor changes, and
nothing below here knows an adapter exists.

## const ADAPTER_VERSION

## fn adapter_path

The three-tier token resolution, shared verbatim with snapshot tokens: a pure
snake resolves under the adapters directory, a bare relative path resolves
relative to it, and anything else is an ordinary path with expansion. Sharing
the tiers rather than reimplementing them is why a user learns the rule once.

## enum AdapterDelta

Deltas stay LOW-RANK in memory and materialize per fetch. A dense precompute
across the whole placement surface would be about 23 GB, which is more than the
weights themselves.

The convolution delta is dense because decomposing a four-tap kernel buys
nothing.

The channel-rows delta merges by slicing the seven rows at the real ids rather
than materializing anything vocabulary-wide; the head is the one target whose
full delta would be the size of the weight itself.

### fn materialize

### fn merge

Rows splice by narrow-and-cat, mirroring the trainer's apply_head_delta:
the seven merged rows go through f32 and one cast back, and the other
hundred thousand ride as views into one cat in the base dtype. The first
cut cast the whole head to f32 and chained two slice_assigns over it;
candle's slice_assign pads its source and a u8 mask to the full tensor
shape and runs where_cond, so beside ~13.6 GiB of resident weights the
vocabulary-wide f32 transients exceeded the card's remaining ~10 GiB and
every r2_sft_head daemon load died with CUDA_ERROR_OUT_OF_MEMORY - while
plain-LoRA adapters loaded fine, which is what localized it. The tail
guard covers a checkpoint whose vocabulary ends exactly at the run
block; the two leading segments are non-empty for any geometry
validate_geometry admits.

## fn target_weight_name

The placement check, and it is a REJECT-BY-NAME surface rather than a filter.
Only the readout surface may carry a delta - the GDN query, query convolution,
gate and output projections, attention query and output, all three MLP
projections, and the channel-rows head delta. A state-carrying target has no
legal name here at all, which is what keeps cached KV and prefix snapshots
expert-invariant; the head rows pass that test because the unembedding is
post-cache.

## fn kind_str

## struct AdapterFile

### fn adapter_tensor

### fn load

Validation is strict and loud at load rather than lazy at first use: format
version, model identity, placement, pair completeness, and rank consistency
against every pair. An adapter trained for another checkpoint is the case this
exists to catch, and catching it here means the failure names the adapter
rather than surfacing as wrong logits.

### fn validate_geometry

The wrong-model guard proper: every delta target must exist in the checkpoint
header at the delta's implied shape. Metadata can be edited; geometry cannot.

## struct DeltaBackend

### fn apply

The merge goes base to f32, plus delta, then ONE cast to the requested dtype.
Adding in the model dtype would round twice. Rows come back already in the
base dtype - the same one-cast discipline applied per changed row - so the
trailing cast is a no-op there.

## impl SimpleBackend for DeltaBackend

### fn get

### fn get_unchecked

### fn contains_tensor

## fn load_weights

An absent adapter token returns the plain mmap path itself rather than a
zero-delta backend, which is what makes adapter-off byte-identical to a build
with no adapter code at all.
