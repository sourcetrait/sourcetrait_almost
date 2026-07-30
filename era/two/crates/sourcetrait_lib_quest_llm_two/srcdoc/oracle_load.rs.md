# oracle_load.rs

Independent checkpoint plumbing for the burn oracle: its own config
deserialization and its own safetensors-to-host reader, plus the dump readers the
parity gates consume.

The independence is the point and it extends to loaders rather than stopping at
the mathematics. Era one's third stack turned out to be bit-identical to its
second because both rode one graph, so a shared reading of the checkpoint would
reintroduce exactly the common-mode blindness the oracle triangle exists to
remove. A misread shape or a wrong dtype assumption is as much a shared
misreading as a wrong formula.

Everything here loads to host f32 regardless of the file's dtype, which is what
lets one loader serve both grades: the CPU reference computes in f32, and the
CUDA grade's tensors are built from the same host buffers and cast on upload.

## struct HybridCheckpointConfig

Only the fields the frozen forward needs, so an unknown field is ignored rather
than rejected. That is deliberate here and the opposite of the library's own
config-file discipline: this reads a foreign artifact whose shape upstream owns,
where the file layer we author denies unknown fields.

## fn load

Two invariants are checked at load rather than trusted. The layer-type array's
length must equal the layer count, since the model indexes it per layer, and the
GDN key and value head counts must be equal, since nothing in the mixer
implements the unequal case. Both would otherwise surface much later as a shape
mismatch deep in a forward.

## fn is_gdn_layer

An unknown layer type raises rather than defaulting, because either default
would silently build the wrong architecture.

## struct HybridWeights

A map consumed by name as the model takes each tensor, so a tensor is moved out
rather than cloned and the map shrinks as construction proceeds. A name the
model never asks for therefore stays behind, which is how an unexpected
checkpoint tensor makes itself known.

## fn from_tensors

The toy-config builders' entry point. Their tensors are generated rather than
read, and giving them the same type as a real load is what makes the gate
exercise the real construction path.

## fn load

## fn take

Removes rather than borrows, which is what makes the by-name consumption above
work.

## fn take_linear_transposed

A pytorch `Linear` weight is stored `(out, in)` and the forward wants a plain
`x.matmul(w)`, so the transpose happens once at load rather than per forward.
The transpose is done on the host by index arithmetic rather than as a tensor
operation, since the values are already in a host buffer at this point.

## fn take_vector

## fn take_row

Rank-1 as a broadcastable `(1, n)` row, because the gating scalars and the norm
weights are all broadcast against a `(n, width)` activation and burn wants the
rank to match before an expand.

## fn take_conv_taps

The depthwise convolution weight arrives `(channels, 1, kernel)` and is split
into `kernel` broadcastable rows, tap-major.

The orientation is the load-bearing half: tap `k-1` multiplies the current token
and earlier taps reach back in time. That is the pinned upstream orientation, and
reversing it produces a convolution that is anticausal rather than merely
different - which the stateless parity gate catches and the unit orientation
lock catches sooner.

## fn take_host_matrix

The embedding table stays host-side rather than becoming a tensor, because the
forward gathers rows by id and a host gather over a flat buffer is both simpler
and cheaper than a device gather for one sequence at a time.

## fn dump_read_u32

## fn dump_read_f32_matrix

The dtype is asserted rather than converted. These read the reference dumps,
where a dtype that is not what the contract says means the dump was written by
something other than the tool that is supposed to have written it.
