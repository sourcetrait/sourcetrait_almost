# load.rs

## struct TensorInfo

## fn tensor_inventory

Reads the JSON header alone and never touches tensor data, which is what makes
it affordable against a 13.84 GiB shard - the adapter geometry guard calls it
on every adapter load.

The implausible-header bound exists because the length is the first eight bytes
of an arbitrary file: without it, a wrong or truncated file allocates whatever
those bytes happen to say before anything can reject it.

Entries come back name-sorted because the callers diff them, and an unstable
order would make an inventory comparison report differences that are not there.

## fn mmap_weights

Tensors materialize lazily per `get()`, so the constructors' access pattern is
what actually pages the shard in. That is also the seam the adapter merge wraps
- see `adapter.rs` - which is why nothing downstream of here knows whether it
is reading merged or plain weights.
