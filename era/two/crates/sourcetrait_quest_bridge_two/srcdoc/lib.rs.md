# lib.rs

The manifest, carrying no comments by convention.

The public face is ONE item - `BridgeTwo`. Everything else is crate-private,
which is the point of an implementation crate: a consumer names the era once
and then works entirely through the API crate's types.

`lib` and `bridge` are aliased short so every call site says which side it is
on - the engine below or the contract above.
