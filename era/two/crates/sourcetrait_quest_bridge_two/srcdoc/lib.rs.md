# lib.rs

The manifest, carrying no comments by convention.

The public face is the era and the engine it builds. `BridgeTwo` is what a
consumer names, and `TwoEngine` rides beside it only because `Era::Engine` is
an associated type and a caller writing that bound has to be able to spell it -
nothing else here is reachable, which is the point of an implementation crate.
A consumer names the era once and then works entirely through the API crate's
types.

`lib` and `bridge` are aliased short so every call site says which side it is
on - the engine below or the contract above.
