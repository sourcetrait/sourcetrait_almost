# lib.rs

The manifest, carrying no comments by convention.

THE MODULE SPLIT IS THE TRANSPORT'S SHAPE. `wire` is the language, `tls` the
material both ends present, `client` the half that connects, and `all` the data
those messages carry. There is no server half here: the daemon owns its own
listener, and a client that could also serve would invite one process to be
both.

It used to carry three more modules - `cli`, `srvc` and `tui` - one per
consumer class, empty and waiting for something to specialize the era trait
beyond the shared surface. The trait is gone and nothing ever specialized them,
so they were speculative surface twice over. An empty public module is not free:
it is a promise about where a future reader should put something, and this one
pointed at an architecture that no longer exists.

`error` is the one PRIVATE module, with its two items re-exported at the root.
A consumer writes `bridge::BridgeError` rather than `bridge::error::BridgeError`,
because the error is part of the contract while the module holding it is not.

THIS CRATE NEEDS A RUNTIME, which is a change from when it carried channel
types alone. `client` connects a socket, awaits a handshake, spawns the task
that owns the stream, and bounds its goodbye with a timer - so a consumer must
already be inside a tokio runtime before it calls `connect`. What the crate
still does not do is CHOOSE one: no runtime is built here, and a caller running
current-thread or multi-thread is equally served.
