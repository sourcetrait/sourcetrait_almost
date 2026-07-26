# lib.rs

The manifest, carrying no comments by convention.

THE MODULE SPLIT IS THE API's SHAPE, not an organizational habit. `all` is the
surface every consumer needs; `cli`, `srvc` and `tui` are per-consumer-class
surfaces that exist for anything genuinely specializing beyond it. Three of the
four are empty, which is the intended state rather than unfinished work - see
their own mirrors.

`error` is the one PRIVATE module, with its two items re-exported at the root.
A consumer writes `bridge::BridgeError` rather than `bridge::error::BridgeError`,
because the error is part of the contract while the module holding it is not.

This crate depends on tokio for the channel TYPES alone. It carries no runtime
and starts no threads: an implementation brings its own, and a consumer brings
its own. That is what lets camp run a current-thread runtime while the engine
runs on a plain spawned thread.
