# lib.rs

The manifest, carrying no commentary by convention.

What its shape says: this crate is a front end, not an implementation. Every
module here is either a verb that spawns a pinned python driver, the plumbing
those verbs share, or one of the two things that genuinely compute in Rust -
the safetensors header read and the dump comparison.

The name is the trap this file cannot warn about, so the mirror does. In era one
`refquest` was the compat crate's baseline binary, an incumbent Olmo 3
implementation in Rust. In era two it reimplements nothing at all: it drives
ai2's original stack, which is the sanctioned python surface and the thing our
own engine is measured against. No era-one fact about refquest transfers.

## use lib

Bound as `lib` and carried with an `unused_imports` allow, which is honest about
what this crate does: almost nothing here needs the engine. The binding exists
for the checkpoint-shaped helpers rather than for the model.

## use std

`process` is the load-bearing one. This crate's real work is spawning and
streaming, so the standard library is most of its dependency surface - the
manifest lists five crates and one of them is the workspace library.
