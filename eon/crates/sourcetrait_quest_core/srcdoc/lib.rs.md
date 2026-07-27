# lib.rs

The manifest, carrying no comments by convention. What follows is only what the
file's SHAPE implies rather than states.

This crate is the era-independent core: the data grammar the whole toolset
speaks, and the template grammar beside it. The test of whether something
belongs here is whether it would change when the era's model changes. Nu types,
values, NUON and Liquid do not; a chat template byte-locked to one checkpoint
does, which is why rendering stayed in the era library.

The dependency direction is deliberate and reads backwards at first glance: the
era library depends on this eon crate rather than the reverse. It follows from
the era rule that an eon component reaches an era only through a bridge
artifact. A shared core in an era workspace would force every eon binary to
link that era's engine to speak NUON, which for a nushell plugin means dragging
candle and CUDA in to parse a value.

## mod r

The re-export hub for external items whose qualifying path is long. A hub
module's name must DIFFER from the crate it re-exports, or `use crate::*` makes
the identifier ambiguous, which is why the family is `nu` rather than
`nu_protocol`.

## pub use

The deliberate PUBLIC FACE. The error pair is here because consumers wrap it as
a transparent source variant in their own crate error, and `nu` is a whole
module re-export rather than a list because the era library re-exports it again
under its historical path.
