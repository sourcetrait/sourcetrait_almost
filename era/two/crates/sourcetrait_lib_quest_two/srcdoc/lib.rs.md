# lib.rs

The manifest. It carries no comments by convention - the module tree, the star
imports and the public face are self-describing - so what follows is only what
the file's SHAPE implies rather than states.

THE MODULE TREE IS DECLARED INLINE with nested `mod` blocks and there is no
`mod.rs` anywhere. Adding a sibling is creating a file and adding a line, with
no per-directory boilerplate. The cost is that leaf names must stay descriptive
where the same concept appears at several nesting levels, which the star
imports then flatten at every use site.

THE STAR IMPORT IS WHY EVERY OTHER FILE HAS EXACTLY ONE `use`. Crate-visible
items are re-exported here at global scope, so inner-crate items must be
uniquely named; `consts` is the deliberate exception and is always pathed.

## mod r

The re-export hub for external items whose qualifying path is long. A hub
module's name must DIFFER from the crate it re-exports, or `use crate::*` makes
the identifier ambiguous - which is why these are `flash`, `nu` and `sampling`
rather than the crate names.

`gen` is unusable as a family name here: edition 2024 reserves it as a keyword,
and it bites re-export hubs first.

## pub use

The deliberate PUBLIC FACE, and every entry is a contract rather than a
convenience. Internal errors, building blocks and markers are absent on
purpose; an extra entry fossilises, because removing it later is a break while
never adding it costs nothing.

Two of these exist only for the offline replay instrument - the speculation
index and policy - so that the replay drives the shipped code rather than a
copy of it. That is the one case where a public entry is justified by a test
harness.
