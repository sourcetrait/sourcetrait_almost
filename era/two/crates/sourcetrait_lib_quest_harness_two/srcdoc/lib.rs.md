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

`r::nu` and the crate-global `nu` are different things and the near-collision is
deliberate rather than an oversight. The global one is this crate's own data
module, reached bare; the hub one carries nushell's own parser and engine types,
reached as `r::nu`. Both are in scope together under the star import and stay
unambiguous because one is only ever reached through `r`.

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

SIX MODULES ARE `pub` RATHER THAN FACED, which is the one deliberate exception
to the rule above. `bubble`, `channel`, `harness`, `nu`, `session` and
`template` are pathed into by consumers - bquest alone reaches `lib::nu` from
several hundred call sites - so their contents are the contract and a curated
re-export would only be a second name for the same thing. Everything else stays
private behind the face.

They were a separate crate until they were folded in, and the honest reason
that crate existed is worth keeping because the reasoning was the same one that
had already been wrong once. It was there so an eon binary could speak the
program's data language "without linking an era's engine" - a compile-weight
claim, unmeasured, of exactly the kind that had produced the harness crate. It
did not survive contact: every eon binary already depended on this library,
directly or through the era bridge, so the property the split protected was
held by nothing. And a binary carries only what it CALLS, so a consumer that
never touches the engine was never carrying it whatever the manifest said.

## mod think_harness

Private, with its four submodules reached only through the public face above.
That asymmetry is the point: a consumer builds a `Request`, calls `step`, and
reads an `Answer`, while the evaluator, the def contract and the sub-turn
machinery are none of its business.

It lived in a crate of its own until the merge, and the honest reason it did is
worth keeping because the reasoning was wrong twice over. A build-weight cost
was raised without being measured, and a crate boundary was drawn to solve it.
The measurement, taken later, said the boundary bought nothing: the workspace
builds as one unit, so the nushell command set the evaluator needs was already
being compiled for the member crate that declared it, and rustc drops dead code
per artifact so no consumer's binary carried what it did not call. Both halves
of the original argument were about a cost that was never there.
