# lib.rs

The manifest. It carries no commentary by convention - the module tree, the star
imports and the re-export block are self-describing - so what follows is only
what the file's shape implies rather than states.

The module tree is declared inline with no `mod.rs` anywhere, and the star
import is why every other file in the crate has exactly one `use`. Crate-visible
items are re-exported here at global scope, so inner-crate items must be
uniquely named; `consts` is reached through `lib::consts` and is the deliberate
exception.

Two modules are feature-gated in the tree itself rather than internally:
`objective` and `train` exist only under `train`. That is what makes a non-train
build genuinely smaller rather than merely a build with unreachable code, and it
is why so many items elsewhere carry a `dead_code` allow - a non-train build
sees the framework with its only consumer absent.

The re-export list is flat where the source is not, which is the trade the
convention makes. A reader looking for where `verify` or `pack_row` lives reads
this file, not the call site.

## use std

The families the crate leans on everywhere, brought in at global scope because
std is prelude-worthy under the convention. The `unused_imports` allow covers
the feature-gated case: a non-train build genuinely does not reach all of them.

## use lib

The library is bound as `lib` rather than by its crate name, so every call site
reads `lib::nu::...` at four characters instead of twenty-six. That is the
sister-crate binding convention, and it is what keeps the star-imported
namespace readable.

## type CpuBack

The reference grade. It is host-only and shares nothing with the CUDA stack
under test, which is what makes it usable as an oracle for that stack rather
than a second opinion from the same machinery.

## type CudaBack

The fast oracle and the trainer ride the same backend, and its default float is
bf16. That does not make the recurrence bf16: the gate, norm and recurrence
region casts to f32 per tensor inside the block forward, so the reference
stacks' f32-state discipline holds on every grade and the cast is a no-op on the
CPU one.

Gated on either feature that needs it, because the two features are separate
build postures over one type.
