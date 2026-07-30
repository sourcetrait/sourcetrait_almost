# error.rs

Everything error-shaped in the crate lives here regardless of which module it
represents, so a module's own error wraps in transparently as it lands and a
nested call can bubble a module result straight into the crate result.

## type LibQuestResult

## enum LibQuestError

The transparent variants carry no message of their own, so a wrapped foreign
error displays exactly as it would have unwrapped. The `whatever` variant is
what `snafu::whatever!` and `ensure_whatever!` construct, which is why almost
every fallible path in this crate raises through those macros rather than
naming a variant.
