# error.rs

Everything error-shaped in the crate lives here regardless of which module it
represents, so a module's own error wraps in transparently as it lands and a
nested call can bubble a module result straight into the crate result.

## type BquestResult

## enum BquestError

The transparent variants carry no message of their own, so a wrapped foreign
error displays exactly as it would have unwrapped.

The shell error is boxed where the others are not. It is much the largest of
them, and an unboxed variant sets the size of every `BquestResult` in the
crate - which is nearly every return type here.

The `whatever` variant is what `snafu::whatever!` and `ensure_whatever!`
construct, and almost every fallible path in this crate raises through those
macros rather than naming a variant. The trade is deliberate: a named variant
per failure would be a large enum whose arms nothing matches on, since the
consumer is a command-line front end that prints and exits.
