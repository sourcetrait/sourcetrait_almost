# error.rs

Everything error-shaped in the crate lives here regardless of which module it
represents, so a module's own error wraps in transparently as it lands.

Two transparent variants and a catch-all is the whole enum, which is right for
this crate: its failures are a file that will not open, a driver that exited
non-zero, and an environment that is not provisioned. None of those wants a
named variant, because nothing matches on them - the consumer prints and exits.

## type RefquestResult

## enum RefquestError

The `whatever` variant is what `snafu::whatever!` constructs, and almost every
fallible path here raises through it. The messages carry the remedy where one
exists, which is why the missing-environment error spells out the provisioning
command rather than merely reporting absence.
