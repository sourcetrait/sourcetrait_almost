# lib.rs

The manifest, carrying no comments by convention.

The crate is a library plus a thin binary, matching dquest: `main.rs`
calls `run::run` and nothing else, so the whole plugin is reachable from
an in-crate test without spawning a process and speaking msgpack at it.

Only `run` is public. Nothing consumes this crate - it is an executable -
so a public face here would be a contract with nobody.

The crate is named for the tool rather than for what it is built as.
`quest` is the flagship smart command in the toolset; serving it as a
nushell plugin is how it is delivered, not what it is. Era one's own
plugin crate is where the plugin-shaped name already lives, and it
belongs to that era.

## use sourcetrait_lib_quest_two as lib

## use sourcetrait_quest_bridge as bridge

Aliased short, and in the same pair of names the era bridge uses, so
every call site says which side it is on: `lib` is the codec and the
harness, `bridge` is the way to the daemon.

Depending on the era library at all was once ruled out, on the argument
that it would drag an era's engine into an eon binary. That argument did
not survive measurement: a binary carries only what it calls, so a plugin
that never drives the engine never links it, whatever the manifest says.

The dependency that matters is subtler and is a lockfile property rather
than a binary one. This crate and the era library must resolve ONE
`nu-protocol`, or a `Value` the library builds is a different type from
the one this crate returns to nushell - which reads as an inscrutable
type error rather than as a version problem. The workspace pins the nu
crates at the estate's own tag for that reason, and the lock is re-read
after touching them.

## use nu_plugin / nu_protocol

Referenced by path rather than through the re-export hub, because both
qualifiers sit inside the sixteen-character rule. The traits are named in
impl headers for the same reason.
