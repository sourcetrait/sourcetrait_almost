# run.rs

## struct Cli

Empty, and parsed anyway. The daemon takes no arguments yet, so the
parser buys only `--help` and `--version` - which a background process is
exactly the kind of thing someone interrogates from a shell. It also
means the first real argument is a field rather than a new dependency and
a new entry point.

There is deliberately no configuration-home override, even though
`srcert` has one. The startup path takes its path as an argument, so the
locks drive it against a scratch root without one, and adding a flag is a
line whenever a caller needs it. A daemon's argument surface is design
rather than convenience.

## fn start

Reports the profile path on a normal start, which is the one thing an
operator wants to see in a log when a certificate stops working: the
daemon says which file it is minting from, so a customisation in the
wrong place is visible rather than inferred.

The serving half says plainly that it is not built. An empty success
would be indistinguishable from a daemon that came up and served
nothing, and this binary exists today for its startup precondition
alone.
