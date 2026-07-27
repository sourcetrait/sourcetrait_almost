# style.rs

The house convention for human-facing command-line output, narrowed to
what a daemon is allowed to say. A near-identical and larger copy lives in
the certificate tool's command-line crate, in its own repository; the
duplication is deliberate, since a shared crate across a repository
boundary is exactly the dependency shape already carried as debt.

## Why this copy is so much smaller

A DAEMON IS SILENT UNLESS SOMETHING IS WRONG (the_user's rule), and
normal operation belongs to logging rather than to a user. That leaves
this binary with exactly one thing to say, so the module has one emitter
and no informational, success or question halves at all. They are absent
rather than present and unused, because dead code behind an allow
attribute would have been the worse answer.

EVERYTHING GOES TO STDOUT, which inverts the sibling copy's split. The
reason is that this output is SETUP rather than operation - the daemon is
not running yet when it says any of this - so there is no operational
stream to keep clear and nothing to justify splitting across two.

## fn collapse_home

Split from `pretty_path` so the rule is testable without an environment,
and the rule has one edge worth the test: a longer name that merely
STARTS with the home is not inside it. `/home/boxer/a` under a home of
`/home/box` must not collapse, which a bare prefix comparison gets wrong.
That is the substring-prefix trap in path clothing.

The result is for READING rather than for use. Nothing consumes it back
as a path, and a collapsed path handed to a filesystem call would not
resolve.

## fn paint

Takes the colour decision as an argument rather than testing the stream
itself, which is what makes both branches lockable. The escapes come from
`anstyle`, already in the graph because clap uses it for its own coloured
help, so nothing new enters the dependency graph and `--help` and our own
lines agree about colour.

## fn fail

Splits on newlines so a multi-line cause keeps the prefix on every line.
That matters here specifically: the certificate-owed message is five
lines carrying two commands, and without the split only its first line
would be attributable to this binary.
