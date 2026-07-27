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
line whenever a caller needs it.

## enum Started

An unmet precondition is an OUTCOME rather than an error, because nothing
failed - the operator has simply not minted a certificate yet. Modelling
it as a variant is what lets the message be composed and shown where the
condition is detected, while `run` still turns it into a non-zero exit.

## fn run

A DAEMON IS SILENT UNLESS SOMETHING IS WRONG (the_user's rule). A normal
start prints nothing at all: no profile path, no readiness line, no
narration. Operational narration is what logging is for, and this crate
has no logging yet - so the honest state today is silence rather than a
placeholder line standing in for it.

The exit code carries what the silence does not. A start that could not
proceed exits non-zero, so a supervisor sees a failure rather than a
daemon that came up and vanished.

## fn certificate_owed

The only thing this binary ever says to a user, and it is composed here
rather than inside an error type so that the profile name and its path
can be cyan without any error carrying escape codes.

It is the operator's sole instruction at that moment, so its command
spellings are locked by test - both that the verb precedes the profile
and that the old profile-first order is ABSENT. The negative half is what
catches a partial edit rather than a missing one. The order shipped wrong
once, which is why the lock exists at all.

The path is home-collapsed but the `<dir>` is left as a placeholder. A
suggested staging directory would be this machine's layout leaking into
shipped source, and `install` consumes that directory anyway, so it is
genuinely the operator's to choose.
