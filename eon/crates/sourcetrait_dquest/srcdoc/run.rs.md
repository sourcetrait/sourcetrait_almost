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

## fn start

THE ORDER IS THE DESIGN. The secret data home is resolved BEFORE the
profile is placed, so the one step that mutates anything runs only after
every check that can fail has passed. That is the same
preflight-before-mutation rule the certificate library follows, applied to
a startup path rather than to an install.

It also decides which failure an operator sees first. Resolving the home
last would place a profile, report a missing certificate, and leave the
real fault - an unset variable - unmentioned.

The two branches this used to carry collapsed into one condition. The
trigger was whether the profile had just been written, which answered
"has this box ever been set up" rather than "can this daemon serve TLS";
it is now whether the material exists, with the profile write happening on
the way. The message needed no rewording, because it already said that no
certificate exists yet rather than that no profile did.

## const ADDRESS

A CONSTANT rather than a setting. Everything is loopback today, and a
port becomes configuration the moment something needs it to be - which is
a decision rather than an implementation detail, and adding a knob nobody
asked for is how a config surface grows without anyone choosing it.

## fn serve_forever

THE RUNTIME IS BUILT HERE rather than by an attribute on `main`, which
keeps `main` two lines and keeps the whole daemon reachable from an
in-crate test without a process.

THE ENGINE LOADS ON THE CONTAINER'S THREAD, and the container takes the
loader itself rather than a loaded engine. That is what lets the listener
bind IMMEDIATELY: binding does not wait on fourteen gigabytes, so a client
connecting during load is answered rather than refused by a closed port.

A FAILED LOAD LEAVES THE DAEMON UP. The container answers every request
with the reason instead, which reaches a client as a refused open naming
what went wrong. Exiting at startup would be tidier and tells a client
that connects later nothing at all - it would meet a closed port and have
to guess between "not started", "wrong port" and "model missing".

Verified against the real binary with a scratch configuration home, so
the engine could not load: it bound, stayed up past a timeout, and said
nothing.

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
