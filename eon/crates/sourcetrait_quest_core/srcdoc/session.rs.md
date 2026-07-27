# session.rs

Per-session text logging for both sides of the seam. Text first and
deliberately plain, because the design says to improve it later and a
format decided before anything reads one would be decided on nothing.

## struct SessionLog

Takes its path SEGMENTS rather than deriving them, and that is the
layering decision worth keeping. A ThinkspaceNom is derived from a
username and a SessionNom is random, which makes minting them the job of
whoever owns the session - the daemon. A log that minted its own identity
would need a hash and a random source in the crate every other crate
depends on, to answer a question it is not the one asking.

The consequence is that this type is the same on both sides. The Questness
side passes two segments, a thinkspace over a session; the harness side
passes one, a session. Nothing here knows which side it is on.

## fn is_plain_segment

A nom arrives from a client, so it decides a directory name and therefore
has to be prevented from deciding a directory ELSEWHERE. Rejecting `..`,
separators and the empty string keeps a segment inside the root it is
joined to.

This is cheap and it is worth the paragraph because the alternative is
invisible: joining an unvalidated `../..` to a root produces a perfectly
ordinary path that writes somewhere else, with no error anywhere. The
admission check for a Thinkspace is mutual TLS, and an unrecognised user
gets one generated with no registration step - so a nom is attacker-shaped
input by design rather than by accident.

Note what it does NOT do: it is not a sanitiser and it does not rewrite
anything. A hostile segment is refused rather than cleaned, because a
cleaned nom would silently address a different session than the one asked
for, which is worse than a refusal.

## fn append

Opens per record rather than holding a handle, which costs a syscall per
write and buys two things worth more at this stage. A crash leaves
everything already written, since there is nothing buffered to lose. And
`SessionLog` stays `Clone` and needs no interior mutability, so a caller
can hand one to two places without threading a lock through the turn
loop.

The record format is a label line and a body, trimmed at the end so the
file's line structure survives a body that arrived with trailing
newlines. There is no timestamp: a duration would make two runs of the
same session textually different, and nothing yet reads these logs
mechanically.

## fn read

Treats a missing file as empty rather than as an error, because a session
that has been opened and not yet written to is an ordinary state and a
caller reading its own log back should not have to distinguish it.
