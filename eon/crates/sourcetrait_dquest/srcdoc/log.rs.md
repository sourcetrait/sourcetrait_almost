# log.rs

The daemon writes its own text log rather than reaching for the era
library's, and that is the point of the module existing at all. An eon
component depending on an era's library to append to a file would open
exactly the edge the bridge pair exists to prevent, for a convenience.
The two types are near-identical and the duplication is deliberate on the
same terms the style module's is.

## enum LogFile

ONE FILE AT ONE GRAIN IS UNREADABLE, which is what this splits. A 32K
emission and a one-line open notice were landing in the same file, so the
bulky record drowned the outline of what actually happened. Four grains,
and the discriminator between them is who reads them and when: transport
when a connection misbehaves, turn to see what a turn did, emission when
the model's own text is the question, chunks while it is still being
written.

Named for the reader rather than for the writer. `turn.log` is the file
you open first because it is the only one guaranteed short.

## struct SessionLog

Now a DIRECTORY rather than a file, and everything else follows. The
segments are still supplied rather than derived, because minting a
session's identity belongs to whoever owns the session - and the
validation on them is the load-bearing half: a nom arrives from a client
and an unrecognised user gets a Thinkspace with no registration step, so
these are attacker-shaped by design. A hostile segment is REFUSED rather
than cleaned, since a cleaned nom would silently address a different
session than the one asked for.

`append` reopens per record and that is correct here. These are written
once per turn, the open is cheap against that, and holding four handles
for a session that may write to one of them would cost more than it
saves.

## struct ChunkSink

THE ONLY RECORD WRITTEN WHILE THE MODEL IS STILL WORKING. Every other
record in this crate lands after the generation that produced it has
already ended, which is the whole reason a long turn was
indistinguishable from a hung one: the log was correct, complete, and
silent for as long as it mattered. A generation that runs to the 32,768
budget takes minutes, and until this existed nothing on any surface said
it was progressing.

It holds its file OPEN, unlike the labelled records, and that asymmetry
is the reason it is a separate type. This one is written per TOKEN rather
than per turn, so reopening fifty times a second to append a few bytes is
the one place that cost would be real.

The text goes down VERBATIM and unframed. A `tail -f` then reads as the
answer forming rather than as a record format, which is what someone
watching actually wants; framing each token as `== chunk ==` would make
the file technically complete and practically useless.

A write failure is dropped rather than raised, which is the same trade
every record here makes: the log records the work rather than being part
of it, and trading a turn for the record of it is the wrong direction.

## fn is_plain_segment

Blank, dot, dot-dot and anything carrying a separator are all refused
together. Joining an unvalidated `..` to the root produces an ordinary
path that writes somewhere else with no error anywhere, so the check is
what keeps a client-supplied name from deciding a directory outside the
one it was given.
