# era.rs

## struct EraEngine

Holds the TRAIL as well as the model, because the era library's
continuation surface takes the consumed id trail rather than keeping one
of its own. A daemon session is a conversation, so something has to carry
it between turns, and the engine is the only thing that outlives one.

## fn load

Called on the container's thread through the factory, which is what makes
a not-`Send` model legal here at all. Nothing about this is callable from
the daemon's async side, and it must stay that way.

GRAPHS ARE OFF, deliberately and for a different reason than in a
one-shot binary. Captured graphs bake buffer addresses, and a daemon
restores and clears across turns for the life of the process rather than
generating once and exiting - so the epochs that retire a capture would be
routine here rather than exceptional. The era bridge's own chat engine
turns them off for the same reason.

The call ORDER is the era library's rather than ours, and it is not
arbitrary: the token map is verified before the weights are loaded, so a
checkpoint whose vocabulary does not match fails in a second rather than
after fourteen gigabytes of mmap.

## fn open

THE OPTIONS ARE NOT CONSULTED, and that follows from the singleton rather
than being an oversight. One container owns one model for the service's
life, so a session cannot ask for a different checkpoint or a different
settings profile. What it CAN do is learn which one it got, which is what
the answer carries.

The consequence worth naming: per-session generation options are not
expressible today. A sample length arriving in `ChatOptions` would have to
be stored somewhere, and the only somewhere is the engine, which is shared
- so one session would silently change another's. Threading it per TURN
rather than per session is the shape that would work, and that is a wire
change rather than something to fake here.

## fn turn

The chunk callback's return value is the CANCEL, so the loop breaks on it
rather than treating it as an error. `listening` then decides whether the
detokeniser tail is worth emitting: a cancelled turn has nobody to emit
it to, and sending it anyway would append text after the client already
stopped reading.

The trail is taken from the report unconditionally, cancelled or not,
because the caches advanced either way and the next turn has to continue
from where this one actually stopped rather than where it meant to.

## fn message_of

Library errors cross the container's trait as TEXT, because that trait is
the seam between an era and a daemon that must not know which era it is
serving. A typed error here would put the era's error enum in the
daemon's signature and undo that.

It is named for what it produces rather than `text`, because `turn` has a
parameter called `text` and the shadowing compiles as something else
entirely - a free function and a `&str` are both callable-looking in
`map_err`.
