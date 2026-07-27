# client.rs

A handle the caller holds and a task that owns the stream, and the split
is what gives the connection exactly one owner. A caller cannot reach the
stream, so it cannot be torn out from under a write in progress - which is
the failure a design handing back the socket invites and cannot then
prevent.

There is no server half in this crate. The daemon owns its own listener,
and a client that could also serve would invite one process to be both.

## const OUTBOUND_CAPACITY

Requests are consumer-paced and small, so this bounds nothing in practice
and exists to make the channel bounded at all.

## const INBOUND_CAPACITY

Bounded deliberately, and this is the file's one genuine design property
rather than a tuning number. A full channel stops the reader, which stops
reading the socket, which pushes back through TCP to the daemon's own
generation loop - so a slow consumer slows the model rather than growing a
queue without bound. It is the property the in-process predecessor had,
where a full channel blocked the engine by design, preserved across a
network rather than lost at it.

## const RECV_BATCH

## const GOODBYE

Bounds the polite wait rather than the connection. A peer that has already
gone would otherwise hold a close forever.

## type Reader

## type Writer

Named once because the framed-half types are long enough that spelling
them inline would hide the one thing worth reading in a signature.

## struct TlsClientOptions

## struct TlsClient

## struct TlsClientHandle

## fn connect

The handshake is awaited here, before the task is spawned, and that
deviates from the reference shape deliberately. A refused certificate is
then an error `connect` returns rather than a message the caller has to go
looking for on a channel it has not started reading yet. That is the
difference between a clear failure and a silent one, and it is exactly the
class of thing a certificate problem shows up as.

## fn send

## fn recv

`recv_many` takes a batch rather than one message, which matters because
chunks arrive faster than a consumer processes them and waking per chunk
would spend the bounded channel's headroom on scheduling.

None means the connection is over rather than idle, so a caller looping on
this leaves rather than spinning.

## fn close

## impl Drop for TlsClientHandle

Dropping the handle ends the connection rather than leaking it, which is
what makes the forgetful path safe. `close` is the deliberate ending and
this is the one nobody has to remember; between them there is no way to
leave a task running with no owner.

`close` takes the task out of the handle first, so a second call is a
no-op rather than an abort of something already gone.

## fn run

Three endings, each a path rather than an accident: a cancellation says
goodbye and breaks, a server-initiated `Close` answers in kind, and every
sender dropping means the handle is gone and is the implicit teardown.

The failure is captured and raised after the writer is closed, so a
connection that broke still shuts its half down cleanly rather than
returning early and leaving the socket to the drop.

## fn say_goodbye

Says `Close` and waits briefly to hear it back, so the far end learns the
ending was deliberate instead of reading a dropped socket. The distinction
matters to a daemon deciding whether a client crashed.

The wait is bounded by a timer and it drains rather than matching, because
chunks from a turn that was still running arrive before the close does and
insisting on the next message being `Close` would fail on every cancelled
turn.
