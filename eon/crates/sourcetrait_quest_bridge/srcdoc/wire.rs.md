# wire.rs

The transport's language. The general shape is the standing raw-TLS
pattern rather than anything decided here; what is quest's own is what
these messages carry, and naming them properly decomposed an existing
surface rather than wrapping it.

The decomposition is the file's history and its argument. A single
`ChatEvent` used to carry everything the daemon said. Sorting its variants
by what they answered emptied it: `Ready` was an open response, `TurnDone`
a turn response, `Closed` transport, and only `Chunk` and `Error` were
ever unsolicited. A type whose members answer different questions reads as
a grab bag because it is one, and the fix is to ask what each was an
answer to rather than to reorganise it in place.

Every request can also answer that it failed, which the first cut of this
missed and which is the part most likely to be undone by someone tidying.
A failure with no message of its own has to travel as a general fault,
which loses which request it belonged to and rebuilds the grab bag one
layer down. Granularity is cheap here - a derive-driven typed codec has no
hand-written parser to keep in step and no schema to migrate - so
collapsing distinct outcomes is lost information rather than economy.

## const MAX_FRAME_BYTES

A ceiling the codec refuses before buffering, which is the property that
matters rather than the number. Without it a peer announcing an enormous
frame allocates it here.

## struct OpenRequest

## struct OpenResponse

## struct OpenRefusedResponse

Its own message rather than an outcome inside `OpenResponse`, because a
refusal has no era to report. Folding it in would mean inventing an
`EraInfo` to carry a failure, and a caller that must branch anyway is
better served branching on the message.

## struct TurnRequest

## struct TurnResponse

## struct TurnFailedResponse

The same argument as the refused open, and it is the sharper case: a
failed turn has no accounting, so token counts and timings inside a folded
`TurnResponse` would be fiction rather than merely unused.

## struct CancelRequest

## struct CancelResponse

`stopped` distinguishes a cancel that reached a running turn from one that
arrived with nothing generating. Both are successful cancels, which is why
this is a field rather than two messages - the caller asked for a state
and got it either way.

## struct ResetRequest

## struct ResetResponse

Both unit structs, and deliberately structs rather than bare enum
variants. Every other operation is a request-and-response pair, so making
these the exception would mean a reader could not tell from the shape that
reset works like the rest.

## struct TurnChunk

This is not a notice, and that is the distinction the whole layout rests
on. It belongs to the `TurnRequest` that is running - many arrive before
that request's own response - so it is that response arriving in pieces
rather than something the daemon says unbidden.

## struct ServerFaultNotice

## struct ServerShutdownNotice

Distinct from a transport `Close`, which ends one connection. This says
the far end itself is leaving, so reconnecting will not help - and a
consumer that cannot tell those apart retries forever against a daemon
that has gone.

## enum ServerNotice

Grouped rather than flattened into `ServerToClient` so the split between
an answer and a notice is structural. A new notice then widens this enum
rather than the top-level language, which is what stops the unbidden set
from silently becoming most of the protocol.

## enum ClientToServer

## enum ServerToClient

Split by direction because each is its own language. It also makes a
mis-send a compile error: a half that can only send one of them cannot be
handed the other.

`Close` appears in both and is the transport's rather than the session's.
Ending a connection and ending a session stay separable, which matters
because a consumer reconnecting is not a consumer starting over.

## struct BitcodeCodec

The length codec delimits and bitcode fills, which is the division the
standing pattern prescribes. Never hand-roll the framer: a partial read, a
length prefix split across two packets, backpressure and the maximum-frame
guard are all things `LengthDelimitedCodec` already handles and a
read-the-prefix-then-read-n loop gets wrong.

Typed per direction through `PhantomData`, so the type parameter costs no
bytes and buys the mis-send guard above.

Bitcode rather than a self-describing format because the message set is a
closed enum - there is nothing to discover on the far side. An earlier
draft used NUON for inspectability and the argument recorded for changing
it was that bitcode spared this crate a data-core dependency, which was
never a real cost. What actually recommends it is the closed enum plus its
being the house pattern's own payload.

A decode failure is `InvalidData` rather than a panic, which is what keeps
one corrupt frame from taking down the task that owns the connection.

The `Debug` impl is hand-written and finishes non-exhaustively because
`LengthDelimitedCodec` has no `Debug` of its own, so the derive would not
compile at all rather than merely print poorly.
