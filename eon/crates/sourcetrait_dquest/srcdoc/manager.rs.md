# manager.rs

## struct SessionManager

Clone shares the registry rather than copying it, which is the whole
reason it is cheap to hand one to every accepted connection. A copying
clone would give each connection its own private view and the listing
would always read one.

## struct SessionTicket

DEREGISTRATION IS A DROP RATHER THAN A CALL, and that is the design
decision in this file. A session ends several ways - a handshake that
fails, a peer that vanishes, a task that panics, a clean close - and a
`leave()` call would have to be reached on every one of them. There is no
path that forgets, because forgetting is not expressible: the place comes
back when the ticket goes out of scope, however it goes out of scope.

A lock holds it by panicking inside a caught unwind and asserting the
registry is empty afterwards.

## fn admit

Called BEFORE the handshake in `serve`, deliberately. A peer that opens a
socket and never completes a handshake is a connection being attempted,
and a registry that only counted successful ones would show nothing while
several such peers held sockets open.

## fn locked

A poisoned mutex is RECOVERED rather than propagated. The registry holds
noms and nothing else, so a panic while it was locked cannot have left it
half-updated in any way that matters - and treating one session's panic as
a reason to refuse every later session would turn a contained fault into
an outage.
