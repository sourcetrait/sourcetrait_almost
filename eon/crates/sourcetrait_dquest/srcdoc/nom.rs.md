# nom.rs

## const BASE62

THE ALPHABET IS THE VALIDATION. It carries no separator, no dot and no
character a path resolves, so a nom is a plain path segment by
construction rather than by being checked afterwards - which is what lets
one be joined to a log root with no sanitiser standing between them.

That matters because a username arrives from a client and an unrecognised
one gets a space with no registration step, so it is attacker-shaped input
by design. A cleaned nom would silently address a DIFFERENT space than the
one asked for; a hashed one cannot address anything but its own.

Digits are emitted most significant first so a nom sorts the way its
number does, which costs nothing and makes a listing readable.

## struct ThinkspaceNom

HASHED RATHER THAN COUNTED, because the same user must find the same space
across restarts. A counter would hand the second run a different space and
strand the first run's logs.

## fn of

Takes the first eight bytes of the digest, which is the recorded nom shape
- a base62 u64 - rather than an arbitrary truncation. Collision resistance
is not doing security work here: the space is keyed by username on a box
where the user is always the same one, and mutual TLS is the whole of the
admission check.

It cannot fail, which is deliberate. An unrecognised user gets a space
rather than an error, so there is no registration step for this to
consult and no refusal for a caller to handle.

## fn base62

Zero is spelled rather than falling out of the loop, since the loop emits
nothing for it and an empty nom would be the one value that is not a
plain segment.
