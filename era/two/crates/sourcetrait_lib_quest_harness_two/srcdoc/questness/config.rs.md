# config.rs

The runtime config record, split by AUDIENCE rather than by schema.

That split is the whole file, and it is worth stating because the alternative
is what an earlier revision of the design assumed. The config is not a fixed
structure to validate against - it is composed per call and the model reads
whatever arrives, records being open, so an unfamiliar key is a runtime fact
rather than a schema violation. What is fixed is only which keys are addressed
to us instead of to the model.

## const QUESTNESS_KEYS

## const LIQUID_KEY

## const CONVERSATION_KEY

## const CONVERSATION_KEEP

## enum Conversation

Teardown-by-default is the_user's ruling stated as a type: a bare call
is a fresh conversation, always, and nothing has to be sent to get
that - the inversion of the wire's Reset, which nothing was sending. The
opt-in is `{conversation: keep}`, and it must be said on every call that
wants the conversation to survive that call's end, so the first bare
call after a kept run is fresh again.

An unknown value refuses rather than defaulting in either direction,
because the caller is code and both silent readings are wrong: reading
junk as keep leaks a conversation, reading it as teardown silently
drops one the caller meant to keep.

## struct PreparedTurn

## fn split

Everything not in the Questness list passes through untouched, including keys
nobody has sent before. `env` is the clearest case: it is a curated environment
rather than the real one, and what the model learns about it is a disposition -
that a key called `env` is its own environment - rather than a field this code
knows anything about.

Key order is preserved, because field order is data everywhere else in this
program and there is no reason for it to stop being data here.

## fn prepare

The `liquid` key's PRESENCE is the switch, and that is all this code reads from
it. The design states the presence rule and does not state what the value
means, so nothing is inferred from it - a record there is not treated as extra
fill data, because inventing that convention would make it real by
implementation rather than by decision. When the value acquires a meaning, this
is the line that changes.

A template referencing an unbound channel fails the whole turn rather than
rendering a hole, which follows from the renderer being strict. That is the
wanted behaviour: a prompt that quietly lost a value is worse than one that
never went.
