# config.rs

The runtime config record, split by AUDIENCE rather than by schema.

That split is the whole file, and it is worth stating because the alternative
is what an earlier revision of the design assumed. The config is not a fixed
structure to validate against - it is composed per call and the model reads
whatever arrives, records being open, so an unfamiliar key is a runtime fact
rather than a schema violation. What is fixed is only which keys are addressed
to us instead of to the model.

## const THINK_HARNESS_KEYS

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

Everything not in the ThinkHarness list passes through untouched, including keys
nobody has sent before. `env` is the clearest case: it is a curated environment
rather than the real one, and what the model learns about it is a disposition -
that a key called `env` is its own environment - rather than a field this code
knows anything about.

Key order is preserved, because field order is data everywhere else in this
program and there is no reason for it to stop being data here.

## fn prepare

The `liquid` key's presence is the templating switch, and its value is a
record of template bindings - the infill channel. Each key binds under its
own name beside the pass channels, which is how a literal reaches the
question's own text with nothing piped: `{liquid: {train: {...}}}` binds
`train` and the template reads `{{ train.x }}`. An empty record is a
templated prompt addressing the pass channels alone. The training set
spells this same record in its set rows, so the runtime and the syllabus
share one mechanism rather than the tree inventing a second one.

An earlier revision read only the presence and bound the passes alone,
recording that nothing was inferred from the value because the design had
not yet given it a meaning. The realignment gave it one, and this is the
line that changed.

The value must BE a record, and a liquid key shadowing a pass channel's
name is refused: the caller is code, and one name quietly meaning two
values is the worse reading in both directions.

A template referencing an unbound channel fails the whole turn rather than
rendering a hole, which follows from the renderer being strict. That is the
wanted behaviour: a prompt that quietly lost a value is worse than one that
never went.
