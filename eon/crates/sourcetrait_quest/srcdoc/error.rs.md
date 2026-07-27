# error.rs

## enum QuestPluginError

`Envelope` and `Insufficient` are the two variants that are not failures
of this crate at all. One is the model answering off the shape the caller
declared and the other is the model declining, and both are ordinary
outcomes of running a model that has not been trained on this grammar.
They are errors here only because a plugin command returns one value or
one error, and neither is a value the caller asked for.

`Envelope` keeps its rows as a list rather than flattening them into the
message. A caller that got a shape it did not ask for wants to know which
member was missing, not that something was, and the repair vocabulary
collects rather than bailing at the first row for the same reason.

## impl From<BridgeError> / From<LibQuestError>

Written out beside the transparent variants rather than instead of them.
Snafu's transparent generates the conversion from the BOXED error, which
is what the variant stores; these take the unboxed one, which is what a
`?` at a call site actually has. Without them every call site boxes by
hand and the noise is in the code that matters rather than here.

## fn envelope

## impl From<QuestPluginError> for LabeledError

The only route out, and that is a protocol fact rather than a style
preference. A plugin's stdout is the msgpack channel, so a diagnostic
printed there is a corrupt frame rather than a message - it replaces the
failure being reported with a stranger one. Everything this crate can
fail with therefore has to arrive as a value on the wire, and this is the
conversion that guarantees it.

The rows ride as `help` rather than in the message. The message is what a
reader sees first and should say what happened; the rows say which parts,
and a caller repairing a call wants them in that order.
