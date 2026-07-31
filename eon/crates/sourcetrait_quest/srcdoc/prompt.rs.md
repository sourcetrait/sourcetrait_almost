# prompt.rs

One verb, with capability composed onto it rather than spread across
several. Bare is chat, a config declaring a wider shape is the smart
command, and piped data is bound either way - so what changes between
those uses is what the caller sends rather than which command they reach
for.

## const FILE_FLAG

## const CONFIG_FLAG

## const PROMPT_FIELD

## const FILE_PROMPT_FIELD

## const PIPED_CHANNEL

Named rather than spelled inline because each appears in the signature,
in the resolution and in a lock, and a caller's spelling of `$in` has to
match what the library's `check_agreements` compares against.

## struct Prompt

## struct Asked

## fn name

Two words, so nushell treats it as a subcommand of `quest`. That is what
leaves room for the rest of the verb surface to arrive as siblings rather
than as flags on this one.

## fn signature

The input and output types are both `any`, and that is honest rather than
lazy. What comes back depends on the response shape the caller declared,
which is a runtime value inside a record - there is no static type that
describes it, and narrowing to one would be a claim the command cannot
keep.

There is no typedef argument. Piped data self-describes: the plugin
receives a real nushell value and derives the type from it, which is the
smart command's contract and the reason a caller never writes one.

## fn run

A thin wrapper over `answer`, which exists so the body can use `?`
throughout and let one conversion carry every failure to a
`LabeledError`.

## fn answer

The shape is read BEFORE the turn runs. A malformed shape is the caller's
mistake and cannot be repaired by anyone, so catching it early costs
nothing and catching it late would cost a generation.

The config defaults to an EMPTY record rather than to a composed one
carrying the trained default. Writing `{shape: {request: [text], response:
[text]}}` on the caller's behalf would put a declaration on the wire that
the caller did not make, and the model already knows that default through
training - filling it in is precisely what the design says not to do. The
harness holds the same default independently, which is why an empty
record still shapes a returned value correctly.

## fn asked

Four sources and exactly one may answer. Ranking them silently would make
two spellings of the same call mean different things depending on which
one the caller happened to remember, and the error names every source it
found rather than the first, so a caller sees the collision rather than
guessing at it.

A file is read HERE rather than handed to the model as a path. The prompt
is a Liquid template that ThinkHarness renders, so what the turn needs is
the text; passing the path would make the model's own filesystem access
the mechanism, which it does not have.

## fn piped_record

## fn field

A non-string field is not a prompt, and is passed over rather than
coerced. A record carrying `prompt: 3` is a record with a column called
prompt, not a call asking `3`.

## fn bindings

The prompt-bearing fields are stripped rather than left in place. A
caller piping `{prompt: ..., rows: ...}` said the first as an instruction
and the second as data, and leaving the instruction inside the data shows
the model its own prompt twice - once as the question and once as
something to reason about.

An emptied record binds NOTHING rather than an empty record. `assemble`
skips a config block whose visible record is empty for the same reason,
and the reason is the same here: an empty block teaches the model that an
empty value is a thing it will routinely be handed, when the real signal
is that the caller sent none.

A piped value that is not a record binds whole, because there are no
fields to consume and nothing to strip. That is the ordinary smart-command
shape - a table in, a value out.
