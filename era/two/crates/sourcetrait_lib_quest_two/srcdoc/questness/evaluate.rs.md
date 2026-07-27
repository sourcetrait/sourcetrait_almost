# evaluate.rs

The engine `evaluate` round-trips on: the one nu mode that runs inside
Questness rather than leaving for the client's harness.

IT SITS BEHIND A DEFAULT-OFF FEATURE, and that is a build-weight decision
rather than a doubt about the code. Turning it on pulls the whole nushell
command set into the dependency graph, and the trainer, the baseliner and the
engine all read and write NUON without ever evaluating. Making them pay for a
shell they never call would have added it to every burn and candle build in the
era workspace. Questness and the daemon turn it on; nothing else does.

WHAT THIS IS NOT IS A SECURITY BOUNDARY, and the file should be read with that
in front. Restricting by non-registration is real - an absent command is absent
at parse time, not refused at run time - but it is capability restriction inside
our own address space, and it does nothing about a construct we failed to think
of. The containment is the service-level bubblewrap; everything here is defence
in depth beneath it.

## const PARSE_STACK_BYTES

The parser recurses through nested expressions and module resolution, so its
stack cost scales with input depth, and an overflow is a fatal runtime abort
that `catch_unwind` does not save you from. A default thread gets roughly eight
megabytes and an async runtime worker roughly two, so anything that parses needs
its own sized thread rather than a borrowed one. Sixteen is chosen to sit well
clear of the depth a generated body plausibly reaches; the figure is a margin
rather than a measurement.

## const DENIED

MEASURED, and the measurement changed the design. A throwaway probe enumerated
what each context registers: the language core carries 53 declarations and 422
arrive with the shell context, and every name in this list is in the second
group only. `exit`, `exec` and `panic` - the three the embedding literature
warns end your host - are simply not present unless you ask for the shell.

So this array is a LOCK on not having asked, rather than a mitigation. An
earlier draft of this file shadowed those three with refusing replacements,
which would have been a guard against a danger that did not exist here and would
have read in the record as a risk we had handled. The test asserts none of them
resolve, which fails loudly if someone later adds the shell context and quietly
costs nothing until then.

## const PARSE_TIME_LOADERS

The reach registration cannot close, recorded rather than papered over. `use`
and `overlay use` resolve their argument at PARSE time, so they read a file from
disk without any filesystem command being registered anywhere. They are part of
the language core and removing them means removing the language.

This is the concrete form of the_user's argument for sandboxing at all: nushell
has more surface than a command allowlist reaches, so the allowlist is not where
the line gets held. It is held by a read-only root.

## struct QuestnessEvaluator

## fn new

Built once and cloned per evaluation, which is the standard embedding model and
is not merely an optimisation. `EngineState` is `Clone` with its heavy fields
behind `Arc`, so a clone shares rather than copies; each evaluation needs a
mutable engine of its own to merge its parse delta into, and the clone drops
afterwards, reclaiming what it added. A single long-lived engine would instead
grow without bound, because merging a delta is monotonic and every distinct
snippet extends the files, spans, variables and declarations for the process's
life.

Be precise about what a clone does NOT give you: the jobs table, the regex cache
and the signals flag are interior-mutable and shared through the clone. That is
why each evaluation installs a fresh signals flag rather than inheriting one -
sharing it would make cancelling any evaluation cancel all of them, and leave
the flag set for every evaluation afterwards.

`is_mcp` is set and its name is misleading: it has nothing to do with any wire
protocol and means that this host's standard streams are a protocol channel. It
gives an external a null stdin instead of ours and routes `print` to stderr, so
a body cannot inject text into the channel. A nushell plugin's stdout is exactly
that kind of stream.

Ansi colouring is turned off through `set_config` rather than by assigning the
field, because the function also propagates plugin garbage-collection settings
and that propagation is private.

## fn lock_boundary

The boundary this engine claims, asserted where it is built rather than
only in the test suite.

It moved here because the claim was previously only true of the tests. A
chapter recorded that the code "carries a lock asserting that none of
those names resolve, which fails loudly the day someone adds the shell
context" - and it did not: the assertion lived in `src/tests`, so a
shipped crate would have widened its sandbox silently and only a test run
would have said so. Constructing the thing is when the guarantee is worth
checking, because that is when a caller starts relying on it.

The `PARSE_TIME_LOADERS` half is an INVERSE lock and reads oddly until
you see what it is for. It requires those names to STILL RESOLVE, which
is the opposite of a security check. The reason is that their reach is
what justifies containment sitting at the service level rather than in
registration: `use` and `overlay use` resolve their argument at parse
time and read files regardless of what is registered, so no allowlist
closes them. If a nushell release ever removed that reach, the argument
for the bubblewrap wrap would have quietly lost one of its legs, and this
fires rather than letting the note go stale.

It costs one `decl_names` walk per name at construction, which is a few
hundred string comparisons once per engine. `new` already does far more.

## fn decl_names

## fn resolves

## fn evaluate

The thread is not about parallelism. It is the sized stack from
`PARSE_STACK_BYTES`, and the `catch_unwind` inside it converts a panic in a
builtin into a failed call rather than a dead process. That only works while the
build profile unwinds; an abort profile makes it moot.

## fn eval_once

THE MERGE STEP IS NOT OPTIONAL AND ITS ABSENCE IS CONFUSING. Parsing registers
the blocks, closures, variables and declarations the source introduced onto the
WORKING SET rather than onto the engine, so evaluating without merging produces
command-not-found for names that are plainly in the source.

THE ORDER AROUND IT MATTERS TOO. Parse errors and compile errors are read BEFORE
the merge, because merging the output of a failed parse pollutes the engine with
half-formed definitions - and because `parse` returns a block whether or not it
succeeded, so nothing about its return value tells you to stop.

Errors render through `format_cli_error`, which takes any `miette::Diagnostic`
directly. Wrapping a parse error in a shell error first, which older example
code does, is wasted work that also discards the diagnostic code. Rendering
through `Display` rather than `Debug` is the difference between a message and a
structure dump.

`capture_all` puts both pipe destinations at the value sink so the pipeline's
own result is what comes back. A bare stack would leave them inheriting the
host's real file descriptors, which for a protocol-carrying stdout is a
corruption rather than an inconvenience.

## fn register_filters

The twenty-one filters are the verified starting set rather than the intended
end state. Widening it is a deliberate step with its own locks, and the two
families most obviously owed are `math` and the `from`/`to` format pair, since
an evaluate sub-turn doing arithmetic on a table wants both. Arithmetic itself
already works, because operators are language-level rather than commands.

One consequence of hand-building rather than layering the shell context, worth
knowing because it fails silently: `add_shell_command_context` caches the
`table` declaration id on the engine, and table rendering consults exactly that
cached field. Without it, rendering degrades to unformatted output rather than
raising. It costs us nothing because we render NUON, but a reader wondering why
a table prints as debug text should look here first.
