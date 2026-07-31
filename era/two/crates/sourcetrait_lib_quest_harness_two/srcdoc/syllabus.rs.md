# syllabus.rs

Training assets live as data rather than as constants compiled into a
binary, and this is what reads them. The tree's grammar is the design's:
template -> set[] -> accept[][]. A template IS a question; one set file per
template carries `[teach, data]` rows whose data is the runtime call record
minus prompt (`{config, input}`); one accept file per template is the
answer key, `[teach, accepted]` rows with `[[answer, typedef]; ...]`
values. Rendering a method is discovery, validation, the teach join and
templating, in that order.

The boundary is deliberate and it stops short of stage semantics. What a
stage does with a rendered case - assemble a supervised sequence, pair a
rejected side, attach a verifier - differs per stage and lives in the
method tools, so nothing here presumes it. This layer answers what was
asked, what data rode which channel, and what answers the key accepts.

## const STAGES

The walk iterates this named set rather than reading whatever directories
the root happens to hold, and that is load-bearing rather than tidy. The
mix repository's root carries a corpus beside its stages, so a walk over
the root's own directories would descend thousands of corpus files looking
for template directories. Naming the stages also fixes the order as the
order they are trained in, which alphabetical order would scramble.

A stage directory that is not there is skipped rather than raised, since a
tree carrying only the stage being worked on is an ordinary state.

## const TEMPLATE_DIR

A method is identified by holding this directory, which is what lets the
syllabus above it be any depth at all without a naming convention to
parse. The alternative - fixing the depth, or marking a method with a
sentinel file - would either forbid the nesting the design asks for or add
a file whose only job is to say what the structure already says.

## const SET_DIR

## const ACCEPT_DIR

## const TEMPLATE_EXT

## const TYPEDEF_EXT

## const DATA_EXT

## const FILL_BINDING

Uniform whatever the syllabus, which is what makes the slot contract the
whole of what a template author has to read. Nothing about where a method
sits in the tree leaks into how its template addresses its data.

The sigil is absent because Liquid rejects one in a variable path, and the
template module already resolves that by keeping the channel's name and
dropping the `$`. The same key is what the runtime binds when a caller
sends `{liquid: {train: {...}}}`, so the walker and `config::prepare`
speak one contract.

## struct MethodPath

### fn dir

### fn syllabus_path

## struct Accepted

The typedef travels as a string because that is how the harness holds a
type everywhere (the REV 127 carriage ruling: an answer that IS a typedef
is a string). It is parsed and enforced on the way in, so a reader
downstream can trust it without re-checking.

## struct RenderedCase

The config and input travel VERBATIM, nulls included, because a case is
the runtime call record minus prompt and the packer downstream frames a
declared arm from the row's own config. Interpreting them here - say,
stripping the liquid key after the render - would leave the consumers a
record that no longer matches what the runtime would have received.

### fn to_value

## struct SetRow

## fn methods

The walk stops at a method rather than descending into it, so the template
directory inside one is never mistaken for a nested method. That is the
one thing the identification rule needs to get right, because the rule is
otherwise depth-blind by design. A method's own `accept`, `tool` and
`judge` directories are never walked for the same reason.

## fn render

One set file and one accept file pair with each template by its stem, and
every fault is refused rather than skipped: a skip would let a drifted row
vanish from a generated set silently, which is the failure the gate exists
to catch. The old walk crossed every template with every case file; the
realignment retired that - a case belongs to exactly one question, and the
cross product is what fused questions to data in the first place.

Rows keep their authored file order while templates walk sorted. The set
file's order is the spread its author chose, and two renders of one tree
are byte-identical either way.

## fn emit

Writing and committing are one act rather than two. The design's flow is
that a run's provenance going into a railroad is that railroad's first
change, so a caller that could write without committing could leave one
standing uncommitted, which is precisely the state the railroad exists to
make impossible.

Nothing rendered is written. A prompt is a pure function of the template,
the row and the renderer, so the source commit reproduces every one of
them exactly, and storing the text as well would double the artifact to
say the same thing twice. The render still runs here, because running it
is what proves every row validates and every template fills - an emission
fails on a tree that could not be generated from, rather than later, when
someone tries to assemble it.

The method is the unit, so there is no per-stage aggregate. Stage order
survives only in the walk, which is what keeps `drawn` reproducible.

Provenance records what the design named and stops there: the method paths
drawn from with their case counts, the scheme version, and where the
source tree stood. There is no seed, because rendering is deterministic
and inventing a field to look complete would be worse than its absence.

`source_dirty` neither warns nor refuses. It states whether the commit
beside it is a complete description of what was drawn - a fact the reader
of a finished run needs and one only this moment can establish. An
unversioned tree is refused outright, because there the question cannot be
answered at all.

The source root is deliberately not recorded. It would be an absolute path
on one machine baked into an artifact meant to be readable anywhere, and
the method paths already say what was drawn from.

## struct Emitted

Scalars rather than a per-method breakdown, because the breakdown is in
the artifact the call just wrote and a caller reading it back has better
information than a summary could carry.

## fn contract_of

Presence IS the declaration. A `.nutype` beside a template says the
template carries slots and its rows ride the infill channel; its absence
says the rows pipe. Nothing scans the template text for `{{`, because the
contract file is the authored statement and a scan would be a second
opinion that could disagree with it.

## fn prompt_of

The channel discipline enforced both ways: an infill row against a
slotless template is refused, and a piped row against a slotted one is
refused rather than left for the renderer's unknown-variable error -
the strict render would catch it, but naming the slot contract is the
message a tree author can act on.

The liquid record must bind `train` alone. The runtime binds any keys a
caller sends; the WALKER gates training data, and a binding the tree's
contract does not name is drift, not flexibility.

The rendered prompt's trailing newline is trimmed because it is an
artifact of the template being a file, not part of the question - the
same trim the packer applies, so the two produce identical prompts.

## fn liquid_of

## fn set_rows

Key-exactness on every row (`teach`/`data`, then `config`/`input`) is the
walker's answer to records being open: nu conformance would wave a stray
key through, and a stray key in training data is a misspelling about to
train as silence.

## fn accept_rows

Every accepted answer is gated against its OWN typedef at load, so a
declared type an answer does not have cannot reach a method tool, a
grader, or a pack. Plurality is per teach - genuinely different acceptable
values - and an empty answer list is refused because an unanswerable case
is a set row that should not exist.

## fn table_rows

## fn exact_keys

## fn string_field

## fn walk

## fn sorted_dirs

## fn sorted_files

Both sorts exist so a generated set is reproducible. Directory iteration
order is the filesystem's, so an unsorted walk would emit the same cases
in a different order on a different machine and make two sets diff-noisy
against each other for no reason.

## fn stem

## fn dir_name
