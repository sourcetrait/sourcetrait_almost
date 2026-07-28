# syllabus.rs

Training assets live as data rather than as constants compiled into a
binary, and this is what reads them. A method is a template plus the
contract its fills must satisfy plus the cases that fill it, and rendering
one is discovery, validation and templating in that order.

The boundary is deliberate and it stops short of stage semantics. What a
stage does with a rendered case - pair it with a gold answer, with a
rejected side, with a verifier - differs per stage and is not settled, so
nothing here presumes it. This layer answers what was rendered and what it
was rendered from.

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

## const CASE_DIR

## const TEMPLATE_EXT

## const TYPEDEF_EXT

## const CASE_EXT

## const FILL_BINDING

Uniform whatever the syllabus, which is what makes the contract file the
whole of what a template author has to read. Nothing about where a method
sits in the tree leaks into how its template addresses its data.

The sigil is absent because Liquid rejects one in a variable path, and the
template module already resolves that by keeping the channel's name and
dropping the `$`.

## struct MethodPath

### fn dir

### fn syllabus_path

## struct RenderedCase

### fn to_value

The fill travels with the rendered text so a generated set's provenance
can name what each case was rendered from rather than only that it was.
Set-level provenance - the scheme version, the seed - belongs to whoever
writes the set, since neither is a property of one case.

## fn methods

The walk stops at a method rather than descending into it, so the template
directory inside one is never mistaken for a nested method. That is the
one thing the identification rule needs to get right, because the rule is
otherwise depth-blind by design.

## fn render

Every template renders every case, and a case that does not fit a
template's contract is refused rather than skipped. Refusing is what makes
the contract a gate at all: a skip would let a drifted fill vanish from a
generated set silently, which is the failure the contract exists to catch.

That choice is settled only for the one-template method the scaffold
demonstrates. A method carrying several templates whose contracts differ
would demand every case fit every one of them, which is probably not what
a multi-template method would mean - so it is raised as a question rather
than resolved by picking the lenient behaviour and hoping.

Two facts about the contract, both measured rather than assumed. A
multi-line contract parses, because the typedef is spliced into a closure
signature and nushell accepts newlines inside one. And the sugar collapses
on the way through, so a declared `path` conforms against a NUON string
rather than against anything path-shaped - which is why the scaffold's own
fills pass rather than failing on their own declared type.

An orphaned contract is refused alongside an uncontracted template, and
that pairing is the point: a rename that landed on only one of the two
leaves exactly one of these behind, and catching either catches the rename.

## struct Emitted

Scalars rather than a per-stage breakdown, because the breakdown is in the
artifact the call just wrote and a caller reading it back has better
information than a summary could carry.

## fn emit

Writing and committing are one act rather than two. The design's flow is
that a run's provenance going into a railroad is that railroad's first
change, so a caller that could write without committing could leave one
standing uncommitted, which is precisely the state the railroad exists to
make impossible.

Nothing rendered is written. A prompt is a pure function of the template,
the fill and the renderer, so the source commit reproduces every one of
them exactly, and storing the text as well would double the artifact to
say the same thing twice. The render still runs here, because running it
is what proves every fill conforms and every template fills - an emission
fails on a tree that could not be generated from, rather than later, when
someone tries to answer it.

What the railroad carries is the answers, laid under each method's own
`accept` by the tooling that generates them. That is why nothing here
creates those directories: git does not carry an empty one, so a mirrored
skeleton committed ahead of the answers would be a commit of nothing.

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

## fn walk

## fn sorted_dirs

## fn sorted_files

Both sorts exist so a generated set is reproducible. Directory iteration
order is the filesystem's, so an unsorted walk would emit the same cases in
a different order on a different machine and make two sets diff-noisy
against each other for no reason.

## fn stem

## fn dir_name
