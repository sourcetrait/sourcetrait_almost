# taskgen.rs

The task generators: synthesised, mechanically-verifiable examples for every
training stage plus the bench, from one source.

One generator emits the supervised examples, the preference pairs, the
reinforcement prompts and the bench, split at the source by index. Building the
bench separately would let the two drift, and it would stop measuring what the
training set teaches.

That single source is also the design error attempt two measured. Three stages
pointed at one item set are one dataset wearing three costumes, and each stage
then ran the narrowest possible version of itself. The ruling that follows from it
is that each post-training stage wants its own method, objective and content - so
this file is the shape a redesign changes, not a thing to extend.

Every gold is verified before it ships, against the verifier that will grade the
model on it. A generator whose own answers do not pass its own grader is the one
failure that would poison every stage at once, silently, so it is checked per item
rather than reasoned about. Every rejected answer is checked to be wrong for the
same reason in reverse: a mutation that happened to remain correct would teach the
model away from a right answer.

The rejected side is never invented. The NUON mutations reproduce failure modes
the channel probe measured on this checkpoint - dropping a record's braces for
bare key-value lines, and wrapping a table in a record. The formatting families'
wrong answer is the other style, which is a correct value in the wrong shape. The
nushell families' is the gold with a command head misspelled, which is the
recorded implicit-external trap.

## const FAMILY_CONVERT_NUON

## const FAMILY_FORMAT_NUON

## const FAMILY_NU_FROM_BASH

## const FAMILY_NU_FROM_PROSE

## const FAMILY_NU_ERROR

## const CONVERT_INSTRUCTION

## const PRETTY_INSTRUCTION

## const CONDENSED_INSTRUCTION

## const BASH_INSTRUCTION

## const PROSE_INSTRUCTION

## const ERROR_INSTRUCTION

Each instruction ends by naming the reply's form, because the verifier compares
against exactly that. An instruction that left the form open would make a correct
answer gradeable as wrong.

## const FIELD_NAMES

## const WORDS

Snake field names because that is what our own data uses, and a small vocabulary
because the task is the transform rather than the content. A wide vocabulary would
make the items look more varied while teaching nothing extra.

## enum Shape

## const SHAPES

Ordered so that a small run still exercises every shape, since the generator walks
them cyclically by index rather than drawing at random.

## struct NuTemplate

## const NU_TEMPLATES

Templates rather than fixed items, so a handful of templates yields a corpus while
every variant stays deterministic and self-contained. The bash and prose forms
sit beside the nushell answer in one row, which is what keeps two families in
agreement about what the answer is.

## const MUTABLE_HEADS

Each pair misspells a real command head, so the break is the recorded
implicit-external trap: an undefined name is not a parse error in nushell, it
becomes an external and fails at runtime. A syntax break would be a different and
easier lesson.

## fn word

## fn field

## fn scalar

Stays inside what JSON can carry, so nothing is lost on the way into the
conversion family's prompt and the gold is a faithful answer to the question
asked.

## fn record_of

Distinct fields, because a repeat draw would collide and a record's keys are
unique. The attempt bound stops the loop rather than looping forever on a small
vocabulary, which means a requested width may not be reached - acceptable, since
the shape is what matters.

## fn table_of

Uniform columns in every row, which is what makes it a table rather than a list of
records. That distinction is the point of the family: the modern table literal is
the emission the channel probe measured at zero out of ten.

## fn synth

## struct Item

## fn checked

## fn prompt_json

## fn without_braces

## fn wrapped_table

## fn nuon_families

The formatting families are graded on bytes, because for a formatting task the
bytes are the answer. That is also why their rejected side is the other style:
the same value, differing only in whitespace.

The consequence for preference tuning is measured rather than theoretical. The
preference loss rides an un-normalized sequence log-probability over the whole
masked span, so a pair differing only in whitespace raises and lowers largely the
same tokens in opposite directions. That property generalises past formatting,
because every family's rejected answer is a near-copy of its gold by construction -
a plausible wrong answer has to be close.

## fn fill

## fn misspell

## fn reported_error_line

The answer comes from nushell's own diagnostic rather than from where the mutation
was made, and those differ. An unclosed brace opened on one line is reported where
the parser reaches end of input, which is a later line. Asserting the mutation site
would have shipped a wrong answer for that whole class - found by probing nushell
before writing the family rather than after.

## fn nushell_families

The reference is the value the gold produces, computed by running it, so the
grader compares values and never the rendering.

A candidate whose pipeline does not run cleanly is skipped and counted rather than
shipped. A mutation that did not break the snippet, or that broke it without
nushell saying where, has no answer to teach. A timeout is called out separately,
because it means the snippet hangs rather than fails, which is a defect in the
template rather than in the mutation.

## fn messages_value

## fn write_table

## fn taskgen_all

The bench splits at the source by index, so re-generating with a different seed
invalidates comparison against earlier bench readings. Pin the seed per campaign.

Items without an available wrong answer are counted rather than dropped from the
other stages, so the preference table is legitimately smaller than the supervised
one and the difference is visible.
