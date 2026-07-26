# score.rs

The pure-Rust scorer: a converted run in, per-item scores and per-task aggregates
out, on the CPU with no python at runtime.

It is a fork of olmo-eval's scorers, and the fork constraint is what makes the
reference reading worth keeping: the per-item layer must stay identical, and only
aggregation may differ. Change per-item scoring and the reference stops being
comparable, which is the entire reason to keep it.

That constraint is gate-proven rather than asserted. The scorers reproduce the
reference's per-item and aggregate results at exact float bits across the standing
estate, so a care-weighted re-aggregation is a new layer over a per-item layer that
is already validated equal.

Every aggregate is an exact count ratio, so no float-summation-order hazard exists
anywhere in the file.

Scoring is self-contained from the converted artifacts, because the fixture
documents embed the gold index and choices, the short answer, and the instruction
lists with their arguments. There is no dataset access, which is why a full run
scores in about a second where the reference legs ran for minutes.

## static COMMA_IN_NUMBER

## static NUMBER

## fn extract_last_number

## fn clean_short_answer

The grade-school maths extraction, and the order is upstream's: collapse commas
inside numbers first, then take the last number under a leftmost-first
alternation. The alternation behaves identically in the Rust regex crate, which was
checked rather than assumed - the two engines differ on alternation preference in
general.

## enum TaskKind

## fn resolve_task_kind

Routing mirrors the reference registry, and an unregistered task raises rather than
falling back. A default would mis-score silently, which is the one failure this
whole file is built to avoid.

## struct InstructionVerdict

## struct ScoreRow

The per-instruction verdicts are retained rather than collapsed, and the addition is
strictly additive on purpose. The parity gate compares `scores` by key set and
`extracted` by length, so a new sibling field is the one shape that cannot disturb
it - verified rather than assumed, with the gate reading zero mismatches across the
whole estate with the field in place.

They are retained because a care-weighted re-aggregation needs per-type results to
attach to, and the collapsed per-item score offers nothing to weight.

## struct TaskScores

Metrics are an ordered list rather than a map, in the reference's own
`config.metrics` order, because the order is part of what a comparison checks.

## fn field_f64

Accepts an integer as well as a float, since a metric that happens to be whole
arrives as an integer through the JSON bridge and rejecting it would be an artefact
of the transport.

## fn output_text

## fn first_max

Python's `max` followed by `list.index` takes the first maximum, so ties break
toward the lower index. Rust's own `max_by` takes the last, which would flip every
tied multiple-choice item.

## fn score_mc

The per-item score is the maximum summed logit and the accuracy counts a
first-maximum argmax against the gold index, with every row in the denominator.

The extraction differs by task and that is upstream's: mmlu extracts nothing while
arc extracts the continuation text.

## fn score_gsm8k

## fn kwargs_from_record

Null kwargs are skipped rather than carried as a null, because the reference strips
them before dispatch and a checker reading one would see a present-but-empty
argument.

The integer and float split is preserved rather than normalised to one numeric
type, because it is semantic downstream: an index must be an integer while a
threshold may be either.

## fn score_ifeval_task

The per-item score is the conjunction of the loose results, and the four reported
ratios are prompt-level and instruction-level crossed with strict and loose, in
configuration order.

An empty instruction list scores zero rather than one, which is the reference's
behaviour and the opposite of what a vacuous conjunction would give.

## fn score_task

Rows must arrive doc-id-sorted and the doc id must equal the position, enforced
exactly as the reference rescore does. That is a real alignment guard rather than a
formality: fixtures and predictions are separate files, and a misalignment would
score every item against the wrong gold while producing a plausible number.

## fn load_sorted_rows

## fn rows_as_records

## fn find_fixture_file

An ambiguous match raises rather than picking one. Two fixtures for one task token
means two renders are present, and choosing silently would make the reading depend
on directory order.

## fn score_row_value

The instruction list is present only where a task carries instructions, so the
multiple-choice and exact-match rows stay exactly as they were. Records are open, so
a consumer tests presence rather than needing a null.

## fn aggregate_value

## fn capability_score

Loads the checker data lazily, on the first task that needs it, because it is the
only expensive load here and most tasks never touch it.
