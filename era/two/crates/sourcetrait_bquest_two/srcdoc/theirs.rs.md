# theirs.rs

The instruct-stage counterweight's conversion half: extracted pool chat rows
become quest-native supervised example rows, one class per invocation, so the
composition stays a per-class decision made in configuration rather than a
proportion baked into code. The extraction (parquet to the pool-rows table) is
reference-boundary tooling on the nu side; this module starts where our own
formats do.

Everything structural rides the harness codec - the wrap renders authoring
aliases the pack's `authoring_to_wire` translates, and every wrapped turn is
parsed back through `parse_blocks` before it ships - so a converted row cannot
be out of grammar without the converter refusing to produce it
(validate-equals-serve applied to the counterweight).

## const POOL_ROWS_TYPEDEF

## const REFUSED_MARKERS

The non-pipe added-token family is REFUSED rather than escaped, deliberately
asymmetric with the `<|` family. MarkerEscape breaks a `<|` spelling's
tokenization, so escaping it is safe; the `<functions>` family has no
display-honest break, and rows discussing those markers would otherwise teach
the trained tool-marker attractor mid-prose - the exact conduct class the
counterweight exists to counterweight. Rows carrying them are rare and better
absent than taught.

## const ENGLISH_HINTS

## const ENGLISH_FLOOR

A single-list positive filter rather than a language classifier: English
function words cover a quarter to a third of ordinary English tokens and near
zero of any other language's, so requiring a 0.12 share rejects es/de/fr/ru
rows without naming them. The list avoids spellings other languages share
(no `a`, no `no`, no `me`, no `de`). Crude by design and surfaced as such -
the filter's job is oasst1's language split, not language identification.

## struct TheirsTally

The refusal split is the report's value: a selection review reads WHY rows
fell (markers vs language vs code vs no-assistant vs window) rather than one
opaque skip count, and a class whose refusal profile looks wrong is a
selection finding, not noise.

## struct TheirsOptions

## struct PoolRow

## enum Refusal

## struct Converted

## fn reads_english

## fn wrap_md

The wrap is the no-implied-format ruling landing on the counterweight: every
assistant prose final says `md`. Content is trimmed before wrapping because
leading and trailing blank lines would ride inside the block as payload
structure the source never meant.

## fn codec_gate

Raises rather than counting, against the grain of every other check here. A
refusal says the ROW was unfit; a gate failure says the CONVERTER built an
out-of-grammar turn from fit content - a defect to stop on, since content is
already escaped by the time the wrap happens and no legal input can fail the
parse. Counting it would let a converter bug ship as a slightly smaller part.

## fn convert_messages

The system strip is the "system prompts to the quest posture" conversion in
its literal form: the pool's system turns are dropped so the pack's encode
path prepends QUEST_SYSTEM alone. Keeping a foreign system turn would train
those rows OUTSIDE the quest conditional - the one place the counterweight
must land, since the whole training gradient conditions on that turn.

The fence filter is opt-in per class because it is a subject proxy, not a
grammar rule: the substance-honesty ruling excludes computer-language-subject
rows from the prose selection, and a ``` fence anywhere in the exchange is
the mechanical tell for mixed classes (wildchat carries coding asks beside
its creative and discussion prose). A homogeneous prose class runs without
it; forcing it globally would evict legitimate prose that quotes a fence.

Escaping runs on user content too, not only the wrapped finals: any `<|` in
any turn would tokenize to a real marker at pack time regardless of which
role carries it.

## fn wire_messages

## fn read_pool_rows

## fn example_row

`id` and `source` ride beside `messages` because records are open: the pack's
loader conforms against the messages-only typedef and ignores the extras, so
provenance travels IN the data - the cherry-picked-foreign-data rule - without
a sidecar join.

## fn convert_rows

The draw is a seeded Fisher-Yates over the pool, then a budget walk in
supervised tokens with the crossing row overshooting - the same conventions
as `mix sample`, so a reader of one sampler has read them all. The budget is
in SUPERVISED tokens, measured through the same `encode_supervised` the pack
uses, because the 1:1 admixture ratio is ruled in supervised tokens and any
other measure here would make the ratio a different number at pack time.

The window check happens here as well as at pack time so the budget never
counts a row the pack would drop: a skipped-at-pack row would silently
shrink the counterweight below its ratio.

## fn mix_theirs
