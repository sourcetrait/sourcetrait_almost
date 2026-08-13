# speculate.rs

The speculation depth probe: record plain greedy transcripts, then replay them
offline through the real lookup index under candidate policies and pass-cost
models.

The replay drives the shipped types rather than a copy of them, which is why the
engine's index and policy are part of its public face. A simulator over
reimplemented policy logic would price a policy that does not exist.

Replay fidelity is the whole value, and the round loop mirrors the generation
step exactly: the emitted token joins the index before drafting, queue-popped
tokens owe no round, the bonus rides as pending, a budget-exhausted end skips the
final round while a stop-token end pays it, and accepted drafts join the index at
round close.

One conservative bias is known and left in. A draft that would accept the stop
token itself scores as a rejection here, because stops are never emitted into the
recorded stream, so at most one round per stop-ended turn reads one pass heavy.

The simulator's own limit, measured afterwards: its direction predictions all
landed and it undershoots on magnitude, because a verify chunk is heavier than a
fused single-token step and the host probe and index work never appear in a pass
count. Treat tokens-per-pass as a ceiling rather than an estimate.

Everything data-shaped is whole-value NUON: the fixture plan in, the per-transcript
artifacts out, the report out. Phase annotations ride sibling hand-authored files
joined at simulate time, which keeps a machine-written artifact machine-written.

## const FIXTURE_TYPEDEF

## const TRANSCRIPT_TYPEDEF

## const PHASES_TYPEDEF

The phases form uses an exclusive end with -1 meaning the turn's end, and its
indices index the emitted ids rather than characters. It is hand-authored, so the
sentinel is there to make a whole-turn annotation writable without counting
tokens.

## const RECOVERY_PROBE

The gated policies only update their accept-depth estimate on rounds they actually
drafted, so a gate that closes can never observe recovery. This re-probes every so
many suppressions, which is what stops a cold gate from being permanent.

## const EMA_ALPHA

## struct TranscriptPlan

## fn read_plans

## fn speculate_record

## fn record_transcript

Recording is plain greedy, which is sound rather than a simplification: the depth
statistics are greedy-deterministic, so a sampled recording would measure the
sampler as much as the index.

Later turns chain live through the continuation surface over a constructed handle
rather than through snapshot files, so the recorded trail is the restored-context
contract being exercised rather than a reconstruction of it.

The suffix ids are recovered by slicing the report's context trail at the prompt
token count, which is what separates a turn's rendered prompt from the previous
turns' accumulated context.

## struct RecordedTurn

## struct RecordedTranscript

## fn read_transcript

## struct PhaseRow

## fn read_phases

## fn default_phase

A prose-family kind defaults every token to the prose phase, and anything else is
unlabeled until its phases file lands. That default is what makes the prose floor -
the constraint that speculation must never cost a pure-prose user - measurable
before any annotation exists.

## fn phase_label

## struct CostModel

## const COST_MODELS

The two pass-cost models are the whole reason the simulator exists rather than a
single measurement. Under the two-pass model a partial acceptance costs a verify
plus a shadow replay; under a kernel-managed accept it costs one. Every policy
failed the prose floor under the first and cleared it under the second, and plain
v1 beat every gated variant - which is why no gate was built.

## enum PolicyKind

## struct PolicyState

## fn new

The gates start hot, at a depth that keeps them open, so a policy is judged by what
it observes rather than by having to earn its way out of a cold start.

## fn name

## fn round_draft

Mirrors the real derivation - probe limit, index draft, draft-limit truncation - and
applies the candidate gate after the match rather than before. That ordering matters
for the level-gated variants, which need the matched level to pick their gate.

## fn record

## fn candidate_policies

## struct PhaseTally

## fn replay_transcript

The index is rebuilt per turn and seeded like the real generation start: the
committed trail plus that turn's rendered suffix. Carrying one index across turns
would let a later turn draft from text the real run had not yet committed.

A plain greedy decode reads exactly one pass per token, so tokens-per-pass is the
against-plain ratio directly and needs no baseline run.

The budget check precedes the round rather than following it, which is what
reproduces the asymmetry between the two ending kinds.

The debug assertion that the replayed token matches the recorded one is the guard
that keeps the loop honest: any divergence in the round mechanics shows up there
rather than as a plausible-looking pass count.

## fn speculate_simulate

The aggregate key joins the policy and cost names with a unit separator rather than
a printable character, so a policy name containing the separator cannot collide.
The key order is tracked separately, because the aggregate map does not preserve it
and the report's row order should be stable.

## fn speculate_tokens

The phase-annotation aid: emitted tokens as indexed decoded pieces, one line each.
It exists because annotating a transcript by hand means finding token boundaries in
text, and the decoded pieces are the only honest view of those.
