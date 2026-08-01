# train.rs

Manual per-layer checkpointing is the reason this file is shaped the way it is.
Full-graph autodiff over a 7B measured about 7.5 GiB of retained forward on
Tier A, which overflows the card once the weights are resident, so a no-grad
pass caches each layer's input and the loss block and each layer then run as
their own small graphs chained by vector-Jacobian products. At most one layer's
graph is live at a time.

Burn autodiff over the sequential GDN recurrence is the correctness grade
rather than the fast one. A chunked backward stays a later performance lever.

The objective is a closure over the top hidden state, deliberately, so adding
an objective means adding a loss closure and never touching the chain.

## type FloatTensor

## type OptState

## struct LoopOptions

## struct StepLog

`step` is the loop index plus one, and for `rlvr_loop` the loop index is a
group index. A missing step number in a reinforcement step log is therefore a
skipped uniform group, not a lost or failed step.

## struct TrainReport

## struct StepOutcome

## fn lift_block

`from_inner` yields an untracked constant without copying, which is burn API
semantics rather than anything visible here. Lifting a block therefore costs no
memory and contributes no gradient; only the adapters do.

## fn adapters_inner

The caching pass must not build a graph over the live adapter tensors, which is
the only reason a frozen view is needed at all.

### fn pair_inner

### fn conv_inner

## fn dims_of

## struct HybridDims

## fn dims_block_forward

The invariant the `unreachable!` arm rests on is established in
`ModelAdapters::init`: it builds each layer's adapter kind from the same
`layer_types` the model built its blocks from. Reaching that arm means the
adapter set and the model disagree about the architecture, which is a
construction bug rather than a runtime condition.

## fn cross_entropy_chunked

Chunking bounds one materialization, not the retained total. Every chunk's
logits and log-probs stay live inside the autodiff graph until backward, so
lowering `loss_chunk` makes more chunks rather than less peak memory. That is
the recorded misdiagnosis on this axis, and it applies equally to
`objective::masked_logprob_sum`.

## fn forward_cache

The trade this method makes: one cached hidden-state tensor per layer, against
retaining all 32 layers of autodiff graph.

## fn seed_from_hidden

The chain beneath this never learns what it is training, which is what lets
continued pretraining, supervised, preference and reinforcement share one
trainer.

The head delta is the one adapter whose gradients come out of THIS backward
rather than out of the chain: it lives inside the loss block, so its grads are
extracted here and merged into the accumulated map beside the chain's. The
reference paths pass no delta, which is what keeps the frozen-base
log-probabilities frozen.

## fn head_delta_grads

## type PairedSeedGrads

## fn seed_pair_from_hidden

Both candidates' layer-input caches are live simultaneously, because both
chains run after the paired loss block and the loss spans both sequences in one
graph. That is why the preference stage's peak is roughly twice the supervised
stage's at the same window.

## fn chain_from_seed

## fn accumulate_grads

## fn scale_grads

`rlvr_loop` divides by the group's full rollout count rather than by the number
that actually contributed, so zero-advantage members scale as though they had.
That is deliberate - the group mean is the intended quantity - and not an
off-by-one.

## fn chain_step

## fn full_graph_step

## fn train_loop

Weight decay is explicitly zeroed because adapter tensors are the only
trainable state and decaying them would fight the frozen base.

Two environment knobs ride step zero, both for diagnosing rather than training.
`QUEST_TRAIN_DEBUG_GRADS=1` is the NaN localizer: it names the layer a
non-finite backward came from. `QUEST_TRAIN_PROBE=grads|step` is the memory
bisection tool, parking after the first gradient set or the first optimizer
step so an external poller can read the VRAM level.

## fn warmup_rate

The step argument is treated as one-based, so step zero already carries
`peak / warmup_steps` rather than zero.

Every loop shares this, which is what keeps the schedule from drifting between
stages. It is also the site of a defect - see `## fn rlvr_loop`.

## fn optimizer_step

A missing gradient for a named tensor raises rather than skipping the update,
because silently skipping one would train a subset of the adapter without
saying so.

AdamW normalizes the update by the second-moment estimate, so a uniform
rescaling of the gradient largely cancels in the step size. That is why the
un-normalized sum in `objective::sequence_logprob` does not make a given
nominal learning rate harsher for the preference and reinforcement stages than
for the supervised one, and it is the reason not to infer a rate from loss
magnitudes across stages.

## fn debug_grad_health

How to read it: a layer showing non-finite entries is where the backward broke,
and a layer whose summed magnitude sits far above its neighbours is where it is
about to.

## fn toy_config

Two layers is the minimum that exercises both block kinds, and therefore both
adapter kinds.

## fn toy_tensor

## fn toy_norm

The era-one gate hit this as a fixture trap: a small norm weight multiplies
every activation down and flattens the loss landscape, so a gate that cannot
descend reads as a broken gradient path when the fixture is what is wrong.

## fn toy_weights

The tensor names are the checkpoint's real safetensors names, so the toy model
exercises the same loader paths as a real one.

## struct GradCompare

All-zero reference gradients are matched rather than scored because a fresh
low-rank pair's `b` factor starts at zero, which makes the `a` gradients
exactly zero until `b` moves. Scoring those as nmse would divide by zero.

## fn compare_grads

## fn tensor_values

## struct GateVerdict

## const GATE_GRAD_NMSE_BAR

An f32 reassociation bar, not an equality bar. The two graph shapes compute the
same mathematics in a different order, so they agree to reassociation error and
not to the bit.

## const GATE_DESCENT_MARGIN

Sixty steps over four repeating chunks memorizes them, so a working gradient
path clears this comfortably and a broken one cannot cross it by noise.

## fn run_train_gate

The gradient equivalence is checked twice, at init and after five real
optimizer steps, because the init case alone would pass a chain that only
handles zero adapters - `b` is still zero there.

The losses are asserted equal before the gradients are compared, since a
gradient comparison over two different losses would mean nothing.

## fn forward_plain

## fn reference_logprob

## fn split_row

The mask is sliced `[1..]` alongside the targets, which is what aligns a
per-position mask written over the full row with the shifted targets. Getting
that slice wrong shifts the supervised span by one token.

## struct SupervisedBatch

## struct RowTokenLosses

Positions are full-row token coordinates - target index plus one - so a
dump row joins the pack token table on (row, position) with no
off-by-one anywhere downstream. The position IS the token whose
prediction the loss scored.

## struct SftReport

## fn sft_loop

A row supervising no position raises rather than stepping, because a zero-loss
step would look clean while training nothing. Packing should have dropped it
already.

The per-pass reshuffle exists because a fixed order at batch 1 replays the
identical neighbor sequence every pass, so momentum wake and interference
compound on the same rows instead of averaging out; measured as tail-row
wobble on a 170-row set before it landed. The permutation is Fisher-Yates
over SplitMix64 seeded by a constant xor the epoch, so a run is reproducible
and two passes never share an order. The other loops keep their fixed order:
cpt chunks are already shuffled at pack time, and the preference and
reinforcement loops index pairs and groups whose internal order carries
meaning.

Resuming a settled adapter at the peak rate destabilizes it before it
re-settles, measured live: a two-pass continuation of a floored twelve-pass
run lifted 47 of 170 rows back over the 0.01 bar with a worst of 0.39.
Fresh AdamW moments make early updates near rate-sized regardless of
gradient magnitude - the cold-moments condition the rlvr warmup note names,
arriving through resume. A continuation therefore wants a settle-class rate
or a fresh run; there is no cheap warm-rate top-up.

The token capture overwrites per visit, so each slot ends holding its
row's LAST visit whatever the epoch structure. A cheaper-looking
"capture only the final rows.len() visits" window is wrong: with per-pass
reshuffles a run ending mid-epoch leaves rows whose last visit predates
the window. The dumped values are that visit's pre-update losses -
exactly the numbers the step log records - so at accumulate 1 a dump
row's mean reproduces the step log's loss for that row's final visit,
which is the built-in cross-check. A run shorter than one pass dumps
only the rows it visited.

## struct EncodedPair

## fn dpo_loop

Peak memory is roughly twice the supervised loop's at the same window. At a
window of 512 this stage dies where continued pretraining survives, and the
out-of-memory surfaces as `Memory page 0 doesn't exist`, which reads like a
handle bug - read the `Caused by` chain rather than the panic line.

The causal mask is rebuilt every step although every pair pads to one width, so
it is trivially hoistable. Filed debt rather than intended.

## struct ScoredRollout

## struct RolloutGroup

## fn rlvr_loop

Uniform groups are common rather than exceptional, because the rewards are
binary: a measured attempt-two artifact had 89 of 211 groups scoring uniformly.

The reported loss is not a descent curve and reading it as one is a mistake. It
is `-advantage * sequence_logprob` summed over a group and divided by the group
size, so its scale depends on that group's own response lengths and advantage
spread, and consecutive steps are not comparable. The bench is this stage's
arbiter, not the loss.

`TrainReport::steps` is the count of groups that stepped, not the count
considered.

### The warmup schedule counts groups it never steps on

A defect, filed rather than fixed. The rate comes from `warmup_rate` indexed by
`step`, the group index, but a uniform group `continue`s before the optimizer
step - so the ramp advances on groups that never take one. Every other loop
here indexes the two identically because none of them skip.

Measured on the attempt-two artifact: 9 of the first 10 groups stepped, so
warmup completed at optimizer step 9 instead of 10, which is why the run was
not held for a fix.

The tail is what bites. An ordering that front-loads the uniform groups
consumes the whole ramp before a single update, and the first real optimizer
step then lands at the peak rate with cold Adam moments, which is exactly the
condition warmup exists to prevent. Nothing about shard ordering rules that out
on a later run, and the symptom would be one bad early step rather than an
error.

The fix is one line - index by the `stepped` counter the loop already maintains
- and it needs the train gate re-run afterwards, since every loop shares
`warmup_rate` and the toy trio asserts descent on it.

### The causal mask is rebuilt per rollout

Also filed debt. Every rollout pads to the same width, so the mask is identical
every time. At eight rollouts per group over roughly 122 stepping groups that
is on the order of a thousand rebuilds of one tensor, which is allocation churn
on exactly the axis whose failure signature is repeated small allocations
mid-step.
