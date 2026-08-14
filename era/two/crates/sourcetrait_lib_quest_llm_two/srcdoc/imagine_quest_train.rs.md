# imagine_quest_train.rs

From-scratch full-parameter training for the ImagineQuest organism,
riding the burn oracle stack. The adapter machinery (train.rs) is
frozen-base by construction, so full-parameter training needs its own
parameter walk; this module is that walk plus the optimizer and the
checkpoint writer. Backend-generic: the toy gates run cpu f32, the
real runs ride train-cuda.

The vocabulary dominates the model (a ~1.27B-row embedding over a
~116M core at the full dictionary), and two design choices exist
because of it:

- The embedding's f32 master is the oracle's own host `embed_rows`
  (the oracle gathers embeddings host-side), so the big matrix never
  needs a device-resident master, gradient, or moment pair. Only the
  tied head (`lm_head_transposed`) lives on the device.
- Adam state for the embedding exists per touched row only, created
  on first touch - the row-sparse lever recorded in the imaginary
  subiter before this module was built. Full AdamW state on the
  embedding would be ~15 GB against the 21,248 MiB train constraint.

## the touched-row gradient semantics

A full softmax gives every vocabulary row a nonzero head gradient
(the denominator pushes down all logits), so "rows a step touched"
cannot mean "rows with gradient". The contract implemented: touched =
input ids plus target ids; their gradients are EXACT under the true
full-softmax loss (input-side scatter plus the head formula
`(p - onehot) / N` over touched columns, with p recovered from the
loss pass's per-position log-sum-exp); the dense remainder - the
denominator's push on untouched rows - is deliberately dropped.
Untouched rows freeze at init and act as a static negative floor the
trained rows grow past; the expansion loop's frozen-core training
already contemplates exactly this shape. The walk test demonstrates
the drop is real (untouched rows DO carry reference gradient) and
that touched rows match the dense reference at 1e-9 nmse.

## the chunk-independent loss

`organism_loss_block` runs each loss chunk as its OWN autodiff block:
leaf on the chunk's top rows plus a fresh final-norm leaf, logits,
f32 log_softmax, backward, seed slice out. One graph over all chunks
would hold every chunk's vocabulary-wide logits and softmax buffers
at once - at 1.24M vocabulary and 2048 positions that is ~10 GB and
the whole reason "chunked loss bounds the memory" is true here.
rms_norm is row-wise, so per-chunk final-norming equals whole-tensor
final-norming; the final-norm gradient accumulates across chunks.
The per-position lse rides out of the same pass (target logit minus
target log-prob) for the head formula.

## the full-parameter walk

`organism_chain_from_seed` mirrors train.rs's adapter chain with the
tracked set swapped: each layer replays under autodiff with its OWN
weights lifted as leaves (`lift_gdn_trainable` / `lift_attn_trainable`;
grad_remove works on clones because clones share the autodiff node).
The GDN layer keeps the three-phase segmented shape - post-half
backward, per-segment recurrence backward (weight-free), pre-half
backward - because an unsegmented recurrence graph over thousands of
sequential steps is the memory hazard RECURRENCE_SEGMENT exists for.
The pre/post leaf split (`GdnLeafSet`) tracks each weight only in the
half that uses it; o_norm is a POST weight (rms_norm_last inside
gdn_post). The walk also returns the bottom gradient - the
embedding-output gradient the input-side scatter consumes - which the
adapter chain drops.

## masters and the optimizer

Core parameters keep f32 master + moment + velocity tensors keyed by
CHECKPOINT tensor name (matrices in the oracle's transposed
orientation; conv taps as per-tap rows under `.tapN` suffixes;
A_log/dt_bias as the oracle's (1, h) rows). After each step the
masters cast back into the model's compute dtype (probed once at
load: f32 on cpu, bf16 on cuda), so forwards stay on tensor-core
dtypes while updates never stall on bf16's step size - the classic
mixed-precision master pattern. AdamW is manual (~6 elementwise ops)
with betas (0.9, 0.95): the from-scratch pretraining convention (the
reference stacks use 0.95), against burn's 0.999 default the adapter
loops kept. Weight decay is deliberately absent, matching every
existing loop in the estate; surfacing it is a flag-level addition if
ever wanted.

Embedding rows update host-side in f32, then the head's touched
COLUMNS re-anchor to the master. burn 0.21's float `select_assign`
implements only the Add op (Assign is `unimplemented!()` at the
dispatch layer - measured), so the re-anchor is
`old + (new - old)`: equal to the master within one rounding, and
re-derived whole every step so rounding never compounds.

## fn save_checkpoint

Writes the organism checkpoint format trainer_init established: the
SOURCE config.json bytes verbatim (HybridCheckpointConfig drops
fields, so re-serializing it would corrupt the config), bf16 tensors
transposed back to checkpoint orientation, conv taps reassembled to
(channels, 1, kernel), the tied embedding under both names, tensors
name-sorted for byte-deterministic output. `load_config` validates
the written directory before returning, the same acceptance check
trainer_init runs.

## fn reference_step (test)

The gate's oracle: one full graph with every weight a leaf and the
head tracked densely, so the walk's core gradients, the touched-row
embedding gradients, and the dropped-remainder claim are all checked
against ground truth at toy scale.
