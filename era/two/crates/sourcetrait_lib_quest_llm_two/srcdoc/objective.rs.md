# objective.rs

The training objectives, as loss blocks over the top hidden state.

Every one of these takes the final-normed hidden state and returns a scalar, so
the per-layer gradient chain beneath them is untouched: the chain consumes a seed
gradient at the top hidden state and knows nothing about which objective produced
it. That is what lets supervised, preference and reinforcement training share one
trainer, and it is why adding an objective means adding a function here rather
than touching `train.rs`.

The mean-against-sum split runs through the whole file and is the thing to hold.
Continued pretraining averages over every position. The supervised form is the
same computation weighted by a mask and divided by the masked count instead of
the row count, which is the whole difference between "continue this document" and
"produce this reply". Preference and reinforcement both want a sum rather than a
mean, because the log-ratio and the policy-gradient term are defined over a
sequence's total log-probability and dividing by length would silently make long
replies cheaper to prefer.

The module is gated on the `train` feature in the module tree, so a non-train
build does not compile it at all. The `dead_code` allow covers the interval where
a stage loop is the only consumer of a function and that loop is itself gated.

## type FloatTensor

## const ADVANTAGE_EPS

Guards a division rather than a comparison. A group whose rollouts all score
alike has zero spread, and the standard deviation is what the advantage divides
by, so without this a uniform group divides by noise and produces arbitrary
large advantages instead of no signal.

## fn apply_head_delta

Narrow-and-cat rather than slice-assign, because differentiability through both
is not equally certain across burn versions and the cat form is unambiguous.
Applied per logits chunk rather than to the head matrix, so nothing
vocabulary-wide is ever materialized: the delta's cost is two thin matmuls and
one concatenation per chunk.

## fn masked_logprob_sum

The one primitive underneath all three objectives, which is why the mean-against-
sum decision lives in its callers rather than here.

Chunking bounds one materialization, not the retained total. Every chunk's logits
and log-probabilities stay live inside the autodiff graph until backward, so
lowering the chunk size makes more chunks rather than less peak memory. That is
the recorded misdiagnosis on this axis: it was named as the memory lever for a
preference-stage out-of-memory, and the source refutes it.

Masked positions contribute nothing by multiplication rather than by being
skipped, so the graph shape is independent of the mask. Skipping them would make
the number of graph nodes depend on the data, which is worse for a fixed-shape
device.

## fn supervised_count

## fn masked_cross_entropy

An example with nothing masked raises rather than returning zero. A zero-loss
step looks exactly like a clean step while training nothing, so the error is the
only way that condition is visible.

## fn sequence_logprob

Deliberately not length-normalized. Normalizing here would change what the
preference and reinforcement objectives mean, since both are defined on a
sequence's total.

The consequence for reading a rate across stages: AdamW normalizes the update by
its second-moment estimate, so this un-normalized sum does not make a nominal
learning rate harsher here than in the supervised stage - but the loss magnitudes
are not comparable between them either.

## fn softplus_stable

The overflow-stable form, and the reason is the backward rather than the forward.
The naive form's exponential overflows to infinity on a confident pair; the
forward survives that, because softplus saturates, but the backward then computes
infinity divided by infinity and yields NaN. This is the same guard the gating
path needed, and it is why the preference loss is written as `softplus(-x)`
rather than as `-log(sigmoid(x))`.

## fn dpo_loss

The reference log-probabilities arrive as plain floats rather than tensors, and
that is a memory decision showing in a signature. The reference model is the
adapter-off path - zero-init adapters are bit-exact to the base and the gate
locks it - so they are computed once without gradients and never need a second
copy of the weights.

## fn group_advantages

A group scoring uniformly yields zero advantages, which is the honest gradient
rather than a fault: there is nothing to prefer among identical outcomes. Uniform
groups are common rather than exceptional here, because the rewards are binary.

## fn policy_gradient_loss

## fn scalar_of
