# train.rs

The training verb layer: argument parsing, pack loading, logging and the
stage drivers over the library's objective-agnostic chain. The loops, the
objectives and the gate live in the llm library; what is here is how a
command line reaches them.

## struct ChunkPack

## fn load_chunks

The rejection exists because of what the previous form did. It read one named
tensor and ignored the rest, so the first objective to write a second tensor
beside the ids would have had it skipped in silence - a run that completes
cleanly having trained the wrong objective. The check lives in the reader
rather than the writer because the reader is what a future writer's addition
has to survive.

## fn train_gate_verb

The verdict prints before the failure raises, so a failing gate still reports
which lock failed and by how much.

## fn stage_log_path

## fn cpt_log_path

## fn append_step_log

Appending per step rather than at the end means a crashed or killed run still
leaves its progress on disk. The consequence for a reader is that a step log
accumulates across runs, so the live run starts at the last row whose step
is 1.

## fn print_stage_summary

## fn stage_adapters

Rank and alpha come from the resumed artifact rather than from the flags,
because a resumed adapter's geometry is already decided and taking them from a
flag would let a mismatched rank load as garbage.

## fn stage_options

## fn train_cpt

This verb carries its own argument struct rather than sharing `StageArgs`,
which is why its learning-rate default is 2e-4 where every post-training stage
defaults to 1e-5. It also offers no `--resume`, being first in the chain.

## fn train_sft

The mask comes from the renderer's assistant spans at pack time, so the
supervised boundaries are exact by construction rather than re-measured here.

## fn train_dpo

The reference log-probabilities are computed once up front from the adapter-off
forward. Zero-init adapters are bit-exact to the base and the gate locks that,
so the reference model costs no second copy of the weights - which is why the
loss takes floats rather than tensors on that side.

This stage's window wants to be well under the continued-pretraining stage's,
because the preference loop holds two of everything.

## fn train_rlvr

`--rollouts` takes one artifact, and merging shards into it carries a trap.
`rollout::load_groups` keys groups by the `prompt_index` value, while
`rollout run` numbers prompts from zero per invocation. Concatenating
per-shard artifacts without renumbering therefore fuses unrelated prompts into
one group and computes advantages across them, silently and with no error.
Offset each shard past the whole index space before it.

Rollout and training are separate verbs because the generating stack and the
training stack each hold a full copy of the weights and two do not fit the
card. An on-policy loop alternates processes rather than phases: point the
config's adapter token at the last adapter, roll out, step, repeat.
