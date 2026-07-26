# rollout.rs

Sampled generation graded by a mechanical verifier, which is the data the
reinforcement stage learns from, plus the greedy bench that reads a posture.

Rollouts ride the candle engine rather than the burn oracle. The oracle is a
stateless full-sequence forward with no cache, so generating a reply of any length
through it would replay the whole prefix per token. The engine has the decode path
already.

Rollout and training are separate verbs, deliberately, rather than two phases
inside one loop. Both stacks hold a full copy of the weights and two resident
copies do not fit the card, so the loop is driven from outside: generate, score,
step, repeat. That also makes the scored artifact a real checkpoint rather than a
value in flight.

The policy being sampled is whatever adapter the config names, so an on-policy
loop points the config at the adapter the previous step wrote.

## const ROLLOUT_TYPEDEF

## type PackedRollout

## type RolloutGroups

## fn rollout_run

A group of one is refused rather than allowed to degenerate. Advantages are
computed within a group, so a single sample is always exactly average and carries
no signal at all.

Each member draws its own seed, offset by its position in the whole run. Sampling
temperature is load-bearing for the same reason: a greedy group is one reply
repeated, and every advantage in it is zero.

The response ids are recovered by slicing the report's context trail past the
prompt length rather than by re-encoding the response text. Re-encoding would
cross a BPE boundary differently and produce ids the model never emitted.

## const BENCH_TYPEDEF

## fn bench_run

The same generate-and-grade machinery as a rollout, but greedy and one reply per
prompt, reporting a score rather than training data.

This is the only reading that answers whether the model is any good at the job.
The general battery indicates what a posture is breaking, which is a different
instrument: everything in this bench is something we care about by construction,
so a miss is a defect rather than a trade-off.

Its scope is narrower than it looks. It is a refinement test - held-out items,
each asking for one right answer to one prompt - which is what supervised tuning
does, so it has no axis along which a broadening stage can show anything. The
adapter token is recorded in the summary so a reading can be attributed to a
posture afterwards.

## fn load_groups

Reads a scored artifact back into per-prompt groups, keyed by the `prompt_index`
value.

That keying carries the trap this stage's operation depends on. `rollout run`
numbers its prompts from zero per invocation, so concatenating per-shard artifacts
without renumbering fuses unrelated prompts into one group and computes advantages
across them - no error, just a stage trained on nonsense. Offset each shard past
the whole index space before it, and verify the distinct-group count against the
expected prompt count before launching.

A rollout longer than the window is dropped rather than clipped, since a truncated
reply would be graded on text the model did not finish producing. A group left
with fewer than two members is dropped for the same reason a group of one is
refused at sampling time.
