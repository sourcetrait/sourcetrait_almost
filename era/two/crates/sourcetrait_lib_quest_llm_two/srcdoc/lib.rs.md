# lib.rs

The llm lib: the candle engine (correctness core, speed tracks,
capability), the burn oracle, and the trainer core. Split out of the old
sourcetrait_lib_quest_two so the inference internals are one black-box
crate; the channel/ThinkHarness/data-core half is
sourcetrait_lib_quest_harness_two, which depends on this crate for the
chat-render vocabulary.

Feature ladder: `oracle` carries burn (no autodiff) for the parity
triangle; `train` stacks autodiff (the adapter loops in `train` and
the organism's full-parameter loop in `imagine_quest_train`);
`train-cuda`/`burn-cuda` are the two cuda grades and never combine
with each other. `cuda`+`train-cuda` in one binary is impossible
(cudarc nccl unification), which is why bquest builds twice.

The oracle's `causal_mask` is deliberately off the hub - the candle
mask owns the bare name; the burn one is pathed `oracle::causal_mask`.
