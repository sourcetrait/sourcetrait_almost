# needle.rs

The needle battery as LIBRARY code rather than a test fixture, which is what
lets the engine's own ratchet and the trainer's evaluation drive the identical
instrument.

The method is the refquest needle instrument carried verbatim: filler generated
by an LCG seeded per target length, a four-round calibration loop converging
the chat-wrapped token count onto the target, greedy decode, substring match.
Reproducing it exactly is the point - a re-derived battery would not be
comparable to the recorded rows.

Multi mode is the LOST-IN-THE-MIDDLE probe that single mode structurally
cannot see: distractor needles are planted and only the target is asked for, so
answering a distractor is a distinguishable failure rather than a miss.

## struct NeedleKey

## struct NeedleSpec

### fn load

## enum NeedleMode

## fn label

## struct NeedleCellResult

`distractor_hits` records LEAKED distractor values, which is what separates the
interesting failure - answered the wrong needle - from a plain miss. The
recorded regressions all sit in that column.

## struct CellPrompt

## fn lcg_words

Seeded by `filler_seed` plus the target LENGTH, so every length batch draws its
own filler while any two runs at one length are identical. That is what makes a
per-cell diff against the reference meaningful.

## fn splice

## fn cell_plan

Distractors sit at fixed depth offsets from the target rather than at random
positions, so a cell stays deterministic and internally comparable across runs
and postures.

## struct NeedleRig

### fn encoded_len

### fn render_needle

### fn cell_prompt

Calibration is iterative because the relation between word count and token
count is only approximately linear: a ratio measured on a 2000-word probe seeds
the first guess, and each round corrects by the observed error. Four rounds
with a tolerance scaled to the target is what the reference does.

Equal plant indices bump forward so every plant lands and the splice stays
strictly increasing. Without that, two needles at nearby depths silently
collapse into one.

## fn run_needle_cells

One generation per cell over the model's carried caches, and the whole grid in
ONE process. That is a memory decision as much as a speed one - loading the
model per cell would be unaffordable, and the batteries at length are already
near the card's edge.
