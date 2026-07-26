# profile.rs

The attention-density observation instrument, behind its own feature.

The partition is three ways per query row at absolute position p, eligible only
when a middle exists at all: the sink at column zero, the trailing window, and
everything between. Columns past p carry exactly zero softmax mass, so they
fold into the recent band harmlessly rather than needing their own case.

What it measured decided a design question: over 30 single-mode prompts at 16K
through 32K, per-head prefill middle mass ran 0.71 to 0.96 by layer against
Olmo 3's 0.43 to 0.57, with no head anywhere near a windowable range. That is
why head windowing was struck and the question-window ranking is the only
viable cut.

## const PROFILE_ROW_SLICE

## struct HeadMasses

## struct LayerProfile

## struct ProfileAccum

Interior mutability is forced by the call site rather than chosen: attention
runs under a shared reference, and the accumulator has to mutate.

### fn new

### fn accumulate

The masks build ARITHMETICALLY on device from integer-valued f32 arithmetic
plus a clamp, so the host carries only per-row thresholds rather than uploading
a mask per slice. Integer-valued arithmetic is what keeps the derived masks
exact rather than approximately zero or one.

A slice with no eligible row returns early, which is what keeps short prompts
from contributing a division by zero to the means.

### fn report
