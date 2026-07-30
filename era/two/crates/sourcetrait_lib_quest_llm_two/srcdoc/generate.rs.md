# generate.rs

Era one's settled API shape, implemented freshly. The caller drives the loop:
one `next()` yields exactly one step, per-step errors ride the Item and fuse
the iterator, and dropping the iterator early is a clean stop rather than an
abort.

A SAMPLED STOP TOKEN IS NEVER CONSUMED into the caches. That single decision is
what keeps a saved context transcript-complete mid assistant turn, and it is
why `chat_continue` has to close the open turn.

## const PREFILL_CHUNK

## struct GenerateOptions

The defaults are the checkpoint card's recommended posture rather than ours.

`sample_len` is a plain BUDGET rather than a position clamp, because NoPE
leaves the hybrid with no position ceiling at all - VRAM is the real bound.
Era one's 65536 clamp deliberately does not carry.

### impl Default for GenerateOptions

### fn greedy

## struct GenerationStep

The chunk can legitimately be EMPTY while a multi-token grapheme is pending.

## enum FinishReason

The two endings differ in what the caches hold, which matters to a snapshot: a
stop-token ending is transcript-complete, while a budget-exhausted ending
leaves the final emitted token outside the KV.

## struct GenerationReport

## struct Generation

## impl OlmoHybrid

### fn generate

### fn generate_from

Continuing a RESTORED context. The suffix renders as the next chat turn unless
the caller asked for verbatim, prefills at the restored offset, and then runs
the ordinary post-prefill sequence - including eviction compaction when the
restored length plus the suffix exceeds the cap, whose scores the suffix
prefill has just rebuilt.

### fn start_generation

The shared core, and the arithmetic in it is identical for a fresh run and a
continuation because the carried context is simply zero in the fresh case. That
is why there is one core rather than two paths that must agree.

THE PRE-RESERVE IS THE MEMORY LEVER. One allocation at exactly prompt plus
margin, ahead of the first chunk, replaces roughly one regrow per reserve step
of prefill - and each regrow holds the old and new buffers during its copy and
frees an odd-sized block into the raw allocator, which is the fragmentation
that inflated the 32K peak.

ORDER MATTERS AT THE END of prefill: eviction compaction runs BEFORE graph
arming, so graphs capture against the compacted store and the shrunk buffers.
Arming first would bake addresses the compaction then replaces.

Speculation keeps the classic path silently rather than erroring, because
variable-length verify chunks would churn graph buckets. Graph coexistence
stays design-listed rather than broken.

## impl Generation

### fn sample

### fn emit_chunk

The incremental detokenizer diffs full decodes rather than decoding the new
token alone, which makes it prefix-consistent with the final decode BY
CONSTRUCTION rather than by agreement between two decoders.

The replacement-character hold-back is what stops a partial multi-token
grapheme from being emitted as U+FFFD and then corrected.

### fn step

### fn stage_or_speculate

The emitted token joins the index BEFORE drafting. Without that, every draft
continues the pre-token tail and competes with the token just emitted - era
one's off-by-one, and the reason this ordering is explicit.

On a PARTIAL accept the verify chunk's accepted-prefix rows are KEPT: they
computed identically to a plain advance, so acceptance is a length reset with
zero recompute, and only the cumulative GDN caches re-advance from captured
rule inputs. That is what makes speculation a single-pass mechanism here, and
it is what flipped schema-shaped workloads from a small net loss to a gain.

### fn finish

## impl Iterator for Generation
