# graph.rs

The decode-graph building blocks: the staged device buffers every per-step
dynamic rides, the raw slot-write scatter the KV append uses inside a capture,
and the bucket-keyed cache of instantiated graphs.

ONLY THE EIGHT ATTENTION LAYERS STAGE. The GDN layers need none of this because
their carried state is fixed-shape and updates in place on every path, so the
classic single-token step IS the captured form already. One slot and one mask
serve all eight, because every layer sees every token and their lengths are
uniform by construction.

## const KERNEL_SRC

Raw pointer arguments on the model's own stream rather than candle's safe slice
arguments. Those wait on per-slice events, which is capture-illegal inside an
active capture when the events predate it.

The kernels are element-size generic, so bf16 and f16 share one entry point and
f32 and u32 share the other.

## static MODULE

## fn kernel

## fn launch_slot_write

A CANDLE VIEW'S START OFFSET IS PART OF ITS ADDRESS, and this is the defect
that cost the most in this file. A tensor derived through a narrow can report
itself CONTIGUOUS while pointing mid-storage, because size-one dimensions skip
the stride check - so a value tensor narrowed out of the fused projection
stayed offset through every subsequent contiguity, reshape and transpose call,
all of which were no-ops on that layout.

Reading from the storage base therefore wrote the WRONG SPAN: raw
query-projection bytes landed as value. It presented as a bounded,
context-independent single-step error of about 2e-3 - scores right, values
wrong, so argmax mostly held - compounding through GDN state carry to 1.3e-1
and 48 of 106 argmax over an incremental replay.

The probes missed it precisely because probe tensors sit at offset zero. Every
operand is now sliced at its layout start offset, and two locks hold it.

## struct SlotWrite

### impl InplaceOp3 for SlotWrite

## fn trim_graph_memory

Destroyed graphs leave their in-graph allocation pools driver-cached; the trim
returns them. Best-effort by design - failures are ignored because a failed
trim is a missed reclaim rather than a fault, and it is safe to call beside
live graphs.

## fn bucket_for

## fn pad_mask_values

Pad columns take negative infinity rather than a large negative number, so
post-softmax pad mass is exactly zero rather than merely small.

## struct DecodeStage

Created ONCE per model at first arming and kept. Masks are width-pooled and
value-reset IN PLACE, so every address a captured graph bakes stays stable; a
clear epoch only drops the armed flag rather than dropping buffers.

`capture_permitted` is a LIFETIME switch where `capture_enabled` is a
per-arming one. Once a regrow parks captured buffers, capture stays off for
this model forever, and re-arming restores `capture_enabled` from
`capture_permitted` rather than from true. Conflating the two would silently
re-enable capture against parked buffers.

`kv_capacity` records what the cached graphs were captured against, so a
capacity change is detectable as staleness rather than as a crash.

### fn new

### fn pooled_mask

Get-or-create then value-reset in place. A width's mask must never reallocate
once a graph has baked its address, which is why this returns a handle into a
pool rather than a fresh tensor.

### fn write_mask_values

### fn rearm

### fn ensure_bucket

Re-derives the bucket for the width the NEXT append reaches, so the mask switch
and the graph switch happen on the same boundary - the cache key is the bucket,
so they cannot drift apart.

### fn stage_step

The staged writes are tiny host-to-device transfers OUTSIDE the captured
region, and the operation is state-free: everything derives from the live
length, and re-enabling an already-enabled mask column is a no-op. That is what
makes a replay safe to repeat.

## struct GraphCache

`Rc` rather than `Arc` because a CUDA graph is single-thread by cudarc's
contract. The wrapper exists so the containing types keep their derives -
Debug renders a summary rather than the graphs, and Clone shares the
instantiated executables.

### fn get

### fn insert

### fn clear
