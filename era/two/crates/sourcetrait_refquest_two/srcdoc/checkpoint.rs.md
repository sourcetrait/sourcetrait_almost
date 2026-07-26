# checkpoint.rs

Local checkpoint resolution, and the safetensors header as ground truth.

The header read is the one place this crate establishes a fact rather than
relaying one. It never touches tensor data, which is what makes it usable on a
14 GB shard as a routine check: the header is tens of kilobytes and the rest of
the file is never paged in.

The dead-code allow covers the difference between what the header reader can
answer and what a verb currently asks it. The tensor inventory exists because
pinning a checkpoint's ground truth is a thing we do by hand at a checkpoint
change, and its consumers are the tests and the operator rather than a verb.

## const DPO_MODEL_NAME

## const BASE_MODEL_NAME

## enum ModelPick

Two checkpoints, and the asymmetry between them is real rather than a
convenience. The DPO checkpoint is the deployed artifact and every reading that
matters is taken against it. The base exists for loader generality and
provenance, and it fails the engine's own token-map lock by design, since its
extra-id block sits at different positions and it carries no tool markers.

## fn model_name

## fn data_home

Honours the XDG spec's own fallback rather than requiring the variable.

## fn resolve_model_dir

The layout is a convention rather than something the checkpoints declare, so a
verb takes an explicit directory override for the case where a checkpoint sits
somewhere else. That override is also how a revision-pinned pull is baselined
without moving anything.

## struct TensorInfo

## fn byte_len

## fn read_safetensors_header

Entries come back name-sorted, which is `serde_json`'s object ordering rather
than something this function imposes. That is worth knowing before leaning on it:
the sort is a property of the parse, so a change to how the header is read could
lose it silently, and the inventory comparisons assume it.

`__metadata__` is skipped rather than reported, since it is not a tensor. These
checkpoints carry none, so the branch is there for the general case rather than
for this one.

The header-length sanity bound exists because the first eight bytes of an
arbitrary file are an arbitrary integer, and without the bound a wrong file would
attempt a huge allocation rather than raising.

## fn read_config

## fn read_added_tokens

Sorted by id, because the tokenizer file's own order is insertion order and the
comparison that matters is positional. The added-token map is what splits the two
checkpoints - the DPO map is era one's exactly, across all 22 entries, while the
base has no tool markers at all and its extra ids occupy different positions -
and that comparison is unreadable unless both sides are id-ordered.
