# snapshot.rs

The saved-context file format and the three-tier token resolution.

SNAPSHOTS ARE SETTINGS-AGNOSTIC, and that is a ruling rather than a
convenience: state is state, so a file saved under any settings combination
restores under any other - fusion, graphs, any eviction posture. Settings live
with the model and never with the file, which is why nothing here records them.

Eviction-armed saves are supported, deviating deliberately from era one's
reject guard. The justification is mechanical: the compacted store IS the
model's state, and a continuation's suffix prefill re-scores the whole store
before any overflow epoch can fire, so the score buffers rebuild for free and
are never persisted.

The default snapshots home is the CACHE home rather than the data home, because
a snapshot is regenerable. Checkpoints take the opposite call.

## const SNAPSHOT_VERSION

## struct RestoredContext

The trail is at least the context length, and longer EXACTLY when the save was
eviction-compacted - the store shrank and the trail did not. That inequality is
the validation rule rather than an accident, and equality is what the exact
configuration produces.

## fn is_bare_relative

## fn snapshot_path

## struct SnapshotFile

### fn tensor

## fn write_snapshot

The id trail travels in the same file as the caches rather than beside it, so a
snapshot cannot be separated from the transcript that produced it.

The trail-covers-length check is a caller-bug guard, and the read side repeats
it because the file may not have come from this process.

## fn read_snapshot

Validates FORMAT only - version, model identity, the trail. Layer shape and
dtype validation belongs to the restoring model, which owns the expected
geometry; splitting it that way is what keeps this function checkpoint-agnostic.
