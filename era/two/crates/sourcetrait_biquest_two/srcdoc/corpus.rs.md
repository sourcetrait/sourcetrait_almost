# corpus.rs

## fn corpus_files

The one corpus walker, shared by the ledger and the trainer so both
surfaces see an identical, deterministically sorted file set for the
same roots. Sorted because file order is measurement order for the
ledger's per-file rows and packing order for the trainer.
