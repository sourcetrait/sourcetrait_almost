# corpus.rs

## fn corpus_files

The one corpus walker, shared by the ledger and the trainer so both
surfaces see an identical file set for the same roots. Order is the
contract: roots contribute in argument order and only a directory
root's own walk sorts, because pack order is the trainer's curriculum
(the genesis document rides first, pages follow in reading order) and
measurement order for the ledger's per-file rows. An earlier form
sorted the whole collection globally, which re-ordered deliberately
sequenced roots; the trainer's order test pins the current contract.
