# cli.rs

The global profile flags (dir/config/settings) survive from bquest
because the ledger verb loads the llm LibConfig to reach the
checkpoint's tokenizer, and the trainer verbs will want the same
profile machinery. `tokenize` sits at the top level rather than under
`tokenizer` because it is the everyday test surface (TheUser's
interface: `biquest tokenize "string"`); the `tokenizer` noun groups
the build-and-measure verbs.

The `associations` noun is its own verb family beside `tokenizer`
because the store is its own artifact class - the association
artifact the matrix build and stage-zero corpus consume - not a
tokenizer measurement. `matrix` is likewise its own family: the
embedding artifact is the foundation the trainer consumes, not a
measurement. `trainer` carries the organism's checkpoint and
training verbs; `assemble`/`disassemble` sit at the top level beside
`tokenize` as the everyday Quill test surface. TrainerTrainArgs
stays un-gated so the parser is feature-blind (one `doc cli` tree
per build); only the dispatch is cfg-split.

The `wikimedia` noun is the WikimediaDumpTool's family: raw-source
access over the pinned wikimedia dumps and export saves, feeding the
document pipeline that replaces wiktextract. Its `page` verb takes
the source and index as explicit paths for the same reason the other
artifact paths are flags - the dumps live on the tmp home tier.

Artifact paths are explicit flags rather than config fields: the
tokenizer artifacts live on the tmp home tier and their homes are
operational knowledge, not product configuration. The character table
and the default English word set take no path at all - both are
embedded (ucd.rs, dictionary.rs), so `biquest tokenize "string"` runs
bare and shows the with-dictionary form; --words swaps in another
word file (file order = id order - the whole dictionary IS the
vocabulary).
