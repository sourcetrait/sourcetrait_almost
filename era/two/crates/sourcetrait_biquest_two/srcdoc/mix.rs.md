# mix.rs

The training-mix pipeline's document stage: corpus trees and foreign shards
become dolma-field documents, and those documents become a tokenized chunk
stream.

The presentation is the lineage's rather than ours, and that is the point of the
whole stage. One file per document, the text field carrying verbatim bytes with
nothing prepended, and identity in metadata alone. ai2's own renderers put the
repository, path and license in a sidecar, so a header convention of ours would
teach the model something their model never saw.

Artifacts are whole-value NUON tables rather than the NUON-lines form.
Document text carries raw newlines, which the lines form's single-line guard
rightly refuses, so lines are reserved for logs and progress.

Both sides of the mix land in the same shape. `mix render` walks our own corpus
trees and `mix rip` streams a foreign shard, and nothing downstream can tell
which side a document came from - which is what lets one sampler and one packer
serve both.

## const MIX_SPEC_TYPEDEF

## const MIX_DOCUMENT_TYPEDEF

The `kind` field is our own metadata extra rather than a dolma field, admissible
because records are open. It exists so the pack stage can gate infilling to code
documents, since the lineage never infilled prose.

## struct MixSpecRow

## fn read_spec

## fn language_token

Extensionless files read as text rather than erroring, and an unmapped extension
carries verbatim rather than becoming a default. The field is metadata that
nothing gates on today, so a wrong-but-honest value is better than a guess.

## fn render_tree

The walk collects and then sorts, rather than relying on directory order.
Filesystem order is not stable across systems, and the pack is meant to be
reproducible from the corpus rather than from the machine that read it.

A non-UTF-8 file raises rather than being skipped. The corpus is UTF-8 by its own
audit, so an unreadable file means the tree is not the corpus we think it is.

## const FIM_PREFIX

## const FIM_MIDDLE

## const FIM_SUFFIX

The pipe-wrapped spellings, which this tokenizer already carries as single added
tokens. That was verified directly rather than assumed, and it is what makes
infilling-style training need no vocabulary work at all.

## const FIM_RATE

## struct SplitMix64

## fn new

## fn next_u64

## fn next_unit

## fn next_below

The crate's whole source of randomness, deliberately. Every generator, sampler,
shuffle and adapter initialization draws from this, so an artifact is
reproducible from its seed alone and no external generator's version can change
what a seed means.

## fn fim_transform

Reproduces the lineage transform exactly: the rate per document, two uniform
character break points, and an even split between the two orderings.

It is character-boundary safe and the three spans always reassemble the original
text, which is the property that makes the transform lossless rather than merely
plausible.

Note that it is nearly inert on the current selection. The gate is a document's
`kind`, and our side's selection is documentation, so the setting reaches almost
nothing. That is a property of the selection rather than of this function, and it
becomes a live choice the moment our side draws on source.

## struct PackDocument

## fn tokenize_documents

The sentinels are added tokens, so an infilled document's markers land as single
ids rather than as spelled-out text. Every document ends with the end-of-text id,
which is where document boundaries live: the data itself carries none, because
the lineage wraps at tokenization rather than in the text.

## fn chunk_and_shuffle

The ragged tail is dropped rather than padded, and it is reported so the loss is
visible. Padding it would train the model on padding.

## fn mix_pack

Documents are consumed in the order given, so the argument order is the mix
order. Everything is deterministic per seed end to end, because the infilling
draws and the chunk shuffle share one generator.

## fn mix_sample

One seeded shuffle across the union of every input row, taken in shuffled order
until the byte budget is crossed. The crossing document overshoots, and the
overshoot is visible in the reported byte count rather than hidden.

The trim is therefore proportional rather than selective: each source is thinned
in proportion to its share, so a class allocation survives the trim and no source
is dropped to make the number work.

One measured caution on the budget: bytes do not proxy tokens across languages.
Rust source tokenizes at about 3.29 bytes per token against the nushell family's
4.34, so a byte-matched leg landed at 1.43 to 1 rather than 1 to 1. Budget from
target tokens times a measured bytes-per-token, not from bytes alone.

## fn mix_render

## fn mix_rip

The their-side counterpart to `mix render`, taking a shard rather than a tree.

The decoder lives here because the box ships no zstd binary and nushell reads
none. The harness previously drove a throwaway Rust binary built per session and
lost to the session prune; bquest already carries Rust and already reads these
shards at pack time, so folding it in removes the replay leg's only external
dependency.

Decompression is streamed, and that is a storage requirement rather than a
refinement. The caller's loop fetches one shard, rips it, and deletes it, which
bounds peak storage at one compressed file - materializing the decompressed shard
first multiplies that several-fold for no gain. So the budget gates the read
rather than the collection: a met budget stops decompressing.

`source` carries the stream name rather than the upstream's own source field,
because a wayside audit counts documents per stream and that count is the
evidence that an excluded topic contributed nothing.

A document without text is counted and skipped rather than raising, since these
shards genuinely carry some. A shard yielding no text at all does raise, because
that means the file is not what we think it is.
