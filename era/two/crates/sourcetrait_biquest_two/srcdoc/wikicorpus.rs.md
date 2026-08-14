# wikicorpus.rs

The ruled first attempt: depth 1, pass 1, dictionary side only -
every vocabulary word's own document, nothing spidered.

## fn wikimedia_corpus

The folded word-to-rows map is built in memory from one index scan
per run (seconds), never persisted: a title-map artifact file would
be an unconsumed instrument, since every consumer of the map also
wants the scan's freshness against the pinned dump.

Words iterate ordered by their first block offset, so the dump reads
nearly sequentially through a 16-block LRU: runs of words share a
block, and a word whose pages sit in far-apart blocks (linux/Linux
class) still finds the earlier block cached. The measured smoke: 300
words rendered from 89 block reads in 7.3 s, zero render failures.

A render fault files a render_failed audit row and the run continues
- the corpus never dies on one broken page, and the audit is where
brokenness surfaces (the wiktextract anti-goal, held).

missing_words.txt is the empty-tails measurement made concrete:
vocabulary words with no fold-matching ns0 page - 41,055 of
1,071,639 (3.8%) at the 20260801 pin, so 96.2% of the vocabulary is
backed by its own document. The list is consumed twice: it prices
TheUser's very-few-empty-tails claim, and it is the word-keyed
Wikipedia stage's target input.

The smoke's audit-class distribution is the family-handler growth
map: template_unhandled dominates (3,151 rows over 300 words), then
the pronunciation family, dropped sections, list lines, sense
lines, refs, accent codes, quotes, and head arguments.
