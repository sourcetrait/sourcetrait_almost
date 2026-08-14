# census.rs

## fn census_counts

The corpus type census counts what would TOKENIZE, not what merely
splits: a connected candidate the full dictionary holds counts as one
type, an edged candidate whose core it holds counts the core (edges
tallied as characters), and only a full miss counts the word parts -
so admission frequency measures the forms the segmenter would
actually emit. That is why the census takes the word set: the
dictionary is the tokenizer's decision surface, and a
dictionary-blind census would count parts for forms that tokenize
whole. UTF-8 is asserted, not coerced - a non-UTF-8 corpus file is an
error naming the file, matching the ingestion-refusal posture (we
control the corpus).

## fn tokenizer_census

The counts artifact is word-tab-count lines in census order:
count-descending, then alphabetical - deterministic and
reproducible from the same corpus. Plain lines for the same
bulk-intermediate reason as the dictionary word set.

The census is an INSTRUMENT, not a gate: the vocabulary is the whole
dictionary (TheUser: the ours corpus is far too small to size a
vocabulary), so nothing downstream consumes the counts today. They
price frequency questions - the acquisition-curve instrument, the
expansion loop's is-it-frequent test for non-dictionary lexemes -
and that is the reason the verb survives the admit verb's removal.
