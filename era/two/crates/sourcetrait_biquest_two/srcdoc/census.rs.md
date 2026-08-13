# census.rs

## fn census_counts

The corpus type census: folded word-piece types with counts. UTF-8 is
asserted, not coerced - a non-UTF-8 corpus file is an error naming the
file, matching the ingestion-refusal posture (we control the corpus).

## fn tokenizer_census

The counts artifact is word-tab-count lines in census order:
count-descending, then alphabetical - deterministic so downstream ids
are reproducible from the same corpus. Plain lines for the same
bulk-intermediate reason as the dictionary word set.

## fn tokenizer_admit

The ruled admission: corpus frequency filtered through Wiktionary
membership - the corpus says used, the dictionary says real. Census
order is preserved, and the admitted artifact's row order IS the
dictionary layer's id order; the count column rides along because the
matrix build will want frequency for warm initialization. min_count
defaults to 1 (membership alone gates); it exists as the tuning knob
for a later, larger corpus.

occurrence_coverage in the summary is the share of word-piece
occurrences the admitted set covers - the day-one number that says how
much of the corpus resolves at the dictionary layer versus falling to
characters.
