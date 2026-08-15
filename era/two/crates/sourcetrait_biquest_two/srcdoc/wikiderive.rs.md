# wikiderive.rs

The provenance derivation: the one-pass walker behind both
`tokenizer dictionary` and `associations build`, replacing the
wiktextract jsonl with the raw dump through our own parser. Each
verb runs its own pass rather than sharing an intermediate artifact:
the passes are minutes-class, run at re-vendor cadence only, and a
directory contract between two verbs would be a second schema to
keep current for a cost nobody pays twice a year.

## fn mentions_english_heading

The cheap pre-filter that makes the full-dump walk affordable: only
pages whose text carries an h2-shaped English heading line parse at
all, so the ~85% of ns0 pages with no English section skip the
wikitext parser entirely. The line test accepts the spaced form
("== English ==") and rejects h3+ (a leading "==="); a false
positive (the string inside a comment) costs one parse and is then
refused by the parsed-block check in derive_page - the filter is an
optimization, never the authority.

## struct DumpDerivation / fn absorb

The wiktextract absorption's shape, re-sourced: the title stands
where entry `word` stood, derive_page's forms where `forms[]` stood,
its senses where `glosses` stood, and its form-of lemmas where
`form_of` stood. Admission stays per-candidate and independent - a
single-code-point title ("d") admits nothing itself while its
derived forms still can, exactly as wiktextract entries behaved.
Senses and form-of links key off the folded title and drop when the
title does not admit; forms-to-title links count only on first
insert, so a form-of render of an already-linked pair does not
double-count. Links are NOT closed over the word set here - the
writer filters - because closure is a property of the written
artifact, not of the accumulation.

A page the parser faults counts parse_failures and skips: one page
must never kill a 10M-page pass, and the count is the signal to go
look. POS keys are the lowercased English section names ("noun",
"proper noun") - the section taxonomy is ours now, not wiktextract's
code vocabulary.

## fn derive_pages / fn derive_dump

Generic over the reader so the unit tests feed an in-memory XML
stream through the same code the 1.93 GB multistream dump rides
(open_pages' MultiBzDecoder streams every block, so no index is
needed - the whole-walk case is exactly what multistream decode
into a stream is for).
