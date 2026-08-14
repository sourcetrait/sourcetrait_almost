# ucd.rs

The character layer is authored from the standard, not learned from a
corpus: every Unicode-assigned code point gets a row, so admission is
by the Unicode version and adding a language later is training data,
never a tokenizer migration. Unicode 17 is the pinned source; a future
version bump appends new rows after the 17 set rather than repacking,
so the dense index stays a stable address.

The UCD is vendored at `data/ucd/` (UnicodeData, PropList,
CaseFolding, Scripts) and embedded via include_str, TheUser's call:
the table is a design constant, so the binary carries it and no verb
takes a data path. The Unicode license sits at the repository's
docs/licenses/unicode/. The ~2.6 MB of text parses in ~20 ms at
startup. A Unicode version bump re-vendors those four files; that is
the whole migration.

## enum CharClass

TheUser-ruled shape: Word, Digit, Unicode, Other - the first three
partition the ASSIGNED set whole. Word is L*/M*/N* (letters,
combining marks, numbers - marks ride inside words). Digit is
exactly the ten ASCII digits 0-9 (his scope ruling): word-constituent
for piece formation so identifiers and the dictionary artifact are
unchanged, but exempt from run encoding - a number is place-value
content, not repetition. Non-ASCII Nd digits stay Word. Unicode is
everything else assigned - whitespace, punctuation, symbols,
controls, format characters - one token per character, named for
where it resolves. His ruling closing the totality loop: an assigned
code point is never unlexable (an ESC is just a character); that
totality is why the Unicode floor exists. Other means unassigned -
no row - and is the ingestion refusal, alongside invalid UTF-8. The
White_Space flag is kept as a property for the associations work,
not as a class discriminator.

## struct CharacterTable

### fn from_unicode_data

Range pairs (`<..., First>` / `<..., Last>`) expand with the First
entry's properties per UAX#44; that is what carries the CJK blocks and
Hangul syllables into the row set. Surrogates (Cs) and private use
(Co) are skipped at parse: UTF-16 machinery and private meaning get no
rows, by design. Hangul canonical decompositions are algorithmic and
deliberately not materialized here; the decompositions map carries
only what UnicodeData spells out. Compatibility decompositions (the
`<tag>` forms) are skipped - canonical only.

### fn apply_case_folding

CaseFolding statuses C and S only: the simple, one-to-one fold. The
full fold (F, one-to-many like eszett to ss) is rejected because the
fold must stay a per-character map for the lexer's folded lookup and
the later spelling-composition work. Dictionary matching is therefore
simple-fold case-insensitive.

### fn apply_white_space / fn apply_scripts

White_Space comes from PropList because the general category cannot
express it (TAB/LF/CR are Cc). Scripts intern into a u16 vocabulary;
u16::MAX marks the unlisted.

## fields simple_uppercase / simple_lowercase / decompositions / numeric_values

Parsed now, consumed by the ImagineQuestAssociations task (case
pairs, canonical decompositions, numeric values as association
edges). The ucd stats verb reports their sizes so the parse is
exercised from day one.
