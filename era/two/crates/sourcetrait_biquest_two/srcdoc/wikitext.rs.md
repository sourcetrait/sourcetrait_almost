# wikitext.rs

The wikitext structural parser and the ruled typography
normalization. Faithful capture, no cleaning: policy belongs to the
document renderer, and the audit needs an untouched parse to diff
against.

## fn normalize_typography

The ruled set (TheUser): single-quote marks to the apostrophe,
double-quote marks to the double quote, en-dash to a bare dash,
em-dash to a single dash carrying the spacing the typographic form
omitted. The em-dash algorithm adds a space only on an unspaced side,
so "word - word" comes out identically from "word—word", "word — 
word", and the mixed forms, and no double space is ever produced.
Candidates beyond the ruled set render to TheUser measured (the
typoscan probe over the 1.12M-row gloss corpus plus the raw samples;
counts in the campaign journal) and are never auto-extended.

## struct Cursor and the byte-scanning design

Every structural character in wikitext is ASCII, so the parser scans
bytes and slices text only at structural positions, which are always
char boundaries. The trap this bakes in: the segment scanners advance
byte-wise through arbitrary content, so the cursor can sit inside a
multibyte character between structural positions - any str-slicing
pattern operation panics there. `starts_with` and `find` are
byte-based for exactly that reason; the dump smoke caught it live
(the IPA stress mark U+02C8 in the Microsoft pronunciation line) and
`multibyte_content_scans_safely` locks it. The second instance of
the class was the entity scanner's length-capped lookahead
(`rest[..min(12)]` landing inside an em-dash), caught fourteen
seconds into the first full corpus run - the rule generalizes: any
str slice at a COMPUTED length is suspect; only found-ASCII
positions are boundary-safe.

## fn parse_blocks

Line-shape classification at depth zero: heading (balanced `=` runs,
level from the shorter run), table (`{|`..`|}` captured raw as a
strip class, nesting-aware), horizontal rule, list markers
(`#*:;` runs), blank, else paragraph. A template or link opened on a
line continues across newlines inside its own scan, so a multi-line
quote template stays one inline of its list item.

## fn parse_inlines

Comments strip silently (MediaWiki: an unterminated comment swallows
to the end). Entities resolve into the running text: numeric
references always, thirty named ones, and an unknown name stays
literal so the audit sees it. Emphasis is a TOGGLE stream, not a
tree - wikitext and markdown share the toggle model, so the renderer
maps runs directly; the apostrophe-run rule is 2/3/5 with extras as
leading text (four apostrophes = one literal plus bold), covering
the corpus without MediaWiki's full disambiguation pathology.
Unterminated templates, links, refs, and tables fault loudly; the
document stage catches per page and files audit rows instead of
letting a broken page leak.

## fn scan_segments

The shared template/link argument scanner: raw spans between
top-level pipes, `{{ }}` and `[[ ]]` depth-tracked, comment- and
nowiki-blind (both can contain pipes and closers), newline-crossing.
Args stay RAW wikitext by design - the template family handlers
re-parse a value only when its family needs it, so nothing is lost
to premature interpretation. Named args split at the first top-level
`=` and trim both sides, positional args stay verbatim - the
MediaWiki conventions.

## fn parse_link

The linktrail is the ASCII-lowercase run after `]]` (the enwiki
rule), captured as its own field because it is the display/resource
anchor rule's exact mechanical source: `[[practice]]s` displays
"practices" against target "practice".
