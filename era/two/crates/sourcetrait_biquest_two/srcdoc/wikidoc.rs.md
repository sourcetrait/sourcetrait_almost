# wikidoc.rs

The document renderer, fixture-anchored: the microsoft standard
(TheUser's cleaned document, mechanically normalized to the ruled
typography) is the acceptance surface, and the render is
byte-identical to it.

## Policy, ruled

- Source italics flatten everywhere, in-passage work titles included
  (TheUser's cleanup is the ruling); the renderer's own citation
  titles are the only italics, and source bold survives as the quote
  target.
- Output levels are kind-based and uniform across single- and
  multi-etymology pages: every top section h2 (Etymology N
  included), headwords and POS subsections h3. bank proves the deep
  form.
- The IPA accent joiner is a uniform ", " - the fixture's mixed
  joiners were the draft's own artifact, made uniform there too.
- Page-set merge order: the word's own casing first, the capitalized
  form second, the rest title-sorted; the first page's title is the
  h1 (linux = pages linux + Linux under "# linux").

## Faithfulness boundaries

Anchor targets stay verbatim - they are resource addresses the
resolution layer matches against real titles - while every prose
string passes normalize_typography. Sense-scoped synonyms strip
their angle-bracket inline qualifiers (the fixture shows bare
anchors); dropped sections, refs, external links, plumbing
templates, and unknown accent codes all file audit rows so nothing
disappears silently.

## The head-argument grammar (fn head_line and the engines)

Designed against the pinned dump's own template documentation
(Template:en-noun/en-verb/en-adj/en-adv/head documentation pages,
fetched from the snapshot) plus a measured 100k-word audit
distribution; the audit details carry full template signatures so
the families keep growing audit-driven. The dominant class was the
generic head|en|<pos> form at 79% - non-lemma form pages - which is
why bare `head` is handled silently rather than audited.

Load-bearing details the doc pages pin:

- The default derivations are the en-verb exact rules: C*VC (the
  whole lemma consonants + one vowel + one final consonant not in
  w/x/y/h) doubles before -ed/-ing, -ie becomes -ying, -ue and
  consonant-e drop the e, vowel-e (toe, see, dye) keeps it. The
  en-adj doc omits doubling for -er/-est but the module doubles, so
  graded_form does too.
- Slot-one special indicators (^ ++ +l +! +' and the * multiword
  forms) become the defaults for later verb slots; `+` is always
  equivalent to a blank slot; slot 4 absent or past-equal folds
  into the combined piece, and an explicit `-` (defective) leaves
  the past standing alone - absent and defective are different.
- A lone `~` is the noun countability marker; anywhere else `~`
  substitutes the lemma - the substitution must skip the marker or
  countability silently breaks.
- The angle-bracket verb format is detected by the first `<`'s body
  NOT being an inline-modifier prefix (l:/ll:/q:/qq:/ref:); only
  the single-bracket forms are handled and the alternant/multi-
  bracket residue audits with its signature.
- Head-family templates are recognized by NAME (head, en-*), not
  position, because wp precedes the head on many pages and was
  eating the head slot - the wp family routes to its own
  wikipedia_pointer audit class, kept as the word-to-article signal
  for the spidering service.
- Unknown named arguments audit the whole signature and contribute
  nothing: a wrong derived form is worse than a visible gap.
- Numbered mechanics: |1=x| named-numeric arguments fold into their
  positional slots (the MediaWiki equivalence) before anything else
  reads the template, and trailing-digit named keys collect into
  their base families - past2=/pres_ptc2=/pres_3sg2=/past_ptc2= as
  additional verb-slot forms, sg2=/attr2= and sup2= likewise.
- The pairs engine is shared: `head` skips its language and POS
  positionals; en-pron/en-pronoun pair from the start and append
  desc= as a trailing note; en-head is pairless (its doc: positional
  1 is the POS, nothing derives), so bare en-head|<pos> is handled
  silently and extra positionals audit. A head= of `?` (the
  no-good-headword sentinel) is ignored rather than rendered.
- SpecForm::rendered flattens embedded wikilinks to display text -
  the whole inflected form is the anchor.

## The form-of family (fn form_of_name and the shape rule)

The definitional templates render by NAME SHAPE, not enumeration:
any "<label> of" name becomes "<Label> of [target]" - the rule that
absorbs plural of (278k), the alternative form/spelling family, and
every sibling without a case list, the build-for-an-LLM freeze-form
posture. Aliases expand from vendored data (the same dump-versioned
carriage as the accent map); `form of` itself carries its label as
an argument and is special-cased; t= glosses parenthesize; nocap=
suppresses the capitalization. confix is split from the plain affix
join because its semantics hyphenate the ends (prefix + suffix).
Headword overrides pass through plain_anchor_text - heading lines
carry no anchors. Empty documents (fold-matched page sets with no
English content) are the corpus emitter's .empty.md convention with
the document_empty audit class - skippable by suffix, existence
known (TheUser).

## The etymology reference family (fn ety_reference and kin)

Designed against the snapshot's own template documentation
(derived/borrowed/inherited/cognate and the +/lbor/ubor/doublet/
calque/unknown/surface-analysis pages, fetched from the pinned
dump). One engine carries the whole family because the grammar is
one grammar: a source-language slot (comma-listable, conj=-joined),
a term, an alt display, a gloss at t=/gloss=/the next positional.
The der/bor/inh family carries the entry language in slot 1 and the
source in slot 2; cog and m+ drop the entry language, which is the
langs_slot parameter. The complete-wording variants (bor+, inh+,
der+, lbor, ubor, slbor, obor, calque, doublet, unknown) differ
only in prefix, notext=, and nocap=. uder renders exactly as der -
its difference is a cleanup category. A term of `-` or empty shows
the language name alone (the term-request UI is wiki maintenance,
not document content).

Language codes resolve through the vendored map; an unknown code
renders verbatim and audits language_code_unknown - the accent-map
growth pattern. The legacy dotted codes (LL., VL.) are gone from
the snapshot's own data (the canonical-name JSONs and the etymology
data module carry name aliases only), so they resolve nowhere and
the audit is the correct outcome.

## etymon renders nothing (fn etymon_text)

The etymon template is a data carrier: on the live site it displays
NOTHING unless text= (vote-gated per language) or tree= is set, so
the ~60k ety/etymon instances render empty here with no audit. With
text=, only the one step the page itself carries can render - the
chain modes (++, *, :lang) traverse OTHER pages' etymon data, which
a single-page renderer cannot reach - so every text mode renders
the immediate step: keyword wording (the :kw derivation keywords
map to the same wordings as the standalone templates) plus its
etymons. <unc> prefixes "Possibly"; :af joins with " + "; :root and
:afeq are invisible by the template's own contract; an unknown
keyword audits etymon_keyword_unknown. tree= without text= audits
etymon_tree_dropped - a tree is a visual we cannot carry.

## The inflection-tag engine (fn inflection_of_text)

inflection of (infl of) plus its p=-presetting siblings (noun form
of, verb form of, adj form of) MUST match before the generic
" of"-shape rule, or they render "Inflection of [x]" with the
grammar tags silently dropped - the exact defect this engine
replaced. Tags resolve through the vendored map (data/1 + data/2 +
lang-data/en): shortcuts expand recursively (list-valued ones to
several tags), `//` multiparts join their parts' displays with a
slash, punctuation tags carry the documented spacing (closers
attach left, openers right, slash and hyphen both), and `;` breaks
tag sets, joined "; " with the lemma once at the end. An unknown
tag renders verbatim - the documented spell-it-out convention, the
freeze-form posture again. Comma-separated lemmas carry inline
<mod:value> modifiers; t/alt render, the rest (tr/ts/g/id/sc/pos)
drop. enclitic= audits with the signature. Capitalization follows
the form-of family convention (ucfirst unless nocap) for corpus
uniformity, though the site renders these lowercase.

## The name family (fn surname_text, fn given_name_text)

Assembled from Module:names' own display builders, read from the
snapshot - the piece order, comma discipline (the first qualifying
piece attaches bare, later ones take the comma via need_comma), and
the from= grammar are the Lua's: category keywords (surnames/given
names/nicknames/place names/common nouns/month names transfer "from
the <singular>"; patronymics/matronymics/coinages originate "as a";
occupations/ethnonyms "as an"; the Bible "from the Bible"), language
and family names render bare ("from French", "from the Slavic
languages"), code:term references render "from <Language> [term]",
and " < " chains render the tail as in-turn steps - parenthesized
here where the Lua brackets them, because square brackets are the
anchor syntax. eq= defaults terms to English and always shows the
language name (the Lua's include-langname join); m=/f=/varof= and
kin default to the entry language and show it only when foreign.
The gender article: unknown-gender takes "an", otherwise the
adjective decides, otherwise "a" - and given-name genders include
the animal set, which renders "for a dog" after the noun rather
than "dog given name" before it.

## The place engine (fn place_text and kin)

Designed against Template:place/documentation plus the two data
modules (placetypes, locations), all from the snapshot. The
vendored data carries exactly the render keys: placetype aliases,
qualifier displays with their article overrides (largest -> "the",
several -> none), per-placetype preposition/affix/fallback/
holonym_use_the resolved through the fallback chain at load, the
placename article rows and translated the-patterns (Lua ^/$
patterns become prefix/suffix rows; [Rr] classes expand), and the
locations that carry article or alias data. Location display
semantics were measured the hard way: display = true means the
display CANONICALIZES to the alias target (c/USA renders "United
States"); a string value displays that string; a bare alias_of
categorizes only and keeps its written form (c/Czechia renders
"Czechia") - the first reading (keep-display flag) was backwards
and two tests caught it.

The comma algorithm is the documented one: no comma before the
first holonym, none after raw text, none before and/in or a *-led
piece, comma otherwise. The placetype's preposition inserts only
when a holonym directly follows the placetype - raw text carries
its own preposition ("in central", "of") and suppresses both the
insertion and first-position "the". "the" attaches to a holonym
only in first position, within a grouped name list past the first,
or under :pref/:Pref/:the modifiers; affix defaults (oblast ->
"Oblast" Suf) skip when the name already contains the affix word.
`;` restarts keep their article lowercase (mid-sentence). The
single-spec format re-renders text runs through spaced_fragment
because inline rendering trims - the boundary spaces around
<<markers>> would otherwise fuse. Extra-information tails (capital=,
caplc=, seat=, ...) render as "; label: [X]" lists. @-directives
audit with the signature.

## The straggler set

Mechanical classes the audit tally surfaced beside the families:
{{...}}/nb... render "..." (the quotation elision; bracketed
on-site, but brackets are the anchor syntax here); quote-* books
cited inline (etymology prose) route through the same citation
engine as #* lines; ux/uxi/usex render as quoted example lines
under #: and #* and inline as quoted text; sense renders its
"(gloss):" prefix; taxfmt anchors (its precondition is an existing
entry) while taxlink renders plain (its precondition is a missing
one); cap/U anchor a capitalized display against the lowercase
target; same-page #section links render display text alone and
mid-target sections strip to the page name. SILENT_TEMPLATES is
the handled-as-nothing set, and the three template positions (in
prose, POS paragraphs, list-section paragraphs) all consult it -
the C/cln/topics class was auditing template_unhandled from
paragraph positions while inline occurrences were silent.

## The accent-code map

Third-party data we do not control, so it is carried, never
hardcoded (TheUser's ruling): vendored NUON under a dump-date
version directory (data/accents/20260801/), embedded and parsed
once through a OnceLock - the Syntax.nuon pattern applied to a
foreign vocabulary, with CC BY-SA provenance in the file header.
Grown from the measured corpus distribution (49,067 unknown rows,
the top 40 codes ~86% of volume) against the snapshot's own label
data, Module:labels/data/lang/en - Module:accent qualifier/data is
gone from the snapshot, deprecated into the labels system. Three
mechanisms: ACCENT_NAMES maps codes whose display differs
(abbreviation expansions like GenAm/SSB/CanE, merger displays -
square-nurse IS the fair-fur merger and Mmmm the Mary-marry-merry
merger in the module's own naming, æ-tensing is æ-raising);
ACCENT_VERBATIM lists labels the module names for themselves
(render verbatim, no audit - non-rhotic is its own label, not a
negation); and a non- prefix negates any resolvable base ("without
the X merger" / "without X"). The unresolved tail keeps auditing -
the map stays audit-grown.
