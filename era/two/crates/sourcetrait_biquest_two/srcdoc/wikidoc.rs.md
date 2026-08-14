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
