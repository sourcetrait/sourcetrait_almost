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
string passes normalize_typography. Head templates derive the
REGULAR inflections only when bare; any argument files an audit row
and omits the parenthetical, because a wrong derived form is worse
than a visible gap - the argument grammar grows audit-driven, the
same policy as the template families and the accent-code map.
Sense-scoped synonyms strip their angle-bracket inline qualifiers
(the fixture shows bare anchors); dropped sections, refs, external
links, plumbing templates, and unknown accent codes all file audit
rows so nothing disappears silently.
