# wikidoc.rs

The document renderer, fixture-anchored: the microsoft standard
(TheUser's cleaned document) is the acceptance surface, and the first
full render diffed against it in three hunks - two were the ruled
typography applied to a fixture that predates the ruling, the third
an in-passage work title whose source italics the hand cleanup
flattened (surfaced as his call; the render keeps them).

## Policy choices the fixture could not settle

- The IPA accent joiner is a uniform ", " where the fixture mixes
  "; also of Canada" with ", also of the US" - the fixture is
  internally inconsistent there, so the uniform joiner stands until
  ruled otherwise.
- Multi-etymology pages render structure-preserving (POS at h3 under
  Etymology N, headword h4), which deviates from the single-etymology
  uniform scheme (POS h2, headword h3). bank shows the deep form;
  TheUser's uniformity call is pending.
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
