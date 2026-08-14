# wikivocab.rs

The vendored-vocabulary layer: the language-code map, the
inflection-tag map, and the place data, each carried as dump-versioned
NUON (TheUser's data-carriage ruling - third-party vocabulary is never
hardcoded) and parsed once through a OnceLock. The renderer resolves
against these tables; the derive example (examples/derive_vendored.rs)
re-derives all three from the pinned dump's own module pages on a dump
bump.

## Sources, and what was NOT derivable

- languages.nuon merges three ready-made JSON pages the snapshot
  itself carries (Module:languages/code to canonical name.json plus
  the etymology-languages and families twins) - no Lua parsing at
  all for this map. 10,198 codes. The legacy dotted codes (LL., VL.)
  exist in NO current data - the data modules carry name aliases,
  not code aliases - so they stay unresolved by design and surface
  through the language_code_unknown audit.
- tags.nuon derives from Module:form of/data/1 + /data/2 +
  /lang-data/en: tag display forms (display= overrides with
  wikilinks flattened; else the tag name IS the display) and
  shortcut expansions - slot 3 of a tag row lists its aliases, the
  shortcuts tables add multipart ("mf" -> "m//f") and list-valued
  ("1s" -> 1, s) forms. Expansion elements re-resolve recursively
  with a depth cap.
- place.nuon derives from Module:place/placetypes and /locations:
  only rows carrying render keys are kept (a placetype with nothing
  but categorization data contributes nothing to display), and the
  Lua the-patterns translate to prefix/suffix/contains rows since a
  regex engine is not carried for four anchored patterns.

## fn PlaceTable::placetype_resolved

Resolution strips recognized qualifiers off the left first (the
module's split_qualifiers_from_placetype behavior), then walks the
fallback chain accumulating the first preposition/article/affix
seen, capped at eight hops. This is done at lookup rather than
flattened at derive so the vendored file stays faithful to the
source module's own structure.

## fn indefinite_article

The u-as-"you" words (uni-, unio-, use-, one, eu-) take "a"; the
vowel heuristic covers the rest. Module:names' own get_indefinite
article knows "unisex" the same way - the list is the practical
subset the name and place vocabularies actually hit.

## fn term_modifiers

The shared inline-modifier splitter for the <mod:value> convention
(col/syn/inflection-of/names families): balanced angle counting so
<l:<<rare>>> label syntax stays inside its value, bare <flag> forms
(like <unc>) carry an empty value, and anything else keeps its
angle brackets as term text.
