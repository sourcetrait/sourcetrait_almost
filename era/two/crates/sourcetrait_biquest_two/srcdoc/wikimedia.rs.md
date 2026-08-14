# wikimedia.rs

The WikimediaDumpTool's verb family. `page` is the raw test surface:
it prints the wikitext exactly as the source carries it, because the
extraction layer stays faithful by design - typography normalization,
cleaning, and rendering belong to the document slices, and auditing
those needs an untouched baseline to diff against. `--meta` swaps the
payload for the page record so a lookup can be checked without
flooding a terminal.

The title contract is exact match: enwiktionary's main namespace is
case-sensitive (Linux and linux are different pages), so no folding
happens here; fold-aware word-to-page-set resolution is the
TitleResolution slice's job. The error message carries that fact
because a case miss reads exactly like an absent page.
