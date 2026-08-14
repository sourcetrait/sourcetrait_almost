# wikixml.rs

The XML page layer under the WikimediaDumpTool: one reader shape for
export saves, whole dumps, and index-seeked multistream blocks.

## struct PageReader / fn next_page

quick-xml 0.41 delivers entity references as their own events
(`Event::GeneralRef`, the 1.0-lineage change), so a text node's value
accumulates across Text, CData, and GeneralRef events. The
plausible-but-wrong first reach is `BytesRef::xml10_content`, which
returns the reference's own content with end-of-line normalization
rather than its expansion - measured as "ATampT" for "AT&T". The
resolver is `resolve_char_ref` for numeric references plus the five
predefined entities by name, which is the complete set for the
DTD-less export and dump XML.

Fields are addressed purely by element-stack position, which is what
disambiguates the three id elements (page, revision, contributor) with
no name lists. The stack clears when a page opens, so a reader started
mid-document (a multistream block, or after a prior page) needs no
depth context. `check_end_names` is off because a multistream block is
a rootless page sequence.

## fn open_pages / fn read_block

`MultiBzDecoder` decodes across every concatenated stream - the
whole-dump path; `BzDecoder` stops at one stream's end, so seek plus
`BzDecoder` reads exactly one ~100-page multistream block. bz2 is
detected by extension so the same open serves plain export saves.

## fn parse_index_line / fn index_find_title

Index lines are `offset:pageid:title` with the title free to carry
colons, so only the two leading fields split. The index is page-id
ordered, not alphabetical: a title scan's cost tracks the page's age -
Microsoft (id 91037) resolves in ~54 ms while a young page costs the
full decompressed scan (~250 MB for enwiktionary). The real resolution
artifacts land with the TitleResolution slice; this scan is the test
surface. Measured on the 20260801 pin: the dump's Microsoft page is
byte-identical in size and id to the live export fetched two weeks
later.
