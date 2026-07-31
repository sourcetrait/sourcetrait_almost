# nu.rs

The one owner of the nushell dependencies in the whole toolset. Every other
crate reaches this module and never the nu crates directly, so the estate's
single nushell pin has exactly one place to move.

NOTHING IN IT IS ERA-SPECIFIC - there is no model here, no candle and no
checkpoint - and it lives in the library anyway. It sat in a crate of its own
once, so that an eon binary could speak the program's data language without
linking an era's engine. That justification did not survive measurement: a
binary carries only what it CALLS, so a consumer that never drives the engine
never linked it, and every eon binary already depended on the library
regardless. The module path never moved either way, so `lib::nu` resolves as
it always has.

NUON is the program's primary data format. Artifacts we author are `.nuon` -
one whole value per file, usually a table or a record - while stream-shaped
outputs are NUON LINES, one record literal per line, which is the role JSONL
serves elsewhere. Foreign tool seams keep their own formats at the boundary.

## fn parse_typedef

Rides a SYNTHETIC CLOSURE SIGNATURE because nu-parser exposes no public bare
type-parse at all: the typedef is spliced into `{|x: <t>| null}`, parsed
decl-free on a bare engine state, and the positional's shape carries the type
back out. That is not obvious from anything in the API and is the reason this
function is longer than it looks like it should be.

The engine state needs `$env.PWD` seeded before the parse. The nuon crate
performs the same seeding for its own parser, so this is upstream's expectation
rather than ours.

Sugar note that surfaces in every rendered typedef: `path` and `directory` are
valid spellings that COLLAPSE to string in the Type enum, so a derived typedef
shows the base type. Conformance is unaffected because string values conform;
`glob` survives as its own type.

It parses, which matters to any host that calls it. Parsing recurses and its
stack cost scales with nesting depth, and a stack overflow in the parser is a
fatal abort rather than a catchable error - so a caller must not run this on an
async runtime worker, whose stack is roughly a quarter of a default thread's.

## fn from_nuon_text

## fn to_nuon_text

## fn to_nuon_condensed

## fn to_nuon_pretty

Two spaces is not a choice. NUON pretty indents two and nushell source indents
four, neither borrows from the other, and two-space nushell found anywhere is a
mistake rather than a convention.

## fn render_nuon

## fn conform

Deep conformance through nu's own subtype machinery rather than a checker of
ours, so the answer matches what a nushell pipeline would decide: a union
accepts any member, an empty list binds any list or table, and records stay
OPEN so extra fields pass while a missing declared field is rejected.

## fn load_value

## fn save_value

## fn load_lines

## fn save_lines

## fn append_line

Appending per record rather than writing at the end is what makes the lines
form crash-safe: a killed run leaves its progress on disk.

## fn render_line

The single-line assertion is load-bearing rather than defensive. `to nuon` does
NOT escape a newline inside a string value - it emits the byte raw - so a
record carrying one would span two lines and silently become two records to any
reader. Guarding at the one render point covers every writer.

## fn ensure_parent
