# template.rs

Liquid rendering and the nu value bridge under it. Liquid is a first-class
customer of this crate rather than a preprocessing step bolted to one consumer,
which is why it sits beside the data modules rather than inside ThinkHarness: the
plugin, the daemon and the trainer can all render, and where a given template
renders is an engineering choice per case rather than a fixed side of the wire.

The module is named `template` rather than `liquid` because a module sharing a
name with a crate it uses makes the identifier ambiguous under the star import
every other file relies on.

## fn binding_name

MEASURED, and it is a deviation from the design's worked example that is worth
knowing about rather than discovering. The design writes a template as
`{{ $in.their_name }}`, so that Liquid's variables arrive through the same
`<pass>` binding that feeds `<nu>` and there is no second namespace to learn.
Liquid 0.26 does not parse that: the `$` is rejected in a variable path, and a
probe over the four spellings in the design confirmed it before anything was
built on them.

The adaptation is the smallest one available and preserves the intent. The
channel keeps its name and loses the sigil, which belongs to nushell rather than
to Liquid, so `<pass>$in</>` binds a template variable called `in` and
`<pass>$args</>` binds one called `args`. It is still the same channel and still
not a bespoke namespace; it is the channel spelled for the language addressing
it, exactly as `$args` already names a positional literally called `args`.

Rewriting `$in` to `in` inside the template before parsing was the alternative
and is worse: a template is arbitrary text, so a literal `$in` in prose would be
mangled by the very rewrite meant to help it.

## fn render

Undefined variables are a hard error rather than an empty render, which is the
crate's behaviour rather than a choice of ours and is the better one here. Jinja
renders empty and would let a template silently produce a hole; a renderer whose
output becomes training data wants the hole to be loud. Every binding a template
references must therefore actually be bound.

## fn to_liquid

Seven nu types map onto Liquid's own data model directly. Everything else -
durations, filesizes, dates, cell-paths, globs, ranges - carries its NUON
spelling instead, which is the typed literal a reader of this program already
expects to see and is exactly what the value would render as anywhere else in
the toolset.

The map keys are built owned rather than borrowed. A borrowed key does compile
at first glance and then fails with a lifetime escape, because the object
outlives the binding slice it was keyed from.
