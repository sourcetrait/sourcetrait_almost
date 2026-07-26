# lib.rs

The manifest, carrying no comments by convention.

Camp is a PURE BRIDGE CONSUMER: it depends on the API crate for the contract
and on an era's implementation crate only to name a concrete `Era` at startup.
Nothing here reaches into an era's internals, which is the property that lets
the same binary point at a different era later.

The two bridge crates are aliased short - `bridge` for the API, `bridge_two`
for the era - so a reader can tell at any call site which side of the seam
they are on. That distinction is the whole reason the aliases exist rather
than importing the items directly.

## mod r

The re-export hub. `mpsc` names the channel pair the bridge session is built
from, and `term` gathers the crossterm surface, which is deep enough under
ratatui that unqualified paths would be unreadable.
