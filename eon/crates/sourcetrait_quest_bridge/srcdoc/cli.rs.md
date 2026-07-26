# cli.rs

EMPTY ON PURPOSE, and the emptiness is the documentation.

The per-consumer-class modules were established ahead of need so that a later
specialization has an obvious home and does not get pushed into `all`, where it
would become an obligation on every era and every consumer rather than on the
one class that wanted it.

Content lands here only when something genuinely specializes beyond `all` for
quest-class consumers. Until then, an empty module costs nothing and an
absent one would cost a design argument at the worst moment.
