# tokenizer.rs

## fn load_tokenizer

## fn verify_token_map

A runtime lock over the twelve ids the engine leans on, checked once at load so
that a checkpoint swap fails loudly here rather than as wrong channel routing
much later.

The BASE checkpoint FAILS this deliberately. Its extra-id block sits at
different positions and it carries no tool markers at all, so a base load is
not a degraded mode to tolerate - it is a different vocabulary, and the sole
engine target is the DPO artifact.

The twelve are the anchors and boundaries rather than the whole added-token
map: the channel anchor, both turn markers, both stop tokens, the four tool
markers, the two ends of the reserved extra-id run, and the pad. Checking the
run's ends rather than every member is what keeps this a lock rather than a
second copy of the map.
