# error.rs

## enum QuestCoreError

Two variants where the era library carries five, and the difference is the
crate's reach rather than an omission. Nothing here touches candle, serde_json
or toml, so those source types cannot arrive; what remains is filesystem IO and
the whatever arm the nu and NUON failures raise through.

Consumers wrap this as a transparent source variant rather than converting it,
which is what keeps `?` working unchanged at the hundreds of call sites that
reach these functions through the era library's re-export.
