# error.rs

## enum BiquestError

Renamed from the fork's BquestError; the Candle and Shell variants
went with the capability harness and the nu sandbox. Io, Json, the two
lib transparents, and Whatever cover the tokenizer surface; module
errors wrap in as they land.
