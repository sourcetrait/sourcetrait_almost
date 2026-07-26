# era.rs

## struct BridgeTwo

A unit struct: the era carries no state, because the state is the session and
the session owns its own thread.

## impl Era for BridgeTwo

### fn info

The SOLE-CHECKPOINT identity, built from constants rather than from a load -
that is what makes it answerable without touching disk.

It can therefore differ from what a session actually loads, since a config
profile may point elsewhere. The loaded model's real coordinate rides the
session's `Ready` event instead, which is why a consumer that wants the truth
waits for `Ready` rather than trusting this.

### fn open_chat
