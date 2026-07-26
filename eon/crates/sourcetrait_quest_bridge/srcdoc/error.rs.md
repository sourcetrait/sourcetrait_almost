# error.rs

## type BridgeResult

## enum BridgeError

ONE VARIANT, deliberately. This is the era-agnostic error surface every era
raises through, so a structured variant set here would be a taxonomy imposed on
eras that do not share failure modes - and the eras are complete rewrites of
each other.

The consequence a consumer should hold: the message is the contract. A
`BridgeError` is for showing and logging rather than for matching on, and camp
wraps it transparently for exactly that reason.
