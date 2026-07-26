# error.rs

## type CampResult

## enum CampError

The bridge error wraps transparently, so a session failure surfaces with the
engine's own message rather than a camp-shaped restatement of it. Camp adds
nothing to those errors because it knows nothing the engine did not tell it.
