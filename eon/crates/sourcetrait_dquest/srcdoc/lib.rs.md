# lib.rs

The crate is a library plus a thin binary, matching camp: `main.rs` calls
`run::run` and nothing else, so the whole daemon is reachable from an
in-crate test without spawning a process.

Only `run` is public. Everything else is crate-private because nothing
outside consumes this crate - it is an executable, and a public face here
would be a contract with nobody.

## use sourcetrait_cert_lib as srcert

Aliased to the binary name rather than to a shortening of the crate name,
because every call site reads as an instruction about `srcert` the tool:
the profile it places is the one `srcert generate quest` will read. The
alias is what keeps those call sites path-referenced without a long
qualifier.

The two crates sit on different major versions of snafu - the certificate
library on 0.8 and this workspace on 0.9 - and that is deliberately
invisible here. Only `std::error::Error` crosses the boundary, which is a
standard-library trait rather than a snafu one, so `CertError` wraps as a
source in a 0.9 enum with nothing to reconcile. Aligning the two would be
tidier and is filed as debt rather than done, since nothing forces it.

## use sourcetrait_quest_bridge_two::BridgeTwo as EraBridge

The alias is the whole of era three's cost in this crate, and it earns its
place by being an alias rather than a path. `run` instantiates a generic at
`EraBridge` and every other line is generic over `bridge::all::Era`, so
swapping eras is this line plus the manifest and the daemon's own code does
not change. Naming `BridgeTwo` at the call site would have read as a fact
about that call rather than as the one binding it is.

Aliased at the LIB rather than imported where used, deliberately, because
the point is that there is exactly one of them. An import in `run.rs` would
be equally correct and would leave the next reader unable to tell whether it
was the only one.

## use bridge::all::Engine

In scope so `container` can bound its factory without naming a path in a
signature. What it is not is a definition: the trait belongs to the API
crate, and the rationale for its shape lives beside it there.

