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
