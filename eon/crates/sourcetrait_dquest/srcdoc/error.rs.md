# error.rs

## enum DquestError

One variant, and the shape of the enum is the point rather than an
accident of there being little to go wrong yet.

### Cert

The source is boxed, and that is a lint rather than a taste. `CertError`
keeps its own sources rather than flattening them to messages, which
makes its largest variant about 144 bytes, and an unboxed copy inside
every `DquestResult` trips `clippy::result_large_err` at the standing
zero-warning bar. Boxing costs one allocation on a path that is already
failing.

## What is deliberately NOT here

An earlier cut carried a `CertificateOwed` variant whose `Display` was
the whole five-line instruction shown when no certificate exists. It was
removed rather than reworded.

The reason is the output convention. A resource named in a message is
cyan, and colour is applied where a message is EMITTED so that no error
type ever carries escape codes - which are neither comparable nor
lockable, and leak wherever the error is logged. Those two rules cannot
both hold for a message built inside a `Display`: either the path goes
uncoloured, or the error starts carrying escapes.

Composing that message at the print site satisfies both, so it lives in
`run` as `certificate_owed`, and its command spellings are locked there.
An unmet precondition is now a startup OUTCOME rather than an error,
which also reads more honestly - nothing failed, the operator simply owes
the daemon a certificate.
