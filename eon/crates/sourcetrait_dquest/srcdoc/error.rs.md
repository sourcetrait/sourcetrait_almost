# error.rs

## enum DquestError

Two variants, and what separates them is who has to act: one names a
certificate operation that failed, the other names an environment the
operator has not set.

### Cert

The source is boxed, and that is a lint rather than a taste. `CertError`
keeps its own sources rather than flattening them to messages, which
makes its largest variant about 144 bytes, and an unboxed copy inside
every `DquestResult` trips `clippy::result_large_err` at the standing
zero-warning bar. Boxing costs one allocation on a path that is already
failing.

### Unset

An ERROR rather than a `Started` variant, which is the opposite call to
the one made for a missing certificate, and the difference is what the
operator can do next. A missing certificate is an expected state of a
correctly configured box with a documented remedy, so it is an outcome. An
unnamed secret data home means the daemon cannot even look, so nothing
about the certificate is known and reporting one as missing would be a
finding manufactured out of a configuration fault.

The variable is `&'static str` because it can only ever be the constant
beside it. Carrying it as a field rather than baking it into the message
keeps the name in one place, so a lock asserts the message names the
variable it actually reads.

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
