# error.rs

## enum DquestError

### Cert

The source is boxed, and that is a lint rather than a taste. `CertError`
keeps its own sources rather than flattening them to messages, which
makes its largest variant about 144 bytes, and an unboxed copy inside
every `DquestResult` trips `clippy::result_large_err` at the standing
zero-warning bar. Boxing costs one allocation on a path that is already
failing.

### CertificateOwed

A failure variant rather than an ordinary outcome, so that the operator
sees it on stderr and the process exits non-zero. That matters because
the daemon is launched as a background process: a zero exit with no
daemon listening would read as a successful start to whatever launched
it, and the whole point of the precondition is that certificate
generation stays an explicit act rather than something the daemon quietly
does for itself.

The message carries both commands because they are always run as a pair
and the second is useless without the first. It names `<dir>` rather than
suggesting one: a suggested path would be this machine's layout leaking
into shipped source, and the staging directory is consumed by `install`
anyway, so it is genuinely the operator's choice.
