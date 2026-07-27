# tls.rs

Both configurations from ONE set of installed material, which is why they
share a file rather than splitting by end. Our authority is the only root
either side trusts and the same leaf is presented at both, so a reader
changing one and not the other would be changing half of one decision.

ONE LEAF SERVING BOTH ENDS is bounded rather than general. Admission is the
certificate AUTHORITY rather than the leaf's identity, and a Thinkspace is
keyed from the username rather than from anything in the certificate, so on
a loopback deployment where client and server are the same machine one leaf
is sufficient. A remote client would want its own identity and therefore
its own leaf, which is a profile carrying several entities. The check that
forces that is the first non-loopback deployment.

THE PREMISE THIS FILE WAS DESIGNED AGAINST WAS REFUTED BY MEASUREMENT, and
the refutation is worth carrying because it was cheap and nearly skipped.
The plan was to reuse another server's certificates; that leaf carries `TLS
Web Server Authentication` alone, so a conformant stack refuses it as a
client certificate and mutual TLS cannot be stood up from it at all. Read
the extension off the artifact rather than off the profile that asked for
it - `quest show` reports what was requested and the certificate is what a
peer will refuse.

## const LOOPBACK_NAME

## fn material

The four installed artifacts, as the profile that owns them names. The
certificate library owns the path layout, so a consumer reconstructing a
secret directory plus four filenames would duplicate exactly the knowledge
that library exists to hold.

## fn client_config

CLIENT AUTHENTICATION IS THE POINT rather than an option, and this is the
one place a single-sided example misleads. `with_no_client_auth` is the
client half of a SERVER-authenticated connection; copied across it comes up
working and unauthenticated, which is the failure that looks like success.
The leaf carries both usages precisely so it can be presented here.

## fn server_config

Requires a client certificate from our authority through an explicit
verifier. The root store is the same one the client trusts, so the two
functions are one decision read from either side.

## fn loopback_server_name

## fn provider

RING, NAMED RATHER THAN INHERITED, and this is the load-bearing line in the
file. rustls resolves a provider from process-global state when none is
given, so a library depending on that depends on install ORDER - which is
not something a library gets to control and not something a consumer should
have to know.

The dependency side of the same decision is that rustls 0.23's default
features pull a second backend, and `prefer-post-quantum` pulls it again on
its own. Take rustls and tokio-rustls with default features OFF and `ring`
named, and RE-READ THE LOCK afterwards - that is the only thing that
actually catches a second backend, because the failure is silent rather
than an error.

## fn builder

## fn server_builder

Two functions rather than one generic because the two builder types do not
unify. That is a rustls fact rather than a choice, and it is recorded so
the next reader does not spend the attempt.

## fn authority_store

OUR AUTHORITY AND ONLY OUR AUTHORITY. No platform roots are added, so a
certificate from any other issuer is refused however well the system trusts
it - which is what makes the admission check mean something on a machine
whose trust store we also install into.

## fn leaf

PEM loading needs no extra crate: `rustls_pki_types` is already in the
graph and carries `from_pem_file`, so `rustls-pemfile` is a dependency that
looks required and is not.
