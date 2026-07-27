# tls.rs

Both configurations from one set of installed material, which is why they
share a file rather than splitting by end. Our authority is the only root
either side trusts and the same leaf is presented at both, so a reader
changing one and not the other would be changing half of one decision.

One leaf serving both ends is bounded rather than general. Admission is
the certificate authority rather than the leaf's identity, and a
Thinkspace is keyed from the username rather than from anything in the
certificate, so on a loopback deployment where client and server are the
same machine one leaf is sufficient. A remote client would want its own
identity and therefore its own leaf, which is a profile carrying several
entities. The check that forces that is the first non-loopback
deployment.

The premise this file was designed against was refuted by measurement, and
the refutation is worth carrying because it was cheap and nearly skipped.
The plan was to reuse another server's certificates; that leaf carries
`TLS Web Server Authentication` alone, so a conformant stack refuses it as
a client certificate and mutual TLS cannot be stood up from it at all.
Read the extension off the artifact rather than off the profile that asked
for it - `quest show` reports what was requested and the certificate is
what a peer will refuse.

## const LOOPBACK_NAME

## const LOOPBACK_ADDRESS

One constant rather than one per end. A client and a server disagreeing
about where the rendezvous is produces a connection refused, which reads
as a daemon that is not running rather than as two numbers that differ -
so the fault is made inexpressible instead of diagnosable. It sat in the
daemon until a client needed it, which is the moment a private constant
stops being private.

## const PROFILE

## const SECRET_DATA_ENV

The same argument one level up. Both ends present the same leaf, so both
have to name the same profile and find it by the same variable, and the
transport is the only place either end already depends on.

`SECRET_DATA_ENV` is a sourcetrait extension rather than an XDG variable,
so nothing else on a box sets it and there is no platform default to fall
back to. That is why an absent value is an error rather than a guess:
neither end can know where the material lives, and inventing a location
would put a lookup somewhere nothing installed to. It is also the
variable `srcert` itself reads, so the tools agree without either telling
the other.

## fn material

The four installed artifacts, as the profile that owns them names. The
certificate library owns the path layout, so a consumer reconstructing a
secret directory plus four filenames would duplicate exactly the knowledge
that library exists to hold.

## fn secret_data_home

Blank is the same condition as absent, and treating them alike is the
point rather than tidiness. A variable exported empty is the ordinary
result of a shell assignment that resolved to nothing, and an empty
string joined to the layout below it produces a relative path - so a
consumer would look for its certificate under whatever directory it
happened to be started from, find nothing, and report missing material
rather than a misconfigured environment.

## fn installed_material

The whole of what a consumer needs to present this transport's identity,
in one call. Both ends resolve it the same way because both present the
same leaf, and a consumer reconstructing the home, the profile and four
filenames would duplicate exactly the knowledge this crate exists to
hold.

The daemon keeps its own resolution rather than calling this, and that is
not duplication left in by accident. It has a setup conversation to hold
with an operator - which variable is unset, which certificate is owed -
and that needs its own error type, where this returns the transport's.

## fn client_config

Client authentication is the point rather than an option, and this is the
one place a single-sided example misleads. `with_no_client_auth` is the
client half of a server-authenticated connection; copied across it comes
up working and unauthenticated, which is the failure that looks like
success. The leaf carries both usages precisely so it can be presented
here.

## fn server_config

Requires a client certificate from our authority through an explicit
verifier. The root store is the same one the client trusts, so the two
functions are one decision read from either side.

## fn loopback_server_name

## fn provider

Ring, named rather than inherited, and this is the load-bearing line in
the file. rustls resolves a provider from process-global state when none
is given, so a library depending on that depends on install order - which
is not something a library gets to control and not something a consumer
should have to know.

The dependency side of the same decision is that rustls 0.23's default
features pull a second backend, and `prefer-post-quantum` pulls it again
on its own. Take rustls and tokio-rustls with default features off and
`ring` named, and re-read the lock afterwards - that is the only thing
that actually catches a second backend, because the failure is silent
rather than an error.

## fn builder

## fn server_builder

Two functions rather than one generic because the two builder types do not
unify. That is a rustls fact rather than a choice, and it is recorded so
the next reader does not spend the attempt.

## fn authority_store

Our authority and only our authority. No platform roots are added, so a
certificate from any other issuer is refused however well the system
trusts it - which is what makes the admission check mean something on a
machine whose trust store we also install into.

## fn leaf

PEM loading needs no extra crate: `rustls_pki_types` is already in the
graph and carries `from_pem_file`, so `rustls-pemfile` is a dependency
that looks required and is not.
