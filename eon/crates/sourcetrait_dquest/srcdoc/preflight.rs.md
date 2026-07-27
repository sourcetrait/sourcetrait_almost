# preflight.rs

## const PROFILE_TEXT

Included at compile time rather than read from disk, so the binary always
knows what it ships even when nothing is installed yet. That is what lets
a first start be self-sufficient: there is no bootstrap file to find, and
no ordering problem between placing the profile and reading it.

`install_profile` parses the text before either write, so a malformed
shipped profile fails at the first start rather than at the first
handshake. The parse is also locked in-crate, which is the earlier of the
two and the one that fails during a build rather than in front of a user.

## enum Profile

Carries the live path in both variants rather than only in the written
one, because the caller wants to name the file either way - to tell the
operator where to mint from, or to report which profile a run is using.

## fn ensure_profile

Called on every start rather than guarded behind `profile_exists`, and
the asymmetry inside `install_profile` is why. The live copy is written
only when absent, so an unconditional call cannot overwrite a
customisation; the reference copy is rewritten every time, which is its
whole job, since it is what a customised profile gets diffed against and
a stale one is worse than none. Guarding the call would silently freeze
the reference at whichever build first ran.

WHAT THIS DOES NOT CHECK, and it is the known hole rather than an
oversight. The precondition is the PROFILE, not the material. A second
start after a first one finds the profile present and proceeds, even
though no certificate has been generated or installed - so the check
keeps certificate generation an explicit act only on the very first run.
Once the bridge actually serves TLS that becomes a handshake failure
rather than a startup failure, which is the worse of the two places to
learn it.

The library already carries what would close it: `secret_certs_dir` gives
the install destination, and the four artifacts under it are what
`install` places. Adding the check needs the secret data home, which the
command line reads from the environment, and it widens the certificate
library's public face, so it is a decision rather than an omission.
