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

## const SECRET_DATA_ENV

A sourcetrait extension rather than an XDG variable, so nothing else on a
box sets it and there is no platform default to fall back to. That is why
an absent value is an error rather than a guess: the daemon genuinely
cannot know where the material lives, and inventing a location would put
a lookup somewhere nothing installed to.

There is deliberately no flag for it, matching the crate's standing
position that configuration surface is a decision rather than a
convenience. The variable is also the one `srcert` itself reads, so the
two tools agree about the location without either telling the other.

## fn ensure_profile

Called on every start rather than guarded behind `profile_exists`, and
the asymmetry inside `install_profile` is why. The live copy is written
only when absent, so an unconditional call cannot overwrite a
customisation; the reference copy is rewritten every time, which is its
whole job, since it is what a customised profile gets diffed against and
a stale one is worse than none. Guarding the call would silently freeze
the reference at whichever build first ran.

It hands back the library's own `ProfileInstall` rather than rewrapping
it. An earlier cut carried a two-variant enum here whose whole purpose was
to tell startup whether the profile had just been written, because that
was the condition startup branched on. It is not any more - the condition
is now whether the MATERIAL exists - so the enum was carrying a
distinction nothing read, and the library already answers the same
question for whoever still wants it.

## fn secret_data_home

Splits into an environment read and a pure `named_home` so both branches
are reachable from a lock without mutating process environment, which is
unsound to do from a test thread. Same shape as taking the colour decision
as an argument in `style`.

## fn named_home

BLANK IS THE SAME CONDITION AS ABSENT, and treating them alike is the
point rather than tidiness. A variable exported empty is the ordinary
result of a shell assignment that resolved to nothing, and an empty string
joined to the layout below it would produce a RELATIVE path - so the
daemon would look for its certificate under whatever directory it happened
to be started from, find nothing, and report a missing certificate rather
than a misconfigured environment. Refusing here is what keeps those two
failures distinguishable.

## fn material_exists

The whole of the check is delegated, and that is deliberate: the layout
under the secret data home belongs to the certificate library, so
reconstructing it here would duplicate the knowledge that library exists
to hold and would drift the day a filename changes.

WHAT IT CLOSES. The precondition used to be the PROFILE rather than the
material, so a second start found the profile present and proceeded
whether or not a certificate had ever been generated. Once the bridge
serves TLS that would surface as a handshake failure at connect time
rather than a startup failure, which is the worse of the two places to
learn it.

WHAT IT STILL DOES NOT BUY. Presence is not validity. It answers
never-minted and material-deleted, and says nothing about an expired
certificate or one that no longer matches its profile. Answering those
means resolving the trust store as well, and a daemon that refuses to
start because no trust store was detected is a worse failure mode than
the one being prevented.
