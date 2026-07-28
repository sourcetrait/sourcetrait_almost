# railroad.rs

A railroad is where one training run's generated data lives, and it is a
git repository rather than a directory. That choice is the whole design.
A run's data is throwaway by default, and being versioned means it does
not have to be. It also makes editing a generated set in place a
first-class move - change it, commit a REV, re-run - and the history then
answers what actually changed between attempts, which is exactly the
question the second training attempt could not answer about itself.

## const TMP_HOME_ENV

The throwaway home, rather than a cache or data home, because a railroad
is bulk we have no intention of keeping. It sits a tier below offload in
the permanence gradient: offload is bulk we mean to keep and accept
losing, while this regenerates from templates and fills.

Reading an `XDGX_` variable from shipped source has precedent in the
bridge's own secret-data home, so it is the estate's convention rather
than a harness path leaking into the product.

## const RAILROAD_RELATIVE

Two segments under the tmp home rather than one, so railroads sit beside
whatever else quest parks there rather than colonising its root.

## const BRANCH

## const INIT_SUBJECT

## const BASE62

This is duplicated from the daemon's own nom, deliberately. The design
asks for the same nom shape, and the shape is the contract here rather
than the code - two implementations that each emit base62 digits of a
`u64` agree on everything that matters without sharing a line. Both
alternatives are worse: the library cannot reach into an eon crate at
all, and hoisting the daemon's primitive down here would be a refactor of
the daemon in service of a training feature.

The alphabet is the validation. It carries no separator, no dot and
nothing a path resolves, so a nom is a plain path segment by construction
rather than by being checked afterwards.

## struct TrainNom

### fn fresh

The clock is hashed rather than spelled, and the reason is not secrecy. A
raw nanosecond count is already a plain segment, but it is
variable-width, it sorts by wall time, and it advertises when a run
happened in its own path. Hashing gives it the fixed base62 shape every
other nom in the estate has, at no cost. xxh3_64 is the design's named
choice rather than a substitution of mine, and nothing here is doing
security work, so a non-cryptographic hash is the honest fit.

Two runs minted in the same nanosecond collide, and that is safe rather
than merely unlikely. `lay_in` refuses a directory that already exists,
so a collision is a raised error and never a second run laying track over
the first. The distinctness lock exercises the ordinary case; the refusal
is what makes the pathological one harmless.

### fn as_str

## impl Display for TrainNom

## struct Railroad

### fn home

The tmp home is required and our own leaves are created, which is the
looked-up-versus-invented rule applied precisely. The tmp home is an
assertion the environment makes, so the only correct operation against it
is a check - conjuring it would turn a failed assertion into a fabricated
success and leave a railroad somewhere nothing reads. Everything below it
is a path we named ourselves, so `lay_in` creates it on demand.

An unset variable and an empty one are reported separately from a
variable naming something that is not a directory, because the operator
actions differ.

### fn lay

### fn lay_in

The order is the point, and it is preflight before mutation. The identity
check runs before the nom is minted and before the leaf exists, so a box
with no configured git identity gets a clean refusal rather than a
half-laid repository, a burned nom, and a commit failure two calls later.
Creating the root first is safe because the root is ours and the call is
idempotent.

The base commit carries an empty `.gitignore` rather than being empty
itself, so the branch has a real tree to grow from and a caller has an
obvious place to put ignore rules.

### fn at

Taking the directory name as the nom needs no validation, and that is a
property of `file_name` rather than an omission. It yields one component,
so the result cannot contain a separator whatever the caller passed -
which is the same construction argument the alphabet makes from the other
end.

What is checked is that a repository is actually there. An ordinary
directory would otherwise accept commits into a railroad that was never
laid, numbered off a history that does not exist.

### fn nom

### fn dir

### fn commit

### fn next_rev

The number is derived rather than passed, so a caller cannot get it wrong
and two processes committing to one railroad cannot disagree about where
the history is. `REV N` is therefore the N+1th commit, since the base
commit is `init` and is not a REV.

It counts the branch rather than `HEAD`. The two are the same thing on a
railroad nobody has moved, and they differ exactly when something has, so
naming the branch keeps the numbering anchored to the design's own
history rather than to wherever a caller happens to be standing.

## fn identity_ready

No identity is invented here, and that is deliberate rather than
unfinished. Who authors a run's commits is a decision about the product,
and inventing a name and address to make a commit succeed would bake one
in without anybody having chosen it. What this does instead is move git's
own failure earlier, to the point where nothing has been created yet.

## fn git

Git's stderr is the whole diagnostic. Nothing here interprets a git
failure or maps it onto a vocabulary of our own, because git's messages
are better than anything this could paraphrase and a reader who sees one
knows what to type next.

## fn base62

Zero is spelled rather than falling out of the loop, since the loop emits
nothing for it and an empty nom would be the one value that is not a
plain segment.
