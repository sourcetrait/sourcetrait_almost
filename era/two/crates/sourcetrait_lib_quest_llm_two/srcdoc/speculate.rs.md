# speculate.rs

The era-one design carried whole: the prompt-lookup gram index over the
committed context, the longest-first ladder probe, and the adaptive draft
policy. Pure HOST machinery - the model-side verify and rollback live in
`generate.rs` and the carried-cache shadow surface.

Both types are public so the offline replay instrument drives THIS code rather
than a copy of it. That is what makes the recorded policy verdicts statements
about the shipped engine.

## const LADDER

Probed longest-first because a longer matched context is a higher-precision
draft source. The 2-gram floor raises match rate on the target workloads -
schema-shaped output, repair loops, documentation text - and its precision cost
is paid back through the policy's n-seeded lengths rather than by dropping the
level.

## const NGRAM_MAX

The level rides IN the key, so padding to one width cannot collide across
levels.

## const MAX_DRAFT

16 beat 8 by about 17 percent on echo-shaped decode in era one. The tracking
ceiling only reaches it on deep accepts, so partial-accept workloads never pay
for the wider span.

## const PAUSE_ZERO_STREAK

## const COOLDOWN_ROUNDS

## struct GramSpots

Two spots rather than one, because the tail's OWN insertion always occupies
`latest`. Without `previous`, a gram whose only earlier occurrence is the
current tail would be undraftable - the index would be describing the present
rather than the past.

## struct LookupIndex

### fn gram_key

### fn new

### fn extend

### fn draft

The source selection is the subtle part: when the tail gram is unique so far,
`latest` IS the tail's own insertion and must be skipped in favour of
`previous`. Drafting from the tail's own position would continue the context
with itself.

## struct DraftPolicy

The pause is what makes novel output converge to plain-greedy cost. Without it
a run with no repetition pays a verification forward on every false hit
forever; with it, drafting stops and the workload is bounded by the ordinary
path.

### fn new

### fn probe_limit

Resuming from a pause returns the ceiling to ONE rather than to the maximum, so
a resume probes cheaply and regrows only if accepts justify it.

### fn draft_limit

### fn record

The ceiling tracks TWICE the observed accept depth rather than growing by a
step. Echo-shaped runs climb to the maximum quickly, while partial-accept
workloads hover near their true depth instead of repeatedly paying for
full-span rejections.
