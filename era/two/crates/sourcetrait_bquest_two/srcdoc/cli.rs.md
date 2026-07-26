# cli.rs

The command surface: the global profile flagset over a subcommand tree, all of it
derived rather than hand-parsed.

Two things make this file unlike the rest of the crate. Its doc comments are
product text - clap renders them as `--help` and `doc.rs` renders them as the
command tree - and the surface is deliberately never fixed, since the hierarchy is
re-categorised as the tool grows.

The 80-character cap applies here exactly as it does everywhere else, on the
ruling that cli documentation follows the same rules as the rest. A clap doc
comment being help text changes nothing: the summary stays inside the cap and
whatever it explained lives here. That works because help text wants to be short
anyway, and because most of what the longer versions carried was a rule documented
elsewhere - the profile-token resolution, the drop-never-truncate policy, the
window arithmetic - so the shorter help loses duplication rather than information.

The tree is a category, then topical levels, then an action, and re-organising it
needs no per-change approval. That latitude is why `bquest doc cli` exists: an
always-moving surface has to document itself, and a remembered surface is
untrustworthy by construction.

## struct Cli

The three global flags ride every subcommand present and future through clap's
`global = true`, which is what stops each verb from redeclaring them and drifting.

The profile tokens resolve through the library's own framework rather than here, so
the rules are one implementation: a pure snake resolves under the profile root,
anything else is a path to a component toml, and absent means the default profile
with the embedded-base fallback. The root override exists so a test posture can sit
outside the suite root entirely.

There is no ad-hoc model-directory argument anywhere in this file, deliberately. The
model coordinate comes from the config profile, so a verb cannot be pointed at one
checkpoint while its settings describe another.

## enum Command

The categories. Each one's doc comment is what `bquest doc cli` prints beside it, so
these summaries are the listing's top level rather than incidental help.

## enum TrainCommand

The stages in chain order, which is also the order they must run in: the gate, then
continued pretraining, then supervised, preference and reinforcement tuning.

Every stage after the first takes a resume path, and without it a stage starts from
the base and the chain is silently broken. That is the single most consequential
argument in the file.

## struct StageArgs

Flattened into each stage's own arguments rather than shared by inheritance, so a
posture reads the same whichever objective is running. The defaults are the
post-training ones: a peak rate of 1e-5 against continued pretraining's 2e-4, which
is why `train cpt` carries its own argument struct rather than this one.

Rank and alpha apply to fresh adapters alone. On a resume they come from the
artifact, because a resumed adapter's geometry is already decided and a mismatched
rank would load as garbage.

## struct TrainSftArgs

## struct TrainDpoArgs

The preference window wants to be well under continued pretraining's, because that
loop holds two of everything - two layer-input caches live simultaneously, since
both chains run after the paired loss block. At 512 it dies where continued
pretraining survives.

Size it from the data rather than from the default. Every candidate pads to the
window, so a short pair at a large window pays for the whole window.

## struct TrainRlvrArgs

## enum BenchCommand

## struct BenchRunArgs

## enum TaskgenCommand

## struct TaskgenAllArgs

The bench splits at the source by index, so re-generating with a different seed
invalidates comparison against earlier bench readings. Pin the seed per campaign.

## enum RolloutCommand

## struct RolloutRunArgs

Two arguments here are load-bearing rather than tunable. The group size is the unit
a relative advantage is computed over, so one carries no signal at all. And the
temperature must be above zero, because a greedy group is one reply repeated and
every advantage in it is zero.

## struct TrainCptArgs

Its own struct rather than `StageArgs`, which is why the learning-rate default
differs and why there is no resume path - this stage is first in the chain.

## enum MixCommand

## struct MixRipArgs

The their-side ingest. The stream name rides every document's source field, which is
what a wayside audit counts by, so it is the argument that makes an exclusion
provable rather than assumed.

The kind should never be code on this side. Their code stream arrives already
infilling-transformed upstream, so marking it code would apply the transform twice.

The stamp is verbatim rather than generated, which is what keeps a rebuild
byte-identical: a wall-clock timestamp would make every document differ on every
run.

## struct MixInstructArgs

An example longer than the window is dropped and reported rather than truncated. A
clipped reply teaches the model to stop mid-answer.

## struct MixSampleArgs

## struct MixPackArgs

## struct MixRenderArgs

## enum DocCommand

## enum CapabilityCommand

The four legs in pipeline order: run the engine over fixtures, mirror the JSONL to
NUON, score the mirror, and bridge back to JSONL for a reference comparison.

## struct CapabilityBridgeArgs

## enum SpeculateCommand

## struct SpeculateRecordArgs

## struct SpeculateSimulateArgs

## struct SpeculateTokensArgs

## struct CapabilityConvertArgs

## struct CapabilityScoreArgs

## struct CapabilityRunArgs

Every path here is optional and defaults under the capability home, so the ordinary
invocation takes no arguments at all. The overrides exist to compare a fresh render
or a second run against the standing one, which is a deliberate act rather than a
default.
