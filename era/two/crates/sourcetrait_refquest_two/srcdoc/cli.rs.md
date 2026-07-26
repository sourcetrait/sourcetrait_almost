# cli.rs

The command surface: six verbs, each carrying its whole recipe in its own
arguments.

There is no global flagset, which is the deliberate difference from bquest. That
tool resolves a model and its settings through a profile framework; here every
verb takes an explicit checkpoint choice or an explicit directory, because a
baseline reading is only comparable against another if the recipe is visible in
the command line that produced it. A profile-inherited posture would make two
records look identical while measuring different things.

The 80-character cap applies to these doc comments as it does everywhere, on the
ruling that cli documentation carries no product-text exemption. It costs little
here, since most of these arguments are a name and a default.

Prompts ride files rather than argv throughout. That is not only about length: a
prompt in a shell history is a prompt that has been through a shell, and the
recipes here include exact whitespace.

## struct Cli

## enum Command

## struct EnvArgs

## struct BenchArgs

## struct EvalArgs

## enum EvalSuite

## struct NeedleArgs

The three vllm-only arguments are the capacity surface. The utilisation fraction
tops out near 0.86 beside a live desktop rather than near one; the eager flag
skips graph capture, which is the numerically honest posture their own
documentation requires; and the long-context escape is safe on this model because
it has no positional encoding, so the config's length cap is metadata rather than
a mechanism.

The attention default differs from the generation and dump verbs on purpose. The
battery defaults to sdpa because eager runs out of memory at 8K and above on Tier
A, while a dump defaults to eager because that is the numerically honest path and
the operator overrides it only at length.

## struct GenerateArgs

## struct DumpArgs

Exactly one of the prompt file and the ids source is meant, and the driver
enforces that rather than the type - which is the shape to be aware of, since a
command carrying both parses cleanly here and fails one process later.

## struct DiffArgs

The two dumps are positional rather than flagged, and the order is candidate then
reference. That order is load-bearing: the reference is the comparison's
denominator, so reversing the arguments returns a different number rather than
an error.

## enum ModePick

## enum BackendPick

## enum DevicePick

## enum DtypePick

## enum AttnPick

Small closed enums rather than strings, so an invalid recipe fails at parse
rather than inside a python driver. Each is translated to the driver's own
spelling at the call site, which keeps the wire vocabulary in one place per verb.
