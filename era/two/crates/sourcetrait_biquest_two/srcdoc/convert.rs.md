# convert.rs

The JSONL and nu-value bridge, and the `capability convert` verb over it. It also
holds the value constructors and record accessors the rest of the crate builds
NUON with, because those are the same primitives the bridge needs.

Conversion is lossless field for field, and each half of that is a deliberate
choice with a cost. Object order is preserved, which is why `serde_json`'s
`preserve_order` feature is load-bearing workspace-wide - field order is data in a
mirrored artifact. Integers stay 64-bit signed, and one that would overflow raises
rather than becoming a float, because a silent approximation is exactly the kind of
loss a lossless bridge exists to prevent.

The gate is integral rather than a separate step. Every written file is re-parsed
and must deep-equal every source row, in the same function that wrote it, so every
future conversion is gate-checked by construction rather than by remembering to
run something.

Trees are discovered rather than hardcoded, so any fixture render and any runs
root of the same shape converts through the same code.

## const REQUEST_LOGLIKELIHOOD_TYPEDEF

## const REQUEST_GENERATE_TYPEDEF

## const PREDICTION_TYPEDEF

Harvested from the standing fixture and run trees rather than written from a
specification, so they describe the spine every real writer satisfies. Records
stay open under conformance, which is what lets a writer's own extras pass
undeclared - and there are several, since different runs carry different metric
sidecars.

## fn span

## fn v_int

## fn v_float

## fn v_str

## fn v_bool

## fn v_int_list

Every constructed value carries an unknown span. These values are built rather
than parsed, so there is no source location to point at, and a fabricated one
would make a diagnostic lie about where a value came from.

## fn field

## fn field_str

## fn field_int

## fn field_rows

## fn field_ids

## fn field_strings

Accessors that name the missing or mistyped field in their error. The nu accessors
they wrap report the type mismatch without the field name, which in a row of
twenty fields is the difference between a usable diagnostic and a hunt.

## fn value_to_json

The reverse direction, for rendering our artifacts back into the reference's own
JSONL so its tools can consume them. A non-finite float raises rather than
becoming null, since JSON cannot carry one and null would read as a missing
measurement.

## fn json_to_value

## fn collect_suffix_files

## fn task_token

The stem with its suffix stripped and a trailing render hash removed, which equals
the reference's own sanitisation of a task specification. The hash is recognised by
being exactly six hex digits after the last underscore, so a task whose name
legitimately ends that way would be mis-tokenised - none does, and the alternative
is carrying a list of known task names.

## fn task_selected

## fn sha256_hex

## fn epoch_seconds

## fn request_typedef

## enum SourceKind

## fn convert_file

The lossless gate lives here, at the end, comparing the reparsed rows against the
source values it still holds. It reports the count of mismatched rows and the
first few indices rather than the first mismatch alone, because a systematic
conversion error affects every row and knowing that is the diagnosis.

The request typedef is cached after the first row rather than resolved per row.
A file is one task and one request type by construction, so the first row decides
it.

## fn capability_convert

Every default lands under the capability home, so the common invocation takes no
arguments at all. The overrides exist for comparing a fresh render against the
standing one rather than for ordinary use.

Runs are discovered by looking for a `predictions` directory rather than from a
list, which is what makes a newly-recorded run convert without a code change.
