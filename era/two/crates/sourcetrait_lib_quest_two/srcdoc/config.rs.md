# config.rs

The suite's profile framework, and the split it rests on: core model types stay
FORMAT-FREE while serde TOML shells bridge to them through `TryFrom`, so a
future format adds a shell without touching the core model. Every file-layer
field is optional and every path is a STRING, which is what keeps a config file
portable between machines.

A profile is a DIRECTORY holding one file per suite component rather than a
single file, so components version independently and a tool reads only its own.

THE NAME RULES ARE THREE, and the difference between them is what a missing
file means. `defaults` is reserved for the embedded base every load merges
onto, and never touches the filesystem. `default` is the user's standing
profile - the implied choice when no token is given - and falls back to the
embedded base when its file is absent. ANY OTHER explicit token errors when
missing, because asking for a named profile and silently getting another one is
the failure worth being loud about.

CONFIG IS OPERATION, SETTINGS ARE NUANCE. The checkpoint coordinate, the
directories and the adapter token are stable choices that identify a run;
flash, graphs, fusion, eviction and the generation posture are how it executes.
That line is also why the settings surface doubles as the prototyping home: a
would-be constant rides as a settings field while it is being tuned and
graduates to a constant once it settles.

## const DEFAULTS_LIB_CONFIG

## const DEFAULTS_LIB_SETTINGS

## const DEFAULT_PROFILE

## const DEFAULTS_PROFILE

## fn xdg_or_env

The XDG family falls back per the basedir spec rather than erroring, because
those variables are legitimately unset on a plain login; anything else errors,
because a typo'd variable expanding to nothing would silently produce a path
under the wrong root.

## fn expand_path

## fn config_home

## fn suite_config_root

## fn is_profile_name

A pure snake is a NAME and anything else is a PATH. Snapshot and adapter tokens
share this exact rule, so a user learns one distinction rather than three.

## struct ConfigProfile

### fn resolve_config_profile

The concrete `TryFrom` impls each delegate here because coherence forbids one
generic `TryFrom<P: AsRef<Path>>` beside std's blanket implementation. That is
a language constraint rather than a style choice, and it is why the same four
impls appear again for `SettingsProfile`.

### impl TryFrom for ConfigProfile

### fn try_from_dir

### fn dir

### fn lib_config

### fn cli_config

### fn tui_config

### fn tool_config

### fn baseline_config

### fn bquest_config

## struct SettingsProfile

### fn resolve_settings_profile

### impl TryFrom for SettingsProfile

### fn try_from_dir

### fn dir

### fn lib_settings

### fn cli_settings

### fn tui_settings

### fn tool_settings

### fn baseline_settings

### fn bquest_settings

## struct LibConfigToml

`deny_unknown_fields` is what turns a typo'd key into a loud error rather than
a silently ignored one.

## struct LibConfig

An ABSENT adapter is `None` rather than an empty string or a sentinel, which is
what lets the loader take the plain mmap path and be bit-exact by omission.

### impl TryFrom for LibConfig

The merge reads the embedded base and overlays the user's file field by field.
A missing value in BOTH is an error rather than a silent default, because the
embedded base is checked in and its absence means the build is wrong.

### fn model_dir

### impl Default for LibConfig

### fn from_config_path

### fn load

### fn load_from_dir

The `-d` variant redirects PROFILE NAMES to a custom root while leaving PATH
tokens alone, which is what makes a test or a battery able to run against its
own profile tree without touching the user's.

## struct EvictionToml

## struct EvictionSettings

## const EVICTION_RECENT_DEFAULT

## const EVICTION_SINK_DEFAULT

## fn merged_eviction

A table WITHOUT `decode_cap` is an error rather than a defaulted arm. The cap
IS the arm on this stage-2-only surface, so defaulting it would silently turn
eviction on for anyone who wrote an empty table.

The cap must clear the protected rows, because a cap at or below sink plus
recent leaves the ranked middle no budget at all and the eviction would be
protection alone.

## struct GenerationToml

## struct LibSettingsToml

## struct LibSettings

## fn merged_generation

GREEDY WINS over the sampling parameters whatever else the file says, by
zeroing temperature and top-p rather than by being checked downstream. That
keeps "greedy" a single fact about the run rather than a condition every
sampler site has to re-read.

### impl TryFrom for LibSettings

### impl Default for LibSettings

### fn from_settings_path

### fn load

### fn load_from_dir
