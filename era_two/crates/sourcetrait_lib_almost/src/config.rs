//! The suite's -d/-c/-s profile framework: core model types stay
//! format-free, serde TOML shells bridge via TryFrom (a future format
//! adds a shell without touching the core model), and profiles are
//! DIRECTORIES holding one file per suite component (lib/cli/tui/
//! tool/baseline .toml) under config/ and settings/ roots.
//!
//! Name rules: `defaults` is reserved for the embedded base every
//! load merges onto; `default` is the user's standing profile - the
//! implied choice when no token is given, falling back to the
//! embedded base when its file is absent. Any other explicit token
//! errors when missing.
use crate::*;

/// The embedded base (the reserved `defaults` profile), one per kind.
const DEFAULTS_LIB_CONFIG: &str = include_str!("../defaults/config/lib.toml");
const DEFAULTS_LIB_SETTINGS: &str = include_str!("../defaults/settings/lib.toml");

/// The user's standing profile name (the implied -c/-s choice).
const DEFAULT_PROFILE: &str = "default";
/// The reserved embedded-base profile name (never touches the fs).
const DEFAULTS_PROFILE: &str = "defaults";

/// A $VAR value for path expansion; the XDG family falls back per
/// the basedir spec when unset, anything else errors.
fn xdg_or_env(name: &str) -> LibAlmostResult<String> {
    if let Ok(value) = env::var(name)
        && !value.is_empty()
    {
        return Ok(value);
    }
    let home = || -> LibAlmostResult<String> {
        match env::var("HOME") {
            Ok(home) if !home.is_empty() => Ok(home),
            _ => snafu::whatever!("HOME is not set for the ${name} fallback"),
        }
    };
    match name {
        "XDG_DATA_HOME" => Ok(format!("{}/.local/share", home()?)),
        "XDG_CONFIG_HOME" => Ok(format!("{}/.config", home()?)),
        "XDG_STATE_HOME" => Ok(format!("{}/.local/state", home()?)),
        "XDG_CACHE_HOME" => Ok(format!("{}/.cache", home()?)),
        _ => snafu::whatever!("environment variable ${name} is not set"),
    }
}

/// Expand a path STRING's leading `~` (HOME) or `$VAR` segment so
/// config files carry portable strings; other paths pass through
/// literally.
pub(crate) fn expand_path(raw: &str) -> LibAlmostResult<PathBuf> {
    if let Some(rest) = raw.strip_prefix("~") {
        let Ok(home) = env::var("HOME") else {
            snafu::whatever!("HOME is not set for ~ expansion");
        };
        return Ok(PathBuf::from(format!("{home}{rest}")));
    }
    if let Some(rest) = raw.strip_prefix("$") {
        let (name, tail) = match rest.find('/') {
            Some(split) => rest.split_at(split),
            None => (rest, ""),
        };
        return Ok(PathBuf::from(format!("{}{tail}", xdg_or_env(name)?)));
    }
    Ok(PathBuf::from(raw))
}

/// XDG config home, honoring the spec fallback (~/.config).
fn config_home() -> LibAlmostResult<PathBuf> {
    Ok(PathBuf::from(xdg_or_env("XDG_CONFIG_HOME")?))
}

/// The suite's config root under the XDG config home.
fn suite_config_root() -> LibAlmostResult<PathBuf> {
    Ok(config_home()?.join(consts::SUITE_CONFIG_RELATIVE))
}

/// A -c/-s token is a profile NAME when it is a pure snake; anything
/// else is a filesystem path (snapshot tokens share the same snake
/// rule).
pub(crate) fn is_profile_name(token: &Path) -> bool {
    let Some(text) = token.to_str() else {
        return false;
    };
    !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// A resolved config-profile DIRECTORY; the component accessors name
/// the file inside it. A pure-snake token resolves under the suite's
/// XDG config root (config/<name>); any other token is used as the
/// directory itself. Exact component-FILE tokens bypass profiles
/// entirely (the load fns take them directly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigProfile(PathBuf);

/// The shared -c token resolution (coherence forbids one generic
/// TryFrom<P: AsRef<Path>> beside std's blanket, so the concrete
/// impls below each delegate here).
fn resolve_config_profile(token: &Path) -> LibAlmostResult<ConfigProfile> {
    if is_profile_name(token) {
        Ok(ConfigProfile(
            suite_config_root()?.join("config").join(token),
        ))
    } else {
        Ok(ConfigProfile(token.to_path_buf()))
    }
}

impl TryFrom<&Path> for ConfigProfile {
    type Error = LibAlmostError;

    fn try_from(token: &Path) -> LibAlmostResult<Self> {
        resolve_config_profile(token)
    }
}

impl TryFrom<PathBuf> for ConfigProfile {
    type Error = LibAlmostError;

    fn try_from(token: PathBuf) -> LibAlmostResult<Self> {
        resolve_config_profile(&token)
    }
}

impl TryFrom<&str> for ConfigProfile {
    type Error = LibAlmostError;

    fn try_from(token: &str) -> LibAlmostResult<Self> {
        resolve_config_profile(Path::new(token))
    }
}

impl TryFrom<String> for ConfigProfile {
    type Error = LibAlmostError;

    fn try_from(token: String) -> LibAlmostResult<Self> {
        resolve_config_profile(Path::new(&token))
    }
}

impl ConfigProfile {
    /// Resolve against a custom root (-d) instead of the XDG config
    /// root: <dir>/config/<name> for a snake, the token as-is
    /// otherwise.
    pub fn try_from_dir<P1: AsRef<Path>, P2: AsRef<Path>>(
        dir: P1,
        token: P2,
    ) -> LibAlmostResult<Self> {
        let token = token.as_ref();
        if is_profile_name(token) {
            Ok(Self(dir.as_ref().join("config").join(token)))
        } else {
            Ok(Self(token.to_path_buf()))
        }
    }

    pub fn dir(&self) -> &Path {
        &self.0
    }

    pub fn lib_config(&self) -> PathBuf {
        self.0.join("lib.toml")
    }

    pub fn cli_config(&self) -> PathBuf {
        self.0.join("cli.toml")
    }

    pub fn tui_config(&self) -> PathBuf {
        self.0.join("tui.toml")
    }

    pub fn tool_config(&self) -> PathBuf {
        self.0.join("tool.toml")
    }

    pub fn baseline_config(&self) -> PathBuf {
        self.0.join("baseline.toml")
    }

    pub fn lmst_config(&self) -> PathBuf {
        self.0.join("lmst.toml")
    }
}

/// A resolved settings-profile DIRECTORY (the settings/ sibling of
/// ConfigProfile; same token rules).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsProfile(PathBuf);

/// The shared -s token resolution (the same concrete-impl shape as
/// ConfigProfile, for the same coherence reason).
fn resolve_settings_profile(token: &Path) -> LibAlmostResult<SettingsProfile> {
    if is_profile_name(token) {
        Ok(SettingsProfile(
            suite_config_root()?.join("settings").join(token),
        ))
    } else {
        Ok(SettingsProfile(token.to_path_buf()))
    }
}

impl TryFrom<&Path> for SettingsProfile {
    type Error = LibAlmostError;

    fn try_from(token: &Path) -> LibAlmostResult<Self> {
        resolve_settings_profile(token)
    }
}

impl TryFrom<PathBuf> for SettingsProfile {
    type Error = LibAlmostError;

    fn try_from(token: PathBuf) -> LibAlmostResult<Self> {
        resolve_settings_profile(&token)
    }
}

impl TryFrom<&str> for SettingsProfile {
    type Error = LibAlmostError;

    fn try_from(token: &str) -> LibAlmostResult<Self> {
        resolve_settings_profile(Path::new(token))
    }
}

impl TryFrom<String> for SettingsProfile {
    type Error = LibAlmostError;

    fn try_from(token: String) -> LibAlmostResult<Self> {
        resolve_settings_profile(Path::new(&token))
    }
}

impl SettingsProfile {
    /// Resolve against a custom root (-d) instead of the XDG config
    /// root: <dir>/settings/<name> for a snake, the token as-is
    /// otherwise.
    pub fn try_from_dir<P1: AsRef<Path>, P2: AsRef<Path>>(
        dir: P1,
        token: P2,
    ) -> LibAlmostResult<Self> {
        let token = token.as_ref();
        if is_profile_name(token) {
            Ok(Self(dir.as_ref().join("settings").join(token)))
        } else {
            Ok(Self(token.to_path_buf()))
        }
    }

    pub fn dir(&self) -> &Path {
        &self.0
    }

    pub fn lib_settings(&self) -> PathBuf {
        self.0.join("lib.toml")
    }

    pub fn cli_settings(&self) -> PathBuf {
        self.0.join("cli.toml")
    }

    pub fn tui_settings(&self) -> PathBuf {
        self.0.join("tui.toml")
    }

    pub fn tool_settings(&self) -> PathBuf {
        self.0.join("tool.toml")
    }

    pub fn baseline_settings(&self) -> PathBuf {
        self.0.join("baseline.toml")
    }

    pub fn lmst_settings(&self) -> PathBuf {
        self.0.join("lmst.toml")
    }
}

/// The lib component's config file shape (TOML format layer).
/// models_dir/snapshots_dir are STRINGS so files stay portable - a
/// leading `~` or `$VAR` expands at load (the XDG family with spec
/// fallbacks).
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibConfigToml {
    pub model: Option<String>,
    pub models_dir: Option<String>,
    pub snapshots_dir: Option<String>,
}

/// The lib component's GENERAL OPERATION - the stable choices: the
/// checkpoint as an author-qualified coordinate (`author/name`,
/// joined beneath models_dir), where models live (models_dir defaults
/// to the XDG data-home models root), and where snapshots live
/// (snapshots_dir defaults to the XDG cache-home snapshots root -
/// snapshots are regenerable). Nuance and tweaks (flash, graphs, the
/// generation posture) are LibSettings.
#[derive(Debug, Clone)]
pub struct LibConfig {
    pub model: String,
    pub models_dir: PathBuf,
    pub snapshots_dir: PathBuf,
}

impl TryFrom<LibConfigToml> for LibConfig {
    type Error = LibAlmostError;

    fn try_from(user: LibConfigToml) -> LibAlmostResult<Self> {
        let base: LibConfigToml = toml::from_str(DEFAULTS_LIB_CONFIG)?;
        let Some(models_dir_raw) = user.models_dir.or(base.models_dir) else {
            snafu::whatever!("the embedded defaults carry no models_dir");
        };
        let Some(snapshots_dir_raw) = user.snapshots_dir.or(base.snapshots_dir) else {
            snafu::whatever!("the embedded defaults carry no snapshots_dir");
        };
        let Some(model) = user.model.or(base.model) else {
            snafu::whatever!("the embedded defaults carry no model coordinate");
        };
        Ok(Self {
            model,
            models_dir: expand_path(&models_dir_raw)?,
            snapshots_dir: expand_path(&snapshots_dir_raw)?,
        })
    }
}

impl LibConfig {
    /// The configured checkpoint's directory: the author-qualified
    /// coordinate joined beneath the models home.
    pub fn model_dir(&self) -> PathBuf {
        self.models_dir.join(&self.model)
    }
}

impl Default for LibConfig {
    fn default() -> Self {
        LibConfigToml::default()
            .try_into()
            .expect("embedded defaults parse")
    }
}

impl LibConfig {
    /// Load from an explicit component-toml file path; an absent file
    /// is an error (explicit tokens never fall back).
    pub fn from_config_path(path: &Path) -> LibAlmostResult<Self> {
        let text = fs::read_to_string(path)?;
        let shell: LibConfigToml = toml::from_str(&text)?;
        shell.try_into()
    }

    /// Resolve + load per the -c token rules against the XDG root:
    /// None -> the `default` profile (absent file falls to the
    /// embedded base); `defaults` -> the embedded base alone; another
    /// snake -> that profile's lib.toml (absent = error); a path ->
    /// that component toml file (absent = error).
    pub fn load<P: AsRef<Path>>(token: Option<P>) -> LibAlmostResult<Self> {
        Self::load_from_dir(None::<&Path>, token)
    }

    /// The -d variant of load: profile names resolve under `dir`
    /// instead of the XDG config root; path tokens are unaffected.
    pub fn load_from_dir<P1: AsRef<Path>, P2: AsRef<Path>>(
        dir: Option<P1>,
        token: Option<P2>,
    ) -> LibAlmostResult<Self> {
        let profile_for = |name: &str| -> LibAlmostResult<ConfigProfile> {
            match &dir {
                Some(dir) => ConfigProfile::try_from_dir(dir, name),
                None => ConfigProfile::try_from(name),
            }
        };
        let Some(token) = token else {
            let path = profile_for(DEFAULT_PROFILE)?.lib_config();
            if path.is_file() {
                return Self::from_config_path(&path);
            }
            return LibConfigToml::default().try_into();
        };
        let token = token.as_ref();
        if !is_profile_name(token) {
            return Self::from_config_path(token);
        }
        let name = token.to_str().expect("snake tokens are utf-8");
        if name == DEFAULTS_PROFILE {
            return LibConfigToml::default().try_into();
        }
        let path = profile_for(name)?.lib_config();
        if name == DEFAULT_PROFILE && !path.is_file() {
            return LibConfigToml::default().try_into();
        }
        Self::from_config_path(&path)
    }
}

/// The [eviction] table of a lib settings file: presence (with
/// decode_cap) arms KvEviction stage-2-only attention-KV eviction;
/// absent = the exact configuration, bit-identical to an
/// eviction-free build.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvictionToml {
    pub decode_cap: Option<usize>,
    pub recent: Option<usize>,
    pub sink: Option<usize>,
    pub score_tail: Option<usize>,
    pub score_slice: Option<usize>,
}

/// Armed KvEviction stage-2-only eviction: one post-prefill
/// compaction of every attention layer's KV to `decode_cap` rows per
/// head (last-pass ranked, sink prefix + recent suffix protected),
/// then overflow re-compactions as decode outgrows the cap.
/// score_tail is the per-prefill-chunk observation window (tail
/// queries per re-score pass; the RescoreTuning prefill-cost lever)
/// and score_slice the query rows per scoring matmul (the
/// peak-transient lever) - both riding the settings surface as the
/// prototyping home while tuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionSettings {
    pub decode_cap: usize,
    pub recent: usize,
    pub sink: usize,
    pub score_tail: usize,
    pub score_slice: usize,
}

/// The era-one v1 protection defaults (recent 512, sink 4).
const EVICTION_RECENT_DEFAULT: usize = 512;
const EVICTION_SINK_DEFAULT: usize = 4;

/// Merge + validate the [eviction] table: a table without decode_cap
/// is an error (stage-2-only surface - the cap IS the arm), and the
/// cap must clear the protected rows.
fn merged_eviction(
    user: Option<EvictionToml>,
    base: Option<EvictionToml>,
) -> LibAlmostResult<Option<EvictionSettings>> {
    let (user, base) = match (user, base) {
        (None, None) => return Ok(None),
        (user, base) => (user.unwrap_or_default(), base.unwrap_or_default()),
    };
    let Some(decode_cap) = user.decode_cap.or(base.decode_cap) else {
        snafu::whatever!("[eviction] needs decode_cap (stage-2-only: the cap is the arm)");
    };
    let settings = EvictionSettings {
        decode_cap,
        recent: user
            .recent
            .or(base.recent)
            .unwrap_or(EVICTION_RECENT_DEFAULT),
        sink: user.sink.or(base.sink).unwrap_or(EVICTION_SINK_DEFAULT),
        score_tail: user
            .score_tail
            .or(base.score_tail)
            .unwrap_or(evict::SCORE_TAIL),
        score_slice: user
            .score_slice
            .or(base.score_slice)
            .unwrap_or(evict::SCORE_SLICE),
    };
    if settings.decode_cap <= settings.recent + settings.sink {
        snafu::whatever!(
            "eviction decode_cap {} must exceed recent {} + sink {}",
            settings.decode_cap,
            settings.recent,
            settings.sink
        );
    }
    if settings.score_tail == 0 || settings.score_slice == 0 {
        snafu::whatever!(
            "eviction score_tail ({}) and score_slice ({}) must be at least 1",
            settings.score_tail,
            settings.score_slice
        );
    }
    Ok(Some(settings))
}

/// The [generation] table of a lib settings file; every field
/// optional (absent fields fall through the embedded base to the
/// code defaults). greedy = true forces argmax by zeroing
/// temperature and top_p, whatever else the file says.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationToml {
    pub greedy: Option<bool>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub sample_len: Option<usize>,
    pub chat: Option<bool>,
    pub ignore_stops: Option<bool>,
    pub speculate: Option<bool>,
}

/// The lib component's settings file shape (TOML format layer).
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibSettingsToml {
    pub use_flash_attn: Option<bool>,
    pub graph: Option<bool>,
    pub graph_bucket_grain: Option<usize>,
    pub fused_gdn: Option<bool>,
    pub fused_prefill: Option<bool>,
    pub generation: Option<GenerationToml>,
    pub eviction: Option<EvictionToml>,
}

/// The lib component's NUANCE - the execution and posture tweaks.
/// use_flash_attn arms the flash prefill dispatch where eligible (a
/// flash-attn build on cuda at bf16/f16 - ineligible runs stay
/// eager); graph arms CUDA-graph decode capture and
/// graph_bucket_grain is its static attention-width grain (pad
/// columns carry zero softmax mass - at most one grain of padded
/// compute per step); generation is the sampling/budget posture.
#[derive(Debug, Clone)]
pub struct LibSettings {
    pub use_flash_attn: bool,
    pub graph: bool,
    pub graph_bucket_grain: usize,
    /// The GdnChainFusion decode-step kernel (cuda decode only; the
    /// cpu path always runs the classic candle chain).
    pub fused_gdn: bool,
    /// The PrefillDispatch chunk kernels (cuda multi-token prefill
    /// chunks only; the cpu path and the stateless parity form always
    /// run the classic candle chain).
    pub fused_prefill: bool,
    pub generation: GenerateOptions,
    /// KvEviction stage-2-only; None = the exact configuration.
    pub eviction: Option<EvictionSettings>,
}

/// Overlay chain per field: the user file, then the embedded base,
/// then the code defaults; greedy wins over sampling params.
fn merged_generation(user: GenerationToml, base: GenerationToml) -> GenerateOptions {
    let code = GenerateOptions::default();
    let greedy = user.greedy.or(base.greedy).unwrap_or(false);
    let temperature = user.temperature.or(base.temperature).or(code.temperature);
    let top_p = user.top_p.or(base.top_p).or(code.top_p);
    GenerateOptions {
        temperature: if greedy { None } else { temperature },
        top_p: if greedy { None } else { top_p },
        seed: user.seed.or(base.seed).unwrap_or(code.seed),
        sample_len: user.sample_len.or(base.sample_len).unwrap_or(code.sample_len),
        chat: user.chat.or(base.chat).unwrap_or(code.chat),
        ignore_stops: user.ignore_stops.or(base.ignore_stops).unwrap_or(code.ignore_stops),
        speculate: user.speculate.or(base.speculate).unwrap_or(code.speculate),
    }
}

impl TryFrom<LibSettingsToml> for LibSettings {
    type Error = LibAlmostError;

    fn try_from(user: LibSettingsToml) -> LibAlmostResult<Self> {
        let base: LibSettingsToml = toml::from_str(DEFAULTS_LIB_SETTINGS)?;
        Ok(Self {
            use_flash_attn: user
                .use_flash_attn
                .or(base.use_flash_attn)
                .unwrap_or(true),
            graph: user.graph.or(base.graph).unwrap_or(false),
            graph_bucket_grain: user
                .graph_bucket_grain
                .or(base.graph_bucket_grain)
                .unwrap_or(2048),
            fused_gdn: user.fused_gdn.or(base.fused_gdn).unwrap_or(true),
            fused_prefill: user
                .fused_prefill
                .or(base.fused_prefill)
                .unwrap_or(true),
            generation: merged_generation(
                user.generation.unwrap_or_default(),
                base.generation.unwrap_or_default(),
            ),
            eviction: merged_eviction(user.eviction, base.eviction)?,
        })
    }
}

impl Default for LibSettings {
    fn default() -> Self {
        LibSettingsToml::default()
            .try_into()
            .expect("embedded defaults parse")
    }
}

impl LibSettings {
    /// Load from an explicit component-toml file path; an absent file
    /// is an error (explicit tokens never fall back).
    pub fn from_settings_path(path: &Path) -> LibAlmostResult<Self> {
        let text = fs::read_to_string(path)?;
        let shell: LibSettingsToml = toml::from_str(&text)?;
        shell.try_into()
    }

    /// Resolve + load per the -s token rules against the XDG root
    /// (the same name rules as LibConfig::load).
    pub fn load<P: AsRef<Path>>(token: Option<P>) -> LibAlmostResult<Self> {
        Self::load_from_dir(None::<&Path>, token)
    }

    /// The -d variant of load: profile names resolve under `dir`
    /// instead of the XDG config root; path tokens are unaffected.
    pub fn load_from_dir<P1: AsRef<Path>, P2: AsRef<Path>>(
        dir: Option<P1>,
        token: Option<P2>,
    ) -> LibAlmostResult<Self> {
        let profile_for = |name: &str| -> LibAlmostResult<SettingsProfile> {
            match &dir {
                Some(dir) => SettingsProfile::try_from_dir(dir, name),
                None => SettingsProfile::try_from(name),
            }
        };
        let Some(token) = token else {
            let path = profile_for(DEFAULT_PROFILE)?.lib_settings();
            if path.is_file() {
                return Self::from_settings_path(&path);
            }
            return LibSettingsToml::default().try_into();
        };
        let token = token.as_ref();
        if !is_profile_name(token) {
            return Self::from_settings_path(token);
        }
        let name = token.to_str().expect("snake tokens are utf-8");
        if name == DEFAULTS_PROFILE {
            return LibSettingsToml::default().try_into();
        }
        let path = profile_for(name)?.lib_settings();
        if name == DEFAULT_PROFILE && !path.is_file() {
            return LibSettingsToml::default().try_into();
        }
        Self::from_settings_path(&path)
    }
}
