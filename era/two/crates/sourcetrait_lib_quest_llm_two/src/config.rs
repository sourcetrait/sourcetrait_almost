//! The suite's profile framework: config and settings, by directory.
use crate::*;

/// The embedded base (the reserved `defaults` profile), one per kind.
const DEFAULTS_LIB_CONFIG: &str = include_str!("../defaults/config/lib.toml");
const DEFAULTS_LIB_SETTINGS: &str = include_str!("../defaults/settings/lib.toml");

/// The user's standing profile name (the implied -c/-s choice).
const DEFAULT_PROFILE: &str = "default";
/// The reserved embedded-base profile name (never touches the fs).
const DEFAULTS_PROFILE: &str = "defaults";

/// A $VAR value for path expansion; the XDG family has spec fallbacks.
fn xdg_or_env(name: &str) -> LibQuestResult<String> {
    if let Ok(value) = env::var(name)
        && !value.is_empty()
    {
        return Ok(value);
    }
    let home = || -> LibQuestResult<String> {
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

/// Expand a path string's leading `~` or `$VAR` segment.
pub(crate) fn expand_path(raw: &str) -> LibQuestResult<PathBuf> {
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
fn config_home() -> LibQuestResult<PathBuf> {
    Ok(PathBuf::from(xdg_or_env("XDG_CONFIG_HOME")?))
}

/// The suite's config root under the XDG config home.
fn suite_config_root() -> LibQuestResult<PathBuf> {
    Ok(config_home()?.join(consts::SUITE_CONFIG_RELATIVE))
}

/// A token is a profile NAME when it is a pure snake, else a path.
pub(crate) fn is_profile_name(token: &Path) -> bool {
    let Some(text) = token.to_str() else {
        return false;
    };
    !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// A resolved config-profile DIRECTORY; accessors name its files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigProfile(PathBuf);

/// The shared config-token resolution the TryFrom impls delegate to.
fn resolve_config_profile(token: &Path) -> LibQuestResult<ConfigProfile> {
    if is_profile_name(token) {
        Ok(ConfigProfile(
            suite_config_root()?.join("config").join(token),
        ))
    } else {
        Ok(ConfigProfile(token.to_path_buf()))
    }
}

impl TryFrom<&Path> for ConfigProfile {
    type Error = LibQuestError;

    fn try_from(token: &Path) -> LibQuestResult<Self> {
        resolve_config_profile(token)
    }
}

impl TryFrom<PathBuf> for ConfigProfile {
    type Error = LibQuestError;

    fn try_from(token: PathBuf) -> LibQuestResult<Self> {
        resolve_config_profile(&token)
    }
}

impl TryFrom<&str> for ConfigProfile {
    type Error = LibQuestError;

    fn try_from(token: &str) -> LibQuestResult<Self> {
        resolve_config_profile(Path::new(token))
    }
}

impl TryFrom<String> for ConfigProfile {
    type Error = LibQuestError;

    fn try_from(token: String) -> LibQuestResult<Self> {
        resolve_config_profile(Path::new(&token))
    }
}

impl ConfigProfile {
    /// Resolve a name against a custom root instead of the XDG one.
    pub fn try_from_dir<P1: AsRef<Path>, P2: AsRef<Path>>(
        dir: P1,
        token: P2,
    ) -> LibQuestResult<Self> {
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

    pub fn bquest_config(&self) -> PathBuf {
        self.0.join("bquest.toml")
    }
}

/// A resolved settings-profile DIRECTORY, the ConfigProfile sibling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsProfile(PathBuf);

/// The shared settings-token resolution, shaped like the config one.
fn resolve_settings_profile(token: &Path) -> LibQuestResult<SettingsProfile> {
    if is_profile_name(token) {
        Ok(SettingsProfile(
            suite_config_root()?.join("settings").join(token),
        ))
    } else {
        Ok(SettingsProfile(token.to_path_buf()))
    }
}

impl TryFrom<&Path> for SettingsProfile {
    type Error = LibQuestError;

    fn try_from(token: &Path) -> LibQuestResult<Self> {
        resolve_settings_profile(token)
    }
}

impl TryFrom<PathBuf> for SettingsProfile {
    type Error = LibQuestError;

    fn try_from(token: PathBuf) -> LibQuestResult<Self> {
        resolve_settings_profile(&token)
    }
}

impl TryFrom<&str> for SettingsProfile {
    type Error = LibQuestError;

    fn try_from(token: &str) -> LibQuestResult<Self> {
        resolve_settings_profile(Path::new(token))
    }
}

impl TryFrom<String> for SettingsProfile {
    type Error = LibQuestError;

    fn try_from(token: String) -> LibQuestResult<Self> {
        resolve_settings_profile(Path::new(&token))
    }
}

impl SettingsProfile {
    /// Resolve a name against a custom root instead of the XDG one.
    pub fn try_from_dir<P1: AsRef<Path>, P2: AsRef<Path>>(
        dir: P1,
        token: P2,
    ) -> LibQuestResult<Self> {
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

    pub fn bquest_settings(&self) -> PathBuf {
        self.0.join("bquest.toml")
    }
}

/// The lib component's config file shape (the TOML format layer).
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibConfigToml {
    pub model: Option<String>,
    pub models_dir: Option<String>,
    pub snapshots_dir: Option<String>,
    pub adapters_dir: Option<String>,
    pub adapter: Option<String>,
}

/// The lib component's GENERAL OPERATION - the stable choices.
#[derive(Debug, Clone)]
pub struct LibConfig {
    pub model: String,
    pub models_dir: PathBuf,
    pub snapshots_dir: PathBuf,
    pub adapters_dir: PathBuf,
    pub adapter: Option<String>,
}

impl TryFrom<LibConfigToml> for LibConfig {
    type Error = LibQuestError;

    fn try_from(user: LibConfigToml) -> LibQuestResult<Self> {
        let base: LibConfigToml = toml::from_str(DEFAULTS_LIB_CONFIG)?;
        let Some(models_dir_raw) = user.models_dir.or(base.models_dir) else {
            snafu::whatever!("the embedded defaults carry no models_dir");
        };
        let Some(snapshots_dir_raw) = user.snapshots_dir.or(base.snapshots_dir) else {
            snafu::whatever!("the embedded defaults carry no snapshots_dir");
        };
        let Some(adapters_dir_raw) = user.adapters_dir.or(base.adapters_dir) else {
            snafu::whatever!("the embedded defaults carry no adapters_dir");
        };
        let Some(model) = user.model.or(base.model) else {
            snafu::whatever!("the embedded defaults carry no model coordinate");
        };
        Ok(Self {
            model,
            models_dir: expand_path(&models_dir_raw)?,
            snapshots_dir: expand_path(&snapshots_dir_raw)?,
            adapters_dir: expand_path(&adapters_dir_raw)?,
            adapter: user.adapter,
        })
    }
}

impl LibConfig {
    /// The configured checkpoint's directory under the models home.
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
    /// Load an explicit component-toml path; an absent file errors.
    pub fn from_config_path(path: &Path) -> LibQuestResult<Self> {
        let text = fs::read_to_string(path)?;
        let shell: LibConfigToml = toml::from_str(&text)?;
        shell.try_into()
    }

    /// Resolve and load per the token rules against the XDG root.
    pub fn load<P: AsRef<Path>>(token: Option<P>) -> LibQuestResult<Self> {
        Self::load_from_dir(None::<&Path>, token)
    }

    /// The custom-root variant of load; path tokens are unaffected.
    pub fn load_from_dir<P1: AsRef<Path>, P2: AsRef<Path>>(
        dir: Option<P1>,
        token: Option<P2>,
    ) -> LibQuestResult<Self> {
        let profile_for = |name: &str| -> LibQuestResult<ConfigProfile> {
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

/// The [eviction] table; its presence with a cap is what arms eviction.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvictionToml {
    pub decode_cap: Option<usize>,
    pub recent: Option<usize>,
    pub sink: Option<usize>,
    pub score_tail: Option<usize>,
    pub score_slice: Option<usize>,
}

/// Armed eviction: the cap, its protected bands, and the two levers.
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

/// Merge and validate the [eviction] table; a capless table errors.
fn merged_eviction(
    user: Option<EvictionToml>,
    base: Option<EvictionToml>,
) -> LibQuestResult<Option<EvictionSettings>> {
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

/// The [generation] table; absent fields fall through to the defaults.
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
#[derive(Debug, Clone)]
pub struct LibSettings {
    pub use_flash_attn: bool,
    pub graph: bool,
    pub graph_bucket_grain: usize,
    /// The fused decode-step kernels; cuda decode only.
    pub fused_gdn: bool,
    /// The fused prefill kernels; cuda multi-token chunks only.
    pub fused_prefill: bool,
    pub generation: GenerateOptions,
    /// None is the exact configuration, with no eviction at all.
    pub eviction: Option<EvictionSettings>,
}

/// Overlay per field: the user file, the embedded base, then code.
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
    type Error = LibQuestError;

    fn try_from(user: LibSettingsToml) -> LibQuestResult<Self> {
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
    /// Load an explicit component-toml path; an absent file errors.
    pub fn from_settings_path(path: &Path) -> LibQuestResult<Self> {
        let text = fs::read_to_string(path)?;
        let shell: LibSettingsToml = toml::from_str(&text)?;
        shell.try_into()
    }

    /// Resolve and load per the same name rules as LibConfig::load.
    pub fn load<P: AsRef<Path>>(token: Option<P>) -> LibQuestResult<Self> {
        Self::load_from_dir(None::<&Path>, token)
    }

    /// The custom-root variant of load; path tokens are unaffected.
    pub fn load_from_dir<P1: AsRef<Path>, P2: AsRef<Path>>(
        dir: Option<P1>,
        token: Option<P2>,
    ) -> LibQuestResult<Self> {
        let profile_for = |name: &str| -> LibQuestResult<SettingsProfile> {
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
