use crate::*;

/// User-declared run intent, loaded from a TOML profile (or code
/// defaults). Profiles live under $XDG_CONFIG_HOME/sourcetrait/almost/:
/// config/<name>.toml declares intent; settings/<name>.toml (same name -
/// the auto-pair) surgically overrides the runtime Settings that
/// Settings::from_config derives from this. Partial files are the norm:
/// absent fields take these defaults; unknown fields are rejected.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub model_id: String,
    /// Checkpoint home; None = the shared XDG cache home default.
    pub model_dir: Option<PathBuf>,
    pub device: DeviceConfig,
    pub dtype: DtypeConfig,
    pub generation: GenerationConfig,
    /// A3 eviction intent; absent (or capless) = the exact configuration.
    pub eviction: Option<EvictionConfig>,
    pub chat: ChatConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_id: String::from(consts::DEFAULT_MODEL_ID),
            model_dir: None,
            device: DeviceConfig::Auto,
            dtype: DtypeConfig::Auto,
            generation: GenerationConfig::default(),
            eviction: None,
            chat: ChatConfig::default(),
        }
    }
}

/// Chat-surface intent (talmost). log defaults ON: each turn rewrites
/// <cache>/sourcetrait/almost/session/<session nom>.txt.
#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatConfig {
    pub log: bool,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self { log: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceConfig {
    /// CUDA when available, else CPU.
    Auto,
    /// CUDA or a hard error (explicit intent never degrades).
    Cuda,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DtypeConfig {
    /// bf16 on cuda, f32 on cpu.
    Auto,
    Bf16,
    F32,
}

/// Sampling and length intent; feeds GenerateOptions.
#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenerationConfig {
    pub greedy: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub sample_len: usize,
    pub seed: u64,
    pub speculate: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            greedy: false,
            temperature: consts::DEFAULT_TEMPERATURE,
            top_p: consts::DEFAULT_TOP_P,
            sample_len: consts::DEFAULT_SAMPLE_LEN,
            seed: 299_792_458,
            speculate: false,
        }
    }
}

/// A3 eviction intent. Absent prefill_cap = uncapped prefill (the
/// validated stage-2-only shape); armed at all only when at least one
/// cap is set.
#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvictionConfig {
    pub prefill_cap: Option<usize>,
    pub decode_cap: Option<usize>,
    pub recent: usize,
    pub sink: usize,
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            prefill_cap: None,
            decode_cap: None,
            recent: 512,
            sink: 4,
        }
    }
}

impl EvictionConfig {
    /// The engine knobs this intent arms; None when capless.
    pub(crate) fn to_settings(self) -> Option<EvictionSettings> {
        if self.prefill_cap.is_none() && self.decode_cap.is_none() {
            return None;
        }
        Some(EvictionSettings {
            prefill_cap: self.prefill_cap.unwrap_or(usize::MAX),
            decode_cap: self.decode_cap,
            sink_keep: self.sink,
            recent_keep: self.recent,
        })
    }
}

impl Settings {
    /// Derive runtime settings from declared intent, reactively: flash
    /// when the build, the resolved device, and the resolved dtype all
    /// support it; eviction as the config caps it. The observation pass
    /// (profile_attn) is invocation-armed, never config-derived.
    pub fn from_config(config: &Config) -> Self {
        let device_is_cuda = match config.device {
            DeviceConfig::Cuda => true,
            DeviceConfig::Auto => r::candle::cuda_is_available(),
            DeviceConfig::Cpu => false,
        };
        let dtype_supports_flash = match config.dtype {
            DtypeConfig::Bf16 => true,
            DtypeConfig::F32 => false,
            DtypeConfig::Auto => device_is_cuda,
        };
        Settings {
            use_flash_attn: cfg!(feature = "flash-attn") && device_is_cuda && dtype_supports_flash,
            profile_attn: false,
            eviction: config.eviction.and_then(EvictionConfig::to_settings),
            graph: false,
        }
    }
}

/// A settings profile file: an overlay over the from_config derivation.
/// Absent fields inherit the derivation; unknown fields are rejected.
#[derive(Debug, Clone, Copy, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SettingsPatch {
    pub(crate) use_flash_attn: Option<bool>,
    pub(crate) profile_attn: Option<bool>,
    pub(crate) graph: Option<bool>,
}

impl SettingsPatch {
    pub(crate) fn apply(self, settings: &mut Settings) {
        if let Some(use_flash_attn) = self.use_flash_attn {
            settings.use_flash_attn = use_flash_attn;
        }
        if let Some(profile_attn) = self.profile_attn {
            settings.profile_attn = profile_attn;
        }
        if let Some(graph) = self.graph {
            settings.graph = graph;
        }
    }
}

/// A -c/-s token that is purely a snake names a profile under the
/// profiles home; anything else is a filesystem path.
pub(crate) fn is_snake(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// $XDG_CONFIG_HOME/sourcetrait/almost (XDG spec fallback ~/.config).
fn profiles_home() -> PathBuf {
    let base = match std::env::var("XDG_CONFIG_HOME") {
        Ok(config_home) if !config_home.is_empty() => PathBuf::from(config_home),
        _ => {
            let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
            PathBuf::from(home).join(".config")
        }
    };
    base.join("sourcetrait").join("almost")
}

/// Minimal command-boundary path expansion (~/ to $HOME).
pub(crate) fn expand_path(token: &str) -> PathBuf {
    if let Some(rest) = token.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(token)
}

fn parse_config(path: &Path) -> AlmostResult<Config> {
    let raw = std::fs::read_to_string(path)?;
    match toml::from_str::<Config>(&raw) {
        Ok(config) => Ok(config),
        Err(error) => snafu::whatever!("config {}: {error}", path.display()),
    }
}

fn parse_settings_patch(path: &Path) -> AlmostResult<SettingsPatch> {
    let raw = std::fs::read_to_string(path)?;
    match toml::from_str::<SettingsPatch>(&raw) {
        Ok(patch) => Ok(patch),
        Err(error) => snafu::whatever!("settings {}: {error}", path.display()),
    }
}

/// Resolve the -c token: absent tries config/default.toml then falls
/// back to code defaults; a snake requires config/<snake>.toml; a path
/// requires that file. Returns the config plus its profile identity
/// (None for a path-loaded config - nothing to auto-pair against).
pub fn load_config(arg: Option<&str>) -> AlmostResult<(Config, Option<String>)> {
    match arg {
        None => {
            let path = profiles_home().join("config").join("default.toml");
            let config = if path.exists() {
                parse_config(&path)?
            } else {
                Config::default()
            };
            Ok((config, Some(String::from("default"))))
        }
        Some(token) if is_snake(token) => {
            let path = profiles_home().join("config").join(format!("{token}.toml"));
            snafu::ensure_whatever!(
                path.exists(),
                "config profile {token} not found at {}",
                path.display()
            );
            Ok((parse_config(&path)?, Some(String::from(token))))
        }
        Some(token) => {
            let path = expand_path(token);
            snafu::ensure_whatever!(
                path.exists(),
                "config file {} not found",
                path.display()
            );
            Ok((parse_config(&path)?, None))
        }
    }
}

/// Resolve the -s token over the from_config derivation: an explicit
/// snake/path is required to exist and overlays it; absent auto-pairs
/// settings/<profile>.toml when the config has a profile identity, and
/// derives reactively otherwise (or when the pair does not exist).
pub fn load_settings(
    arg: Option<&str>,
    config_profile: Option<&str>,
    config: &Config,
) -> AlmostResult<Settings> {
    let mut settings = Settings::from_config(config);
    match arg {
        Some(token) => {
            let path = if is_snake(token) {
                let path = profiles_home().join("settings").join(format!("{token}.toml"));
                snafu::ensure_whatever!(
                    path.exists(),
                    "settings profile {token} not found at {}",
                    path.display()
                );
                path
            } else {
                let path = expand_path(token);
                snafu::ensure_whatever!(
                    path.exists(),
                    "settings file {} not found",
                    path.display()
                );
                path
            };
            parse_settings_patch(&path)?.apply(&mut settings);
        }
        None => {
            if let Some(profile) = config_profile {
                let path = profiles_home().join("settings").join(format!("{profile}.toml"));
                if path.exists() {
                    parse_settings_patch(&path)?.apply(&mut settings);
                }
            }
        }
    }
    Ok(settings)
}

/// Resolve the config's device intent (Auto = cuda when available).
pub fn resolve_device(config: &Config) -> AlmostResult<candle_core::Device> {
    match config.device {
        DeviceConfig::Cpu => Ok(candle_core::Device::Cpu),
        DeviceConfig::Cuda => {
            snafu::ensure_whatever!(
                r::candle::cuda_is_available(),
                "config demands cuda but no cuda device is available"
            );
            Ok(candle_core::Device::new_cuda(0)?)
        }
        DeviceConfig::Auto => {
            if r::candle::cuda_is_available() {
                Ok(candle_core::Device::new_cuda(0)?)
            } else {
                Ok(candle_core::Device::Cpu)
            }
        }
    }
}

/// Resolve the config's dtype intent against the resolved device.
pub fn resolve_dtype(config: &Config, device: &candle_core::Device) -> candle_core::DType {
    match config.dtype {
        DtypeConfig::Bf16 => candle_core::DType::BF16,
        DtypeConfig::F32 => candle_core::DType::F32,
        DtypeConfig::Auto => {
            if matches!(device, candle_core::Device::Cpu) {
                candle_core::DType::F32
            } else {
                candle_core::DType::BF16
            }
        }
    }
}
