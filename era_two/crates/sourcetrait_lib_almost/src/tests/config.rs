//! Profile-resolution and overlay-merge locks for the -d/-c/-s
//! framework (checkpoint-free; filesystem cases ride a temp dir).
use crate::*;

fn temp_root(tag: &str) -> PathBuf {
    let root = env::temp_dir().join(format!(
        "lib_almost_config_{tag}_{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root).expect("clean temp root");
    }
    fs::create_dir_all(&root).expect("create temp root");
    root
}

#[test]
fn snake_tokens_resolve_under_a_custom_root() {
    let profile = ConfigProfile::try_from_dir("/opt/suite", "bench").expect("resolves");
    assert_eq!(
        profile.lib_config(),
        PathBuf::from("/opt/suite/config/bench/lib.toml")
    );
    assert_eq!(
        profile.cli_config(),
        PathBuf::from("/opt/suite/config/bench/cli.toml")
    );
    let settings = SettingsProfile::try_from_dir("/opt/suite", "bench").expect("resolves");
    assert_eq!(
        settings.lib_settings(),
        PathBuf::from("/opt/suite/settings/bench/lib.toml")
    );
    assert_eq!(
        settings.baseline_settings(),
        PathBuf::from("/opt/suite/settings/bench/baseline.toml")
    );
}

#[test]
fn path_tokens_are_used_as_the_profile_directory() {
    let profile = ConfigProfile::try_from("/opt/elsewhere/bench").expect("resolves");
    assert_eq!(profile.dir(), Path::new("/opt/elsewhere/bench"));
    assert_eq!(
        profile.lib_config(),
        PathBuf::from("/opt/elsewhere/bench/lib.toml")
    );
}

#[test]
fn models_dir_defaults_to_the_data_home_and_overrides_by_file() {
    let config = LibConfig::default();
    assert!(config.models_dir.ends_with(consts::MODELS_HOME_RELATIVE));
    assert!(config.model_dir().ends_with(format!(
        "{}/{}/{}",
        consts::MODELS_HOME_RELATIVE,
        consts::MODEL_AUTHOR,
        consts::DPO_MODEL_NAME
    )));
    let shell: LibConfigToml =
        toml::from_str("models_dir = \"/mnt/models\"\n").expect("parses");
    let config: LibConfig = shell.try_into().expect("merges");
    assert_eq!(config.models_dir, PathBuf::from("/mnt/models"));
}

#[test]
fn path_strings_expand_leading_tilde_and_env_vars() {
    let shell: LibConfigToml =
        toml::from_str("models_dir = \"~/models\"\n").expect("parses");
    let config: LibConfig = shell.try_into().expect("expands");
    assert!(!config.models_dir.to_string_lossy().contains('~'));
    assert!(config.models_dir.is_absolute());
    assert!(config.models_dir.ends_with("models"));

    let shell: LibConfigToml =
        toml::from_str("models_dir = \"$XDG_DATA_HOME/alt/models\"\n").expect("parses");
    let config: LibConfig = shell.try_into().expect("expands");
    assert!(!config.models_dir.to_string_lossy().contains('$'));
    assert!(config.models_dir.ends_with("alt/models"));
}

#[test]
fn embedded_defaults_carry_the_card_posture() {
    let config = LibConfig::default();
    assert_eq!(
        config.model,
        format!("{}/{}", consts::MODEL_AUTHOR, consts::DPO_MODEL_NAME),
        "the defaults file carries the author-qualified coordinate"
    );
    let settings = LibSettings::default();
    assert!(settings.use_flash_attn);
    assert!(!settings.graph);
    assert_eq!(settings.generation.temperature, Some(0.6));
    assert_eq!(settings.generation.top_p, Some(0.95));
    assert_eq!(settings.generation.seed, 299_792_458);
    assert_eq!(settings.generation.sample_len, 32_768);
    assert!(settings.generation.chat);
    assert!(!settings.generation.ignore_stops);
}

#[test]
fn partial_file_overlays_the_embedded_base() {
    let shell: LibSettingsToml =
        toml::from_str("[generation]\nsample_len = 48\n").expect("parses");
    let settings: LibSettings = shell.try_into().expect("merges");
    assert_eq!(settings.generation.sample_len, 48);
    assert_eq!(
        settings.generation.temperature,
        Some(0.6),
        "untouched keys keep the base"
    );
    assert!(settings.use_flash_attn);
}

#[test]
fn greedy_forces_argmax_over_sampling_params() {
    let shell: LibSettingsToml =
        toml::from_str("[generation]\ngreedy = true\ntemperature = 0.9\n").expect("parses");
    let settings: LibSettings = shell.try_into().expect("merges");
    assert_eq!(settings.generation.temperature, None);
    assert_eq!(settings.generation.top_p, None);
}

#[test]
fn unknown_keys_are_rejected() {
    assert!(toml::from_str::<LibConfigToml>("modle = \"typo\"\n").is_err());
    assert!(toml::from_str::<LibSettingsToml>("graf = true\n").is_err());
    assert!(
        toml::from_str::<LibSettingsToml>("[generation]\ntemprature = 0.5\n").is_err()
    );
}

#[test]
fn unspecified_token_prefers_the_default_profile_file() {
    let root = temp_root("default_profile");
    let settings_dir = root.join("settings/default");
    fs::create_dir_all(&settings_dir).expect("profile dir");
    fs::write(settings_dir.join("lib.toml"), "[generation]\nseed = 7\n").expect("write");
    let settings =
        LibSettings::load_from_dir(Some(&root), None::<&Path>).expect("default profile");
    assert_eq!(settings.generation.seed, 7);
    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn unspecified_token_without_a_default_file_is_the_embedded_base() {
    let root = temp_root("no_default");
    let config = LibConfig::load_from_dir(Some(&root), None::<&Path>).expect("embedded");
    assert_eq!(
        config.model,
        format!("{}/{}", consts::MODEL_AUTHOR, consts::DPO_MODEL_NAME)
    );
    let settings = LibSettings::load_from_dir(Some(&root), None::<&Path>).expect("embedded");
    assert!(!settings.graph);
    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn the_defaults_name_is_reserved_and_never_reads_the_fs() {
    let root = temp_root("reserved");
    let poisoned = root.join("settings/defaults");
    fs::create_dir_all(&poisoned).expect("dir");
    fs::write(poisoned.join("lib.toml"), "[generation]\nseed = 13\n").expect("write");
    let settings = LibSettings::load_from_dir(Some(&root), Some("defaults")).expect("embedded");
    assert_eq!(
        settings.generation.seed, 299_792_458,
        "the fs copy is ignored"
    );
    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn explicit_named_profiles_error_when_missing() {
    let root = temp_root("missing_named");
    assert!(LibConfig::load_from_dir(Some(&root), Some("bench")).is_err());
    assert!(LibSettings::load_from_dir(Some(&root), Some("bench")).is_err());
    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn named_profiles_load_their_component_file() {
    let root = temp_root("named");
    let config_dir = root.join("config/bench");
    fs::create_dir_all(&config_dir).expect("dir");
    fs::write(config_dir.join("lib.toml"), "model = \"Olmo-Hybrid-7B\"\n").expect("write");
    let settings_dir = root.join("settings/bench");
    fs::create_dir_all(&settings_dir).expect("dir");
    fs::write(
        settings_dir.join("lib.toml"),
        "graph = true\n\n[generation]\ngreedy = true\nsample_len = 128\n",
    )
    .expect("write");
    let config = LibConfig::load_from_dir(Some(&root), Some("bench")).expect("config");
    assert_eq!(config.model, "Olmo-Hybrid-7B");
    let settings = LibSettings::load_from_dir(Some(&root), Some("bench")).expect("settings");
    assert!(settings.graph);
    assert!(settings.use_flash_attn, "untouched keys keep the base");
    assert_eq!(settings.generation.temperature, None);
    assert_eq!(settings.generation.sample_len, 128);
    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn eviction_arms_by_decode_cap_with_protection_defaults() {
    let shell: LibSettingsToml =
        toml::from_str("[eviction]\ndecode_cap = 2048\n").expect("parses");
    let settings: LibSettings = shell.try_into().expect("merges");
    let eviction = settings.eviction.expect("armed");
    assert_eq!(eviction.decode_cap, 2048);
    assert_eq!(eviction.recent, 512, "the era-one v1 default");
    assert_eq!(eviction.sink, 4, "the era-one v1 default");

    let shell: LibSettingsToml =
        toml::from_str("[eviction]\ndecode_cap = 1024\nrecent = 256\nsink = 8\n")
            .expect("parses");
    let settings: LibSettings = shell.try_into().expect("merges");
    assert_eq!(
        settings.eviction,
        Some(EvictionSettings {
            decode_cap: 1024,
            recent: 256,
            sink: 8
        })
    );
}

#[test]
fn eviction_rejects_capless_tables_and_swallowed_caps() {
    let shell: LibSettingsToml = toml::from_str("[eviction]\nrecent = 256\n").expect("parses");
    assert!(LibSettings::try_from(shell).is_err(), "no decode_cap");
    let shell: LibSettingsToml =
        toml::from_str("[eviction]\ndecode_cap = 512\nrecent = 512\n").expect("parses");
    assert!(
        LibSettings::try_from(shell).is_err(),
        "cap must exceed recent + sink"
    );
    assert!(
        toml::from_str::<LibSettingsToml>("[eviction]\ndecodecap = 1\n").is_err(),
        "unknown eviction keys are rejected"
    );
}

#[test]
fn embedded_defaults_stay_eviction_free() {
    assert!(LibSettings::default().eviction.is_none());
}

#[test]
fn path_tokens_load_the_exact_file_and_error_when_absent() {
    let root = temp_root("path_tokens");
    let file = root.join("one_off.toml");
    fs::write(&file, "[generation]\nseed = 99\n").expect("write");
    let settings =
        LibSettings::load_from_dir(None::<&Path>, Some(&file)).expect("exact file loads");
    assert_eq!(settings.generation.seed, 99);
    let absent = root.join("absent.toml");
    assert!(LibSettings::load_from_dir(None::<&Path>, Some(&absent)).is_err());
    fs::remove_dir_all(&root).expect("cleanup");
}
