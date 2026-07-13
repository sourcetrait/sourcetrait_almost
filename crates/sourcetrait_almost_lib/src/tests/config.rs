use crate::config::{is_snake, SettingsPatch};
use crate::{Config, DeviceConfig, DtypeConfig, EvictionConfig, Settings};

#[test]
fn partial_config_overlays_code_defaults() {
    let config: Config = toml::from_str("[generation]\ngreedy = true").unwrap();
    assert!(config.generation.greedy);
    assert_eq!(config.generation.temperature, crate::consts::DEFAULT_TEMPERATURE);
    assert_eq!(config.model_id, crate::consts::DEFAULT_MODEL_ID);
    assert_eq!(config.device, DeviceConfig::Auto);
}

#[test]
fn unknown_config_fields_are_rejected() {
    assert!(toml::from_str::<Config>("banana = 1").is_err());
    assert!(toml::from_str::<Config>("[generation]\nbanana = 1").is_err());
}

#[test]
fn capless_eviction_is_unarmed() {
    let config: Config = toml::from_str("[eviction]\nrecent = 256").unwrap();
    assert!(config.eviction.unwrap().to_settings().is_none());
}

#[test]
fn stage_two_only_eviction_maps_to_uncapped_prefill() {
    let eviction = EvictionConfig {
        prefill_cap: None,
        decode_cap: Some(2048),
        recent: 512,
        sink: 4,
    };
    let settings = eviction.to_settings().unwrap();
    assert_eq!(settings.prefill_cap, usize::MAX);
    assert_eq!(settings.decode_cap, Some(2048));
    assert!(settings.validate().is_ok());
}

#[test]
fn from_config_derives_flash_reactively() {
    let cpu = Config {
        device: DeviceConfig::Cpu,
        ..Config::default()
    };
    assert!(!Settings::from_config(&cpu).use_flash_attn);

    let f32_on_cuda = Config {
        device: DeviceConfig::Cuda,
        dtype: DtypeConfig::F32,
        ..Config::default()
    };
    assert!(!Settings::from_config(&f32_on_cuda).use_flash_attn);

    let bf16_on_cuda = Config {
        device: DeviceConfig::Cuda,
        dtype: DtypeConfig::Bf16,
        ..Config::default()
    };
    assert_eq!(
        Settings::from_config(&bf16_on_cuda).use_flash_attn,
        cfg!(feature = "flash-attn")
    );
}

#[test]
fn settings_patch_overlays_only_present_fields() {
    let mut settings = Settings {
        use_flash_attn: true,
        profile_attn: false,
        eviction: None,
    };
    let patch: SettingsPatch = toml::from_str("use_flash_attn = false").unwrap();
    patch.apply(&mut settings);
    assert!(!settings.use_flash_attn);
    assert!(!settings.profile_attn);
    assert!(toml::from_str::<SettingsPatch>("banana = 1").is_err());
}

#[test]
fn snake_detection_is_strict() {
    assert!(is_snake("evict_v1"));
    assert!(is_snake("dog2"));
    assert!(!is_snake("dog.toml"));
    assert!(!is_snake("a/b"));
    assert!(!is_snake("Dog"));
    assert!(!is_snake(""));
    assert!(!is_snake("~"));
}
