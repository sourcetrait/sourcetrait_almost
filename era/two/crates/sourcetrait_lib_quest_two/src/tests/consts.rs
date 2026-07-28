//! The training version reaches the binary as the manifest declares it.
use crate::consts::TRAINING_VERSION;

#[test]
fn the_training_version_is_the_one_the_manifest_pins() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest).expect("the manifest reads");
    let parsed: toml::Table = toml::from_str(&text).expect("the manifest parses");
    let declared = parsed
        .get("package")
        .and_then(|package| package.get("metadata"))
        .and_then(|metadata| metadata.get("training"))
        .and_then(|training| training.get("version"))
        .and_then(|version| version.as_str())
        .expect("[package.metadata.training] version is declared");

    assert!(!TRAINING_VERSION.is_empty(), "a version reached the binary");
    assert_eq!(
        TRAINING_VERSION, declared,
        "the compiled version is the one the manifest pins, so a generated \
         set cannot name a version this binary was not built from"
    );
}
