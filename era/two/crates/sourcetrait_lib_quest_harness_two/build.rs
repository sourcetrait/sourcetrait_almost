//! Surface the training scheme's version from the manifest that pins it.
//!
//! Cargo does not hand `[package.metadata.*]` to compiled code, so a
//! consumer would otherwise have to parse a manifest at runtime and could
//! then report a version the binary was not built from. Reading it here
//! makes the two the same fact by construction.
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo names the manifest dir");
    let manifest = std::path::Path::new(&dir).join("Cargo.toml");
    println!("cargo::rerun-if-changed={}", manifest.display());

    let text = std::fs::read_to_string(&manifest).expect("the manifest reads");
    let parsed: toml::Table = toml::from_str(&text).expect("the manifest parses");
    let version = parsed
        .get("package")
        .and_then(|package| package.get("metadata"))
        .and_then(|metadata| metadata.get("training"))
        .and_then(|training| training.get("version"))
        .and_then(|version| version.as_str())
        .expect("[package.metadata.training] version is declared");

    println!("cargo::rustc-env=QUEST_TRAINING_VERSION={version}");
}
