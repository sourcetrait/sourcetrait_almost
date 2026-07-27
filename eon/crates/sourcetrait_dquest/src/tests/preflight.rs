//! Preflight locks: what the shipped profile promises, and when it is
//! written.
use crate::preflight::{
    PROFILE,
    PROFILE_TEXT,
    Profile,
    ensure_profile,
};

/// A scratch root that removes itself, so a lock leaves no residue.
struct Scratch {
    root: std::path::PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("dquest_preflight_{name}"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch root");
        Self { root }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// The load-bearing property of the shipped profile. A leaf without
/// client authentication is refused at the far end of a mutual-TLS
/// handshake, and nothing before the handshake says so.
#[test]
fn the_shipped_profile_carries_both_usages() {
    let config = sourcetrait_cert_lib::config::CertGenConfig::parse(
        PROFILE_TEXT,
        std::path::Path::new("quest.toml"),
    )
    .expect("the shipped profile parses");

    assert_eq!(config.name, PROFILE, "the name is the install segment");
    assert!(
        config
            .entity
            .usages
            .contains(&sourcetrait_cert_lib::EntityUsage::Server),
        "the daemon presents this leaf as a server"
    );
    assert!(
        config
            .entity
            .usages
            .contains(&sourcetrait_cert_lib::EntityUsage::Client),
        "the bridge presents this leaf as a client"
    );
}

/// Hostname verification reads the alternative names rather than the
/// common name, so a loopback caller fails without every spelling.
#[test]
fn the_shipped_profile_answers_to_every_loopback_spelling() {
    let config = sourcetrait_cert_lib::config::CertGenConfig::parse(
        PROFILE_TEXT,
        std::path::Path::new("quest.toml"),
    )
    .expect("the shipped profile parses");

    for name in ["127.0.0.1", "::1", "localhost"] {
        assert!(
            config
                .entity
                .subject_alt_names
                .iter()
                .any(|san| san == name),
            "{name} must be a subject alternative name"
        );
    }
}

#[test]
fn a_first_start_writes_the_profile() {
    let scratch = Scratch::new("first");
    let profile = ensure_profile(&scratch.root).expect("installs");

    assert!(matches!(profile, Profile::Written { .. }));
    assert!(profile.live().is_file());
    assert_eq!(
        std::fs::read_to_string(profile.live()).expect("reads"),
        PROFILE_TEXT,
        "what was written is what the binary ships"
    );
}

#[test]
fn a_later_start_leaves_a_customised_profile_alone() {
    let scratch = Scratch::new("later");
    let first = ensure_profile(&scratch.root).expect("installs");
    let customised = PROFILE_TEXT.replace("3650", "30");
    std::fs::write(first.live(), &customised).expect("the operator edits it");

    let again = ensure_profile(&scratch.root).expect("re-checks");

    assert!(matches!(again, Profile::Present { .. }));
    assert_eq!(
        std::fs::read_to_string(again.live()).expect("reads"),
        customised,
        "the customisation survives a restart"
    );
}

/// The reference copy exists to be diffed against, so a stale one is
/// worse than none.
#[test]
fn the_reference_copy_refreshes_even_when_the_live_one_does_not() {
    let scratch = Scratch::new("reference");
    ensure_profile(&scratch.root).expect("installs");

    let reference =
        sourcetrait_cert_lib::default_profile_path(&scratch.root, PROFILE).expect("path");
    std::fs::write(&reference, "stale = true").expect("something staled it");

    ensure_profile(&scratch.root).expect("re-checks");

    assert_eq!(
        std::fs::read_to_string(&reference).expect("reads"),
        PROFILE_TEXT,
        "the reference says what this build ships"
    );
}
