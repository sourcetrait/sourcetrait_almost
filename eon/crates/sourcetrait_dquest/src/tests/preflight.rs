//! Preflight locks: what the shipped profile promises, and what the
//! daemon requires before it is allowed to serve.
use crate::preflight::{
    PROFILE,
    PROFILE_TEXT,
    SECRET_DATA_ENV,
    ensure_profile,
    material_exists,
    named_home,
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

    /// Stand in for `srcert install`, optionally leaving one file out.
    fn install_material(&self, omit: Option<&str>) {
        let dir = sourcetrait_cert_lib::secret_certs_dir(&self.root, PROFILE);
        std::fs::create_dir_all(&dir).expect("secret certs dir");
        for file in [
            format!("authority_{PROFILE}.key.pem"),
            format!("authority_{PROFILE}.pem"),
            format!("entity_{PROFILE}.key.pem"),
            format!("entity_{PROFILE}.pem"),
        ] {
            if omit == Some(file.as_str()) {
                continue;
            }
            std::fs::write(dir.join(&file), "placed").expect("artifact");
        }
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

    assert!(profile.written, "the live profile was absent");
    assert!(profile.live.is_file());
    assert_eq!(
        std::fs::read_to_string(&profile.live).expect("reads"),
        PROFILE_TEXT,
        "what was written is what the binary ships"
    );
}

#[test]
fn a_later_start_leaves_a_customised_profile_alone() {
    let scratch = Scratch::new("later");
    let first = ensure_profile(&scratch.root).expect("installs");
    let customised = PROFILE_TEXT.replace("3650", "30");
    std::fs::write(&first.live, &customised).expect("the operator edits it");

    let again = ensure_profile(&scratch.root).expect("re-checks");

    assert!(!again.written, "an existing live profile is left alone");
    assert_eq!(
        std::fs::read_to_string(&again.live).expect("reads"),
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

/// The defect CheckExists closes: a start after the first one used to
/// find the profile present and proceed with no certificate anywhere.
#[test]
fn a_written_profile_alone_does_not_satisfy_the_daemon() {
    let scratch = Scratch::new("profile_only");
    let profile = ensure_profile(&scratch.root).expect("installs");

    assert!(profile.written, "the profile is in place");
    assert!(
        !material_exists(&scratch.root),
        "a profile is not a certificate"
    );
}

#[test]
fn installed_material_satisfies_the_daemon() {
    let scratch = Scratch::new("material");
    scratch.install_material(None);
    assert!(material_exists(&scratch.root));
}

/// A half-installed tree must not read as ready, because what it is
/// missing surfaces at a handshake rather than at startup.
#[test]
fn a_partial_install_does_not_satisfy_the_daemon() {
    let scratch = Scratch::new("partial");
    scratch.install_material(Some(&format!("entity_{PROFILE}.key.pem")));
    assert!(!material_exists(&scratch.root));
}

/// Unset and blank are the same condition, and neither may be read as a
/// relative path from wherever the daemon happened to start.
#[test]
fn an_unnamed_secret_home_is_refused_and_names_its_variable() {
    for absent in [None, Some(""), Some("   ")] {
        let error = named_home(absent).expect_err("no secret data home");
        let text = error.to_string();
        assert!(text.contains(SECRET_DATA_ENV), "{text}");
    }
    assert_eq!(
        named_home(Some("/secret/data")).expect("named"),
        std::path::PathBuf::from("/secret/data")
    );
}
