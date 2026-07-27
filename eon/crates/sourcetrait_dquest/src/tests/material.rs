//! Throwaway filesystem state, so a lock depends on none that is
//! installed and leaves none behind.

/// A scratch directory that removes itself.
pub(crate) struct Scratch {
    pub(crate) root: std::path::PathBuf,
}

impl Scratch {
    pub(crate) fn make(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("dquest_scratch_{name}"));
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

pub(crate) struct Material {
    root: std::path::PathBuf,
    pub(crate) files: sourcetrait_cert_lib::CertFiles,
}

impl Material {
    pub(crate) fn mint(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("dquest_material_{name}"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch root");
        let text = format!(
            "name = \"{name}\"\n\
             organization = \"SourceTrait\"\n\
             validity_days = 30\n\
             \n\
             [authority]\n\
             common_name = \"SourceTrait Daemon Test CA\"\n\
             \n\
             [entity]\n\
             common_name = \"localhost\"\n\
             subject_alt_names = [\"127.0.0.1\", \"::1\", \"localhost\"]\n\
             usages = [\"server\", \"client\"]\n"
        );
        let config = sourcetrait_cert_lib::config::CertGenConfig::parse(
            &text,
            std::path::Path::new("test.toml"),
        )
        .expect("the test profile parses");
        let files = sourcetrait_cert_lib::generate(&config, &root).expect("mints");
        Self { root, files }
    }
}

impl Drop for Material {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
