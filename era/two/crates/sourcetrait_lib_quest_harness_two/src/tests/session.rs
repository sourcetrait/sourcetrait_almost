//! Session log locks: where it lands, and what a nom may not do.
use crate::session::{
    LOG_FILE,
    SessionLog,
};

/// A scratch root that removes itself, so a lock leaves no residue.
struct Scratch {
    root: std::path::PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("quest_session_{name}"));
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

#[test]
fn a_log_lands_one_directory_deep_per_segment() {
    let scratch = Scratch::new("nested");
    let log = SessionLog::open(&scratch.root, &["thinkspace", "session"]).expect("opens");
    assert_eq!(
        log.path(),
        scratch.root.join("thinkspace").join("session").join(LOG_FILE)
    );
}

#[test]
fn records_append_in_order_and_read_back() {
    let scratch = Scratch::new("append");
    let log = SessionLog::open(&scratch.root, &["session"]).expect("opens");

    assert_eq!(log.read().expect("reads"), "", "an unwritten log is empty");

    log.append("assembled", "the prompt").expect("appends");
    log.append("emission", "the answer").expect("appends");

    let text = log.read().expect("reads");
    assert_eq!(
        text,
        "== assembled ==\nthe prompt\n== emission ==\nthe answer\n"
    );
}

#[test]
fn a_nom_cannot_climb_out_of_its_root() {
    let scratch = Scratch::new("escape");
    for hostile in ["..", "../elsewhere", "a/b", ".", "", "a\\b"] {
        assert!(
            SessionLog::open(&scratch.root, &[hostile]).is_err(),
            "{hostile:?} must be refused as a session segment"
        );
    }
}

#[test]
fn reopening_a_session_continues_its_log() {
    let scratch = Scratch::new("reopen");
    SessionLog::open(&scratch.root, &["s"])
        .expect("opens")
        .append("first", "one")
        .expect("appends");
    let second = SessionLog::open(&scratch.root, &["s"]).expect("reopens");
    second.append("second", "two").expect("appends");

    let text = second.read().expect("reads");
    assert!(text.starts_with("== first =="), "the earlier run survives");
    assert!(text.contains("== second =="));
}
