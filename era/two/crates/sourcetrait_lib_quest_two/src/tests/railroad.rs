//! Railroad locks: what laying one produces, and how REVs are numbered.
use crate::railroad::{
    BRANCH,
    Railroad,
    TrainNom,
};

/// A scratch root that removes itself, so a lock leaves no residue.
struct Scratch {
    root: std::path::PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "quest_railroad_{name}_{}",
            std::process::id()
        ));
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

/// Read git's own answer back out of a laid railroad.
fn git(dir: &std::path::Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .expect("git runs");
    assert!(output.status.success(), "git {args:?} failed");
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

#[test]
fn a_nom_is_a_plain_path_segment() {
    for _ in 0..64 {
        let nom = TrainNom::fresh().expect("mints");
        let text = nom.as_str();
        assert!(!text.is_empty(), "a nom is never empty");
        assert!(
            text.chars().all(|c| c.is_ascii_alphanumeric()),
            "{text:?} carries something a path would resolve"
        );
    }
}

#[test]
fn laying_one_opens_the_train_branch_on_an_init_commit() {
    let scratch = Scratch::new("lay");
    let road = Railroad::lay_in(&scratch.root).expect("lays");

    assert_eq!(
        road.dir(),
        scratch.root.join(road.nom().as_str()).as_path(),
        "a railroad stands at its own nom"
    );
    assert_eq!(
        git(road.dir(), &["rev-parse", "--abbrev-ref", "HEAD"]),
        BRANCH
    );
    assert_eq!(git(road.dir(), &["log", "-1", "--format=%s"]), "init");
    assert_eq!(
        git(road.dir(), &["ls-tree", "--name-only", BRANCH]),
        ".gitignore"
    );
    assert_eq!(
        git(road.dir(), &["status", "--porcelain"]),
        "",
        "the base commit leaves nothing standing"
    );
}

#[test]
fn revs_number_from_the_history_rather_than_from_a_caller() {
    let scratch = Scratch::new("revs");
    let road = Railroad::lay_in(&scratch.root).expect("lays");

    std::fs::write(road.dir().join("set.nuon"), "{a: 1}").expect("writes");
    assert_eq!(road.commit().expect("commits"), 1);

    std::fs::write(road.dir().join("set.nuon"), "{a: 2}").expect("rewrites");
    assert_eq!(road.commit().expect("commits"), 2);

    assert_eq!(
        git(road.dir(), &["log", "--format=%s"]),
        "REV 2\nREV 1\ninit"
    );
}

#[test]
fn a_reopened_railroad_continues_the_same_numbering() {
    let scratch = Scratch::new("reopen");
    let laid = Railroad::lay_in(&scratch.root).expect("lays");
    std::fs::write(laid.dir().join("set.nuon"), "{a: 1}").expect("writes");
    assert_eq!(laid.commit().expect("commits"), 1);

    let reopened = Railroad::at(laid.dir()).expect("reopens");
    assert_eq!(reopened.nom(), laid.nom(), "the directory name is the nom");

    std::fs::write(reopened.dir().join("set.nuon"), "{a: 2}").expect("writes");
    assert_eq!(
        reopened.commit().expect("commits"),
        2,
        "a later process picks the numbering up rather than restarting it"
    );
}

#[test]
fn an_ordinary_directory_is_refused_as_a_railroad() {
    let scratch = Scratch::new("refuse");
    let plain = scratch.root.join("not_a_railroad");
    std::fs::create_dir_all(&plain).expect("creates");
    assert!(
        Railroad::at(&plain).is_err(),
        "a directory carrying no repository is not a railroad"
    );
}

#[test]
fn consecutive_railroads_take_distinct_noms() {
    let scratch = Scratch::new("distinct");
    let first = Railroad::lay_in(&scratch.root).expect("lays");
    let second = Railroad::lay_in(&scratch.root).expect("lays");
    assert_ne!(first.nom(), second.nom());
    assert!(first.dir().is_dir() && second.dir().is_dir());
}
