//! Nom locks: stability where it is needed, freshness where it is, and
//! a plain path segment either way.
use crate::nom::{
    SessionNom,
    ThinkspaceNom,
};

/// The same user must find the same space across restarts, or a session
/// log lands somewhere new every time the daemon comes up.
#[test]
fn a_username_names_one_space_every_time() {
    assert_eq!(ThinkspaceNom::of("box"), ThinkspaceNom::of("box"));
    assert_ne!(ThinkspaceNom::of("box"), ThinkspaceNom::of("roy"));
}

#[test]
fn a_session_nom_is_fresh_each_time() {
    let mut seen = std::collections::HashSet::new();
    for _ in 0..1_000 {
        assert!(
            seen.insert(SessionNom::fresh().as_str().to_string()),
            "a session nom repeated"
        );
    }
}

/// The property that lets a nom be joined to a log root with no
/// sanitiser between them: the alphabet carries nothing a path reads.
#[test]
fn a_nom_is_a_plain_path_segment_by_construction() {
    let mut noms: Vec<String> = vec![
        ThinkspaceNom::of("box").as_str().to_string(),
        ThinkspaceNom::of("").as_str().to_string(),
        ThinkspaceNom::of("../elsewhere").as_str().to_string(),
        ThinkspaceNom::of("a/b").as_str().to_string(),
        ThinkspaceNom::of("..").as_str().to_string(),
    ];
    noms.extend((0..64).map(|_| SessionNom::fresh().as_str().to_string()));

    for nom in noms {
        assert!(!nom.is_empty(), "a nom is never empty");
        assert!(nom != "." && nom != "..", "{nom}");
        assert!(
            nom.chars().all(|c| c.is_ascii_alphanumeric()),
            "{nom} carries something a path would read"
        );
    }
}

/// A hostile username cannot climb out, because what comes back is a
/// hash rather than the name - which is why nothing here sanitises.
#[test]
fn a_hostile_username_produces_an_ordinary_nom() {
    let hostile = ThinkspaceNom::of("../../etc");
    assert!(!hostile.as_str().contains('/'));
    assert!(!hostile.as_str().contains('.'));
    assert_eq!(hostile, ThinkspaceNom::of("../../etc"), "still stable");
}
