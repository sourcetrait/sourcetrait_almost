//! Manager locks: a place in the registry that cannot be forgotten.
use crate::manager::SessionManager;

#[test]
fn a_ticket_holds_a_place_and_dropping_it_gives_the_place_back() {
    let manager = SessionManager::for_user("box", None);
    assert_eq!(manager.live(), 0);

    let first = manager.admit();
    let second = manager.admit();
    assert_eq!(manager.live(), 2);
    assert_ne!(first.nom(), second.nom(), "each session names itself");

    drop(first);
    assert_eq!(manager.live(), 1);
    drop(second);
    assert_eq!(manager.live(), 0);
}

/// Deregistration is a Drop rather than a call, so a session that ends
/// by unwinding cannot leave itself behind in the registry.
#[test]
fn a_session_that_panics_still_gives_its_place_back() {
    let manager = SessionManager::for_user("box", None);
    let inner = manager.clone();
    let outcome = std::panic::catch_unwind(move || {
        let _ticket = inner.admit();
        assert_eq!(inner.live(), 1);
        panic!("a session task fell over");
    });

    assert!(outcome.is_err(), "the panic was caught rather than missed");
    assert_eq!(manager.live(), 0, "the ticket dropped while unwinding");
}

/// Clones share one registry, which is what lets every accepted
/// connection admit against the same one.
#[test]
fn clones_share_the_registry_and_the_space() {
    let manager = SessionManager::for_user("box", None);
    let other = manager.clone();
    let _ticket = other.admit();

    assert_eq!(manager.live(), 1, "a clone admits into the same registry");
    assert_eq!(manager.thinkspace(), other.thinkspace());
    assert_eq!(
        manager.thinkspace(),
        SessionManager::for_user("box", None).thinkspace(),
        "the space follows the user rather than the registry"
    );
    assert_ne!(
        manager.thinkspace(),
        SessionManager::for_user("someone_else", None).thinkspace()
    );
}

/// A log is addressed by SPACE and then by SESSION, both spelled from
/// noms rather than from anything a client sent.
#[test]
fn a_session_log_is_addressed_by_space_then_session() {
    let scratch = crate::tests::material::Scratch::make("manager_log");
    let manager = SessionManager::for_user("box", Some(scratch.root.clone()));
    let ticket = manager.admit();
    let log = manager.log(&ticket).expect("a log opens under the root");

    assert_eq!(
        log.path(),
        scratch
            .root
            .join(manager.thinkspace().as_str())
            .join(ticket.nom().as_str())
            .join(crate::lib::session::LOG_FILE)
    );

    log.append("turn", "say hello").expect("appends");
    assert!(log.read().expect("reads back").contains("say hello"));
}

/// No root means no log rather than a failure, which is what keeps a
/// missing cache home from refusing connections.
#[test]
fn without_a_root_a_session_simply_is_not_logged() {
    let manager = SessionManager::for_user("box", None);
    let ticket = manager.admit();
    assert!(manager.log(&ticket).is_none());
}

#[test]
fn the_listing_names_exactly_the_live_sessions() {
    let manager = SessionManager::for_user("box", None);
    let first = manager.admit();
    let second = manager.admit();

    let mut expected = vec![first.nom().clone(), second.nom().clone()];
    expected.sort();
    assert_eq!(manager.sessions(), expected);

    drop(first);
    assert_eq!(manager.sessions(), vec![second.nom().clone()]);
}
