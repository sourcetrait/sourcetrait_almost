//! Manager locks: a place in the registry that cannot be forgotten.
use crate::manager::SessionManager;

/// A turn owner that echoes, standing in where the locks are about the
/// registry rather than about what a turn means.
#[derive(Default)]
pub(crate) struct StubThinkHarness;

impl crate::bridge::all::ThinkHarness for StubThinkHarness {
    fn assemble(
        &mut self,
        request: &crate::bridge::InferRequest,
    ) -> Result<String, String> {
        Ok(request
            .text
            .as_ref()
            .map(|text| text.0.clone())
            .unwrap_or_default())
    }

    fn step(
        &mut self,
        emission: &str,
        insufficient: bool,
    ) -> Result<crate::bridge::all::Step, String> {
        if insufficient {
            return Ok(crate::bridge::all::Step::Insufficient);
        }
        Ok(crate::bridge::all::Step::Answered {
            output: None,
            text: Some(crate::bridge::InferText(emission.to_string())),
            config: None,
        })
    }

    fn keeps_conversation(&self) -> bool {
        false
    }
}

#[test]
fn a_ticket_holds_a_place_and_dropping_it_gives_the_place_back() {
    let manager = SessionManager::for_user("box", None, StubThinkHarness);
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
    let manager = SessionManager::for_user("box", None, StubThinkHarness);
    let inner = manager.clone();
    // The manager now holds a turn owner behind a mutex, which is not
    // unwind-safe by inference. What is under test is the ticket's Drop,
    // and nothing here observes the owner across the boundary.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _ticket = inner.admit();
        assert_eq!(inner.live(), 1);
        panic!("a session task fell over");
    }));

    assert!(outcome.is_err(), "the panic was caught rather than missed");
    assert_eq!(manager.live(), 0, "the ticket dropped while unwinding");
}

/// Clones share one registry, which is what lets every accepted
/// connection admit against the same one.
#[test]
fn clones_share_the_registry_and_the_space() {
    let manager = SessionManager::for_user("box", None, StubThinkHarness);
    let other = manager.clone();
    let _ticket = other.admit();

    assert_eq!(manager.live(), 1, "a clone admits into the same registry");
    assert_eq!(manager.thinkspace(), other.thinkspace());
    assert_eq!(
        manager.thinkspace(),
        SessionManager::for_user("box", None, StubThinkHarness).thinkspace(),
        "the space follows the user rather than the registry"
    );
    assert_ne!(
        manager.thinkspace(),
        SessionManager::for_user("someone_else", None, StubThinkHarness).thinkspace()
    );
}

/// A log is addressed by SPACE and then by SESSION, both spelled from
/// noms rather than from anything a client sent.
#[test]
fn a_session_log_is_addressed_by_space_then_session() {
    let scratch = crate::tests::material::Scratch::make("manager_log");
    let manager = SessionManager::for_user("box", Some(scratch.root.clone()), StubThinkHarness);
    let ticket = manager.admit();
    let log = manager.log(&ticket).expect("a log opens under the root");

    assert_eq!(
        log.path(crate::log::LogFile::Turn),
        scratch
            .root
            .join(manager.thinkspace().as_str())
            .join(ticket.nom().as_str())
            .join(crate::log::LogFile::Turn.file_name())
    );

    log.append(crate::log::LogFile::Turn, "turn", "say hello")
        .expect("appends");
    assert!(
        log.read(crate::log::LogFile::Turn)
            .expect("reads back")
            .contains("say hello")
    );
}

/// The split is the whole point: a record on one grain never lands on
/// another, so a bulky emission cannot drown the turn outline beside it.
#[test]
fn each_grain_is_its_own_file() {
    let scratch = crate::tests::material::Scratch::make("manager_grains");
    let manager = SessionManager::for_user("box", Some(scratch.root.clone()), StubThinkHarness);
    let ticket = manager.admit();
    let log = manager.log(&ticket).expect("a log opens under the root");

    log.append(crate::log::LogFile::Transport, "open", "two")
        .expect("appends");
    log.append(crate::log::LogFile::Emission, "emission", "a long answer")
        .expect("appends");

    let transport = log.read(crate::log::LogFile::Transport).expect("reads");
    let emission = log.read(crate::log::LogFile::Emission).expect("reads");
    assert!(transport.contains("two"), "the transport record landed");
    assert!(
        !transport.contains("a long answer"),
        "the emission did not land beside it"
    );
    assert!(emission.contains("a long answer"), "the emission landed");
    assert!(
        !emission.contains("== open =="),
        "the transport record did not land beside it"
    );
}

/// The chunk sink writes VERBATIM and unframed, so a tail on it reads as
/// the answer forming rather than as a record format.
#[test]
fn the_chunk_sink_writes_the_stream_unframed() {
    let scratch = crate::tests::material::Scratch::make("manager_chunks");
    let manager = SessionManager::for_user("box", Some(scratch.root.clone()), StubThinkHarness);
    let ticket = manager.admit();
    let log = manager.log(&ticket).expect("a log opens under the root");

    let mut sink = log.chunks().expect("the sink opens");
    sink.write("Hello, ");
    sink.write("world");
    sink.end();

    assert_eq!(
        log.read(crate::log::LogFile::Chunks).expect("reads"),
        "Hello, world\n"
    );
}

/// No root means no log rather than a failure, which is what keeps a
/// missing cache home from refusing connections.
#[test]
fn without_a_root_a_session_simply_is_not_logged() {
    let manager = SessionManager::for_user("box", None, StubThinkHarness);
    let ticket = manager.admit();
    assert!(manager.log(&ticket).is_none());
}

#[test]
fn the_listing_names_exactly_the_live_sessions() {
    let manager = SessionManager::for_user("box", None, StubThinkHarness);
    let first = manager.admit();
    let second = manager.admit();

    let mut expected = vec![first.nom().clone(), second.nom().clone()];
    expected.sort();
    assert_eq!(manager.sessions(), expected);

    drop(first);
    assert_eq!(manager.sessions(), vec![second.nom().clone()]);
}
