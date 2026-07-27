//! The one message this binary shows a user.
use crate::preflight::PROFILE;
use crate::run::certificate_owed;

/// The message is the operator's only instruction and its command order
/// was wrong once, so the spelling is locked rather than the prose.
#[test]
fn the_certificate_message_puts_the_verb_before_the_profile() {
    let text = certificate_owed(std::path::Path::new("some/where/quest.toml"));

    assert!(text.contains("srcert generate quest <dir>"), "{text}");
    assert!(text.contains("srcert install quest <dir>"), "{text}");
    assert!(
        !text.contains("srcert quest"),
        "the profile-first order must not come back: {text}"
    );
}

/// A test's stdout is not a terminal, so the composed message must carry
/// no escapes at all - which is also what a redirect gets.
#[test]
fn a_non_terminal_stream_gets_no_escapes() {
    let text = certificate_owed(std::path::Path::new("some/where/quest.toml"));
    assert!(!text.contains('\u{1b}'), "{text:?}");
    assert!(text.contains(PROFILE));
}
