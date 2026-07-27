//! Output locks: what collapses, and when colour appears.
use crate::style::{
    collapse_home,
    fail_style,
    paint,
};

#[test]
fn the_home_prefix_collapses_and_a_longer_sibling_does_not() {
    assert_eq!(collapse_home("/home/box/a/b", Some("/home/box")), "~/a/b");
    assert_eq!(collapse_home("/home/box", Some("/home/box")), "~");
    assert_eq!(
        collapse_home("/home/boxer/a", Some("/home/box")),
        "/home/boxer/a",
        "a longer name that merely starts the same is not the home"
    );
    assert_eq!(collapse_home("/etc/ssl", Some("/home/box")), "/etc/ssl");
    assert_eq!(collapse_home("/home/box/a", None), "/home/box/a");
}

/// A cause raised by the certificate library names its path
/// mid-sentence, which a leading-only rule would leave in full.
#[test]
fn a_path_inside_a_sentence_collapses_too() {
    assert_eq!(
        collapse_home("expected /home/box/a.toml. Install", Some("/home/box")),
        "expected ~/a.toml. Install"
    );
    assert_eq!(
        collapse_home("see /home/boxer/a here", Some("/home/box")),
        "see /home/boxer/a here",
        "the boundary rule holds mid-sentence as well"
    );
}

/// The stream test decides whether escapes appear at all, so both
/// branches are locked rather than only the coloured one.
#[test]
fn colour_is_emitted_only_when_it_is_asked_for() {
    let painted = paint(fail_style(), "x", true);
    assert!(painted.contains('\u{1b}'), "an escape is emitted: {painted:?}");
    assert!(painted.contains('x'));
    assert_eq!(paint(fail_style(), "x", false), "x");
}
