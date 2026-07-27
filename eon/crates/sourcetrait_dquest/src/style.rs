//! Setup-time output, which is the only output this binary has.
//!
//! A daemon is silent in normal operation and speaks to a user only when
//! something is wrong. What can be wrong here is always a precondition it
//! cannot satisfy for itself, so everything in this module is SETUP
//! rather than operation - which is why it goes to stdout, and why there
//! is no informational half. Operational narration belongs to logging.
use crate::*;

/// The binary this output speaks for.
const NAME: &str = "dquest";

fn fail_style() -> anstyle::Style {
    anstyle::Style::new().fg_color(Some(anstyle::AnsiColor::Red.into()))
}

/// A resource named inside a sentence.
fn resource_style() -> anstyle::Style {
    anstyle::Style::new().fg_color(Some(anstyle::AnsiColor::Cyan.into()))
}

fn paint(style: anstyle::Style, text: &str, colour: bool) -> String {
    if !colour {
        return text.to_string();
    }
    format!("{}{text}{}", style.render(), style.render_reset())
}

fn stream_is_terminal() -> bool {
    io::stdout().is_terminal()
}

/// A path with the home prefix collapsed, for reading rather than for use.
pub(crate) fn pretty_path(path: &Path) -> String {
    collapse_home(&path.display().to_string(), env::var("HOME").ok().as_deref())
}

/// Collapse the home prefix WHEREVER it appears, not only at the start.
///
/// A cause built by the certificate library carries its path
/// mid-sentence, so a leading-only rule would leave the most-read line
/// uncollapsed. Split from `pretty_path` so it is testable without an
/// environment.
fn collapse_home(text: &str, home: Option<&str>) -> String {
    let Some(home) = home else {
        return text.to_string();
    };
    let home = home.trim_end_matches('/');
    if home.is_empty() {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(home) {
        let (before, tail) = rest.split_at(at);
        let after = &tail[home.len()..];
        out.push_str(before);
        // Only a path boundary counts, or `/home/boxer` collapses too.
        if after.is_empty() || after.starts_with('/') {
            out.push('~');
        } else {
            out.push_str(home);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

/// A resource, styled and home-collapsed, for building a message body.
pub(crate) fn resource(path: &Path) -> String {
    paint(resource_style(), &pretty_path(path), stream_is_terminal())
}

/// A resource that is a name rather than a path.
pub(crate) fn named(name: &str) -> String {
    paint(resource_style(), name, stream_is_terminal())
}

/// Report that the daemon cannot come up, prefixed on every line.
///
/// Split on newlines so a multi-line cause stays attributable rather than
/// only its first line carrying the binary's name.
pub(crate) fn fail(message: &str) {
    let colour = stream_is_terminal();
    let prefix = paint(fail_style(), &format!("[{NAME}]"), colour);
    let home = env::var("HOME").ok();
    let lines: Vec<&str> = if message.is_empty() {
        vec![""]
    } else {
        message.lines().collect()
    };
    let text: String = lines
        .iter()
        .map(|line| format!("{prefix} {}\n", collapse_home(line, home.as_deref())))
        .collect();
    let _ = io::stdout().write_all(text.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::{
        collapse_home,
        fail_style,
        paint,
    };

    /// The stream test decides whether escapes are emitted at all, so
    /// both branches are locked rather than only the coloured one.
    #[test]
    fn colour_is_emitted_only_when_it_is_asked_for() {
        let painted = paint(fail_style(), "x", true);
        assert!(painted.contains('\u{1b}'), "an escape is emitted: {painted:?}");
        assert!(painted.contains('x'));
        assert_eq!(paint(fail_style(), "x", false), "x");
    }

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
}
