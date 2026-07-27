//! Setup-time output, which is the only output this binary has.
use crate::*;

/// The binary this output speaks for.
const NAME: &str = "dquest";

pub(crate) fn fail_style() -> anstyle::Style {
    anstyle::Style::new().fg_color(Some(anstyle::AnsiColor::Red.into()))
}

/// A resource named inside a sentence.
fn resource_style() -> anstyle::Style {
    anstyle::Style::new().fg_color(Some(anstyle::AnsiColor::Cyan.into()))
}

/// Wrap in colour, or not, as the caller has already decided.
pub(crate) fn paint(style: anstyle::Style, text: &str, colour: bool) -> String {
    if !colour {
        return text.to_string();
    }
    format!("{}{text}{}", style.render(), style.render_reset())
}

fn stream_is_terminal() -> bool {
    io::stdout().is_terminal()
}

/// A path with the home prefix collapsed, for reading rather than use.
pub(crate) fn pretty_path(path: &Path) -> String {
    collapse_home(&path.display().to_string(), env::var("HOME").ok().as_deref())
}

/// Collapse the home prefix wherever it appears, boundary-aware.
pub(crate) fn collapse_home(text: &str, home: Option<&str>) -> String {
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

/// A resource, styled and home-collapsed, for a message body.
pub(crate) fn resource(path: &Path) -> String {
    paint(resource_style(), &pretty_path(path), stream_is_terminal())
}

/// A resource that is a name rather than a path.
pub(crate) fn named(name: &str) -> String {
    paint(resource_style(), name, stream_is_terminal())
}

/// Report that the daemon cannot come up, prefixed per line.
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
