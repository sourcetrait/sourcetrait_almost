//! Per-session text logging for the daemon's own side.
//!
//! The era library carries its own for the Questness side of the seam,
//! and this deliberately does not reach for it: an eon component must not
//! depend on an era's library to write a text file, or the one edge the
//! bridge exists to prevent gets opened for a convenience.
use crate::*;

/// The file a session's log is written to, inside its own directory.
pub(crate) const LOG_FILE: &str = "session.log";

/// A session's text log: append-only, one directory per session.
///
/// The path segments are supplied rather than derived, because minting a
/// session's identity belongs to whoever owns the session.
#[derive(Debug, Clone)]
pub(crate) struct SessionLog {
    path: PathBuf,
}

impl SessionLog {
    /// Open a log at `root` under the given segments, creating both.
    pub(crate) fn open(root: &Path, segments: &[&str]) -> DquestResult<Self> {
        for segment in segments {
            if !is_plain_segment(segment) {
                return Err(DquestError::Segment {
                    segment: (*segment).to_string(),
                });
            }
        }
        let mut dir = root.to_path_buf();
        for segment in segments {
            dir.push(segment);
        }
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            path: dir.join(LOG_FILE),
        })
    }

    /// The file this log appends to. Read by the locks alone.
    #[allow(dead_code)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Append one labelled record.
    pub(crate) fn append(&self, label: &str, body: &str) -> DquestResult<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let record = format!("== {label} ==\n{}\n", body.trim_end());
        Ok(Write::write_all(&mut file, record.as_bytes())?)
    }

    /// The whole log as text, for a caller reading its own session back.
    #[allow(dead_code)]
    pub(crate) fn read(&self) -> DquestResult<String> {
        match std::fs::read_to_string(&self.path) {
            Ok(text) => Ok(text),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(String::new()),
            Err(error) => Err(error.into()),
        }
    }
}

/// Whether a segment is a plain name rather than a path of its own.
///
/// A nom arrives from a client, so it decides a directory only if it
/// cannot climb out of the root it is joined to. Refused rather than
/// cleaned, because a cleaned nom would silently address a different
/// session than the one asked for.
fn is_plain_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment != "."
        && segment != ".."
        && !segment.contains('/')
        && !segment.contains('\\')
        && !segment.contains('\0')
}
