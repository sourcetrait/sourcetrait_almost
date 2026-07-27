//! Per-session text logging, for both sides of the seam.
use crate::*;

/// The file a session's log is written to, inside its own directory.
pub const LOG_FILE: &str = "session.log";

/// A session's text log: append-only, one directory per session.
///
/// The path segments are supplied rather than derived, because minting a
/// session's identity belongs to whoever owns the session.
#[derive(Debug, Clone)]
pub struct SessionLog {
    path: PathBuf,
}

impl SessionLog {
    /// Open a log at `root` under the given segments, creating both.
    pub fn open(root: &Path, segments: &[&str]) -> QuestCoreResult<Self> {
        for segment in segments {
            snafu::ensure_whatever!(
                is_plain_segment(segment),
                "a session path segment is a plain name; got {segment:?}"
            );
        }
        let mut dir = root.to_path_buf();
        for segment in segments {
            dir.push(segment);
        }
        fs::create_dir_all(&dir)?;
        Ok(Self {
            path: dir.join(LOG_FILE),
        })
    }

    /// The file this log appends to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one labelled record.
    pub fn append(&self, label: &str, body: &str) -> QuestCoreResult<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let record = format!("== {label} ==\n{}\n", body.trim_end());
        Ok(io::Write::write_all(&mut file, record.as_bytes())?)
    }

    /// The whole log as text, for a caller reading its own session back.
    pub fn read(&self) -> QuestCoreResult<String> {
        match fs::read_to_string(&self.path) {
            Ok(text) => Ok(text),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(String::new()),
            Err(error) => Err(error.into()),
        }
    }
}

/// Whether a segment is a plain name rather than a path of its own.
///
/// A nom arrives from a client, so it decides a directory only if it
/// cannot climb out of the root it is joined to.
fn is_plain_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment != "."
        && segment != ".."
        && !segment.contains('/')
        && !segment.contains('\\')
        && !segment.contains('\0')
}
