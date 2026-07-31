//! Per-session text logging for the daemon's own side, split by grain.
//!
//! The era library carries its own for the ThinkHarness side of the seam,
//! and this deliberately does not reach for it: an eon component must not
//! depend on an era's library to write a text file, or the one edge the
//! bridge exists to prevent gets opened for a convenience.
use crate::*;

/// Which of a session's logs a record belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogFile {
    /// The connection conversation: opens, closes, resets, faults.
    Transport,
    /// What each turn DID, one or two short lines apiece.
    Turn,
    /// The bulky pair - what the model was fed, and what it wrote.
    Emission,
    /// The live stream, as it arrives.
    Chunks,
}

impl LogFile {
    pub(crate) fn file_name(&self) -> &'static str {
        match self {
            Self::Transport => "transport.log",
            Self::Turn => "turn.log",
            Self::Emission => "emission.log",
            Self::Chunks => "chunks.log",
        }
    }
}

/// A session's logs: append-only, one directory per session.
///
/// The path segments are supplied rather than derived, because minting a
/// session's identity belongs to whoever owns the session.
#[derive(Debug, Clone)]
pub(crate) struct SessionLog {
    dir: PathBuf,
}

impl SessionLog {
    /// Open a session's log directory at `root` under the segments.
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
        Ok(Self { dir })
    }

    /// Where one of this session's logs is written.
    pub(crate) fn path(&self, file: LogFile) -> PathBuf {
        self.dir.join(file.file_name())
    }

    /// Append one labelled record to the named log.
    pub(crate) fn append(
        &self,
        file: LogFile,
        label: &str,
        body: &str,
    ) -> DquestResult<()> {
        let mut handle = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path(file))?;
        let record = format!("== {label} ==\n{}\n", body.trim_end());
        Ok(Write::write_all(&mut handle, record.as_bytes())?)
    }

    /// A sink for the live stream, holding its file open.
    pub(crate) fn chunks(&self) -> DquestResult<ChunkSink> {
        let handle = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path(LogFile::Chunks))?;
        Ok(ChunkSink { handle })
    }

    /// One log as text, for a caller reading its own session back.
    #[allow(dead_code)]
    pub(crate) fn read(&self, file: LogFile) -> DquestResult<String> {
        match std::fs::read_to_string(self.path(file)) {
            Ok(text) => Ok(text),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(String::new()),
            Err(error) => Err(error.into()),
        }
    }
}

/// The live stream, written as each piece arrives.
///
/// IT HOLDS ITS FILE OPEN, which the labelled records do not need to.
/// This one is written per TOKEN rather than per turn, so reopening it
/// fifty times a second to append a few bytes is the one place that cost
/// would be real.
///
/// The text goes down VERBATIM and unframed, so a `tail -f` reads as the
/// answer forming rather than as a record format. That is the whole
/// point of the file: a generation that runs for minutes is otherwise
/// indistinguishable from a hung one, because every other record here is
/// written only once the generation has already ended.
pub(crate) struct ChunkSink {
    handle: std::fs::File,
}

impl ChunkSink {
    /// Write one piece of the answer, dropping any failure.
    ///
    /// A write that fails takes the observability and not the turn, which
    /// is the same trade every other record here makes.
    pub(crate) fn write(&mut self, text: &str) {
        let _ = Write::write_all(&mut self.handle, text.as_bytes());
    }

    /// Mark the end of one generation's stream.
    pub(crate) fn end(&mut self) {
        let _ = Write::write_all(&mut self.handle, b"\n");
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
