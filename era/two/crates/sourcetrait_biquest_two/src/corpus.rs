//! Corpus file walking shared by the measuring and training verbs.
use crate::*;

/// Collect corpus files IN ROOT ORDER: a file root holds its argument
/// position and a directory root's own walk sorts within it, so the
/// caller's root sequence is the pack sequence (a curriculum, never a
/// set).
pub(crate) fn corpus_files(roots: &[PathBuf]) -> BiquestResult<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = Vec::new();
    for root in roots {
        if root.is_file() {
            files.push(root.clone());
            continue;
        }
        snafu::ensure_whatever!(
            root.is_dir(),
            "corpus root {} is neither file nor directory",
            root.display()
        );
        let mut walked: Vec<PathBuf> = Vec::new();
        let mut pending = vec![root.clone()];
        while let Some(dir) = pending.pop() {
            for entry in fs::read_dir(&dir)? {
                let path = entry?.path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    walked.push(path);
                }
            }
        }
        walked.sort();
        files.extend(walked);
    }
    snafu::ensure_whatever!(!files.is_empty(), "no corpus files under the given roots");
    Ok(files)
}
