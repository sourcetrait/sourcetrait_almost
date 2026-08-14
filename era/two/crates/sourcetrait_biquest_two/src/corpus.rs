//! Corpus file walking shared by the measuring and training verbs.
use crate::*;

/// Recursively collect the files under a root, sorted; a file passes
/// through as itself.
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
        let mut pending = vec![root.clone()];
        while let Some(dir) = pending.pop() {
            for entry in fs::read_dir(&dir)? {
                let path = entry?.path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    snafu::ensure_whatever!(!files.is_empty(), "no corpus files under the given roots");
    Ok(files)
}
