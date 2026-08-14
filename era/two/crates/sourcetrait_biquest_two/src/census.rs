//! The corpus type census, and the frequency-through-membership gate.
use crate::*;

use crate::dictionary::read_words;
use crate::lexer::CandidateItem;
use crate::lexer::CorePart;
use crate::lexer::candidate_items;
use crate::ucd::CharacterTable;

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

/// What one census pass counted.
#[derive(Default)]
pub(crate) struct CensusTally {
    pub(crate) files: usize,
    pub(crate) text_bytes: usize,
    pub(crate) word_pieces: usize,
    pub(crate) unicode_chars: usize,
    /// Files the lexer refused, skipped whole and reported.
    pub(crate) refused: Vec<(PathBuf, String)>,
}

/// Count word types across the corpus files, counting what would
/// TOKENIZE (TheUser's design): a candidate the full dictionary
/// holds counts whole; an edged candidate whose core it holds counts
/// the core; else each word part counts. A file the lexer refuses is
/// skipped whole and recorded; measurement never ingests a file the
/// trainer would refuse.
pub(crate) fn census_counts(
    table: &CharacterTable,
    words: &HashSet<String>,
    files: &[PathBuf],
) -> BiquestResult<(HashMap<String, u64>, CensusTally)> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    let mut tally = CensusTally::default();
    for path in files {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => snafu::whatever!("read {} failed (corpus is utf-8): {e}", path.display()),
        };
        let items = match candidate_items(table, &text) {
            Ok(items) => items,
            Err(e) => {
                tally.refused.push((path.clone(), e.to_string()));
                continue;
            }
        };
        tally.files += 1;
        tally.text_bytes += text.len();
        for item in items {
            let candidate = match item {
                CandidateItem::Plain(piece_text) => {
                    tally.unicode_chars += piece_text.chars().count();
                    continue;
                }
                CandidateItem::Candidate(candidate) => candidate,
            };
            let edges = usize::from(candidate.leading.is_some())
                + usize::from(candidate.trailing.is_some());
            if words.contains(&candidate.full) {
                tally.word_pieces += 1;
                *counts.entry(candidate.full).or_insert(0) += 1;
            } else if edges > 0 && words.contains(&candidate.core) {
                tally.word_pieces += 1;
                tally.unicode_chars += edges;
                *counts.entry(candidate.core).or_insert(0) += 1;
            } else {
                tally.unicode_chars += edges;
                for part in candidate.parts {
                    match part {
                        CorePart::Word { folded, .. } => {
                            tally.word_pieces += 1;
                            *counts.entry(folded).or_insert(0) += 1;
                        }
                        CorePart::Connector { .. } => tally.unicode_chars += 1,
                    }
                }
            }
        }
    }
    Ok((counts, tally))
}

/// Refusal rows for a summary or provenance record.
pub(crate) fn refused_rows(refused: &[(PathBuf, String)]) -> harness::nu::Value {
    harness::nu::Value::list(
        refused
            .iter()
            .map(|(path, reason)| {
                harness::nu::Value::record(
                    harness::nu::record! {
                        "file" => v_str(&path.display().to_string()),
                        "reason" => v_str(reason),
                    },
                    span(),
                )
            })
            .collect(),
        span(),
    )
}

/// Count-descending then alphabetical: the deterministic census order.
fn ordered(counts: &HashMap<String, u64>) -> Vec<(&String, u64)> {
    let mut rows: Vec<(&String, u64)> = counts.iter().map(|(word, &count)| (word, count)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    rows
}

/// `biquest tokenizer census`: corpus roots to the type-count artifact.
pub(crate) fn tokenizer_census(args: &TokenizerCensusArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let table = CharacterTable::embedded()?;
    let words = read_words(&args.words)?;
    let files = corpus_files(&args.roots)?;
    let (counts, tally) = census_counts(&table, &words, &files)?;

    let rows = ordered(&counts);
    let mut payload = String::with_capacity(rows.len() * 16);
    for (word, count) in &rows {
        payload.push_str(word);
        payload.push('\t');
        payload.push_str(&count.to_string());
        payload.push('\n');
    }
    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, payload)?;

    let provenance = harness::nu::Value::record(
        harness::nu::record! {
            "roots" => harness::nu::Value::list(
                args.roots.iter().map(|p| v_str(&p.display().to_string())).collect(),
                span(),
            ),
            "words" => v_str(&args.words.display().to_string()),
            "files" => v_int(tally.files as i64),
            "text_bytes" => v_int(tally.text_bytes as i64),
            "word_pieces" => v_int(tally.word_pieces as i64),
            "distinct_types" => v_int(rows.len() as i64),
            "unicode_chars" => v_int(tally.unicode_chars as i64),
            "refused_files" => refused_rows(&tally.refused),
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "counted_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&args.out.with_extension("provenance.nuon"), &provenance)?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "files" => v_int(tally.files as i64),
            "refused_files" => v_int(tally.refused.len() as i64),
            "word_pieces" => v_int(tally.word_pieces as i64),
            "distinct_types" => v_int(rows.len() as i64),
            "unicode_chars" => v_int(tally.unicode_chars as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}

