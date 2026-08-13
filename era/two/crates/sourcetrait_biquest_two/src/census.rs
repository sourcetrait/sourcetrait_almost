//! The corpus type census, and the frequency-through-membership gate.
use crate::*;

use crate::bucket::BucketTable;
use crate::dictionary::read_words;
use crate::lexer::PieceKind;
use crate::lexer::boundary_pieces;
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
    pub(crate) bucket_tokens: usize,
    /// Files the lexer refused, skipped whole and reported.
    pub(crate) refused: Vec<(PathBuf, String)>,
}

/// Count folded word types across the corpus files. A file the lexer
/// refuses is skipped whole and recorded; measurement never ingests a
/// file the trainer would refuse.
pub(crate) fn census_counts(
    table: &CharacterTable,
    buckets: &BucketTable,
    files: &[PathBuf],
) -> BiquestResult<(HashMap<String, u64>, CensusTally)> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    let mut tally = CensusTally::default();
    for path in files {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => snafu::whatever!("read {} failed (corpus is utf-8): {e}", path.display()),
        };
        let pieces = match boundary_pieces(table, buckets, &text) {
            Ok(pieces) => pieces,
            Err(e) => {
                tally.refused.push((path.clone(), e.to_string()));
                continue;
            }
        };
        tally.files += 1;
        tally.text_bytes += text.len();
        for piece in pieces {
            match piece.kind {
                PieceKind::Word => {
                    tally.word_pieces += 1;
                    let folded = match table.fold_str(piece.text) {
                        Ok(folded) => folded,
                        Err(c) => snafu::whatever!(
                            "{}: ingestion refusal: unassigned U+{:04X}",
                            path.display(),
                            c as u32
                        ),
                    };
                    *counts.entry(folded).or_insert(0) += 1;
                }
                PieceKind::Unicode => tally.unicode_chars += 1,
                PieceKind::Bucket(_) => tally.bucket_tokens += 1,
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
    let buckets = BucketTable::new();
    let files = corpus_files(&args.roots)?;
    let (counts, tally) = census_counts(&table, &buckets, &files)?;

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
            "files" => v_int(tally.files as i64),
            "text_bytes" => v_int(tally.text_bytes as i64),
            "word_pieces" => v_int(tally.word_pieces as i64),
            "distinct_types" => v_int(rows.len() as i64),
            "unicode_chars" => v_int(tally.unicode_chars as i64),
            "bucket_tokens" => v_int(tally.bucket_tokens as i64),
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
            "bucket_tokens" => v_int(tally.bucket_tokens as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// Read a census-counts artifact back, order preserved.
pub(crate) fn read_counts(path: &Path) -> BiquestResult<Vec<(String, u64)>> {
    let mut rows = Vec::new();
    for (index, line) in fs::read_to_string(path)?.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let Some((word, count)) = line.split_once('\t') else {
            snafu::whatever!("{}:{}: no tab", path.display(), index + 1);
        };
        let Ok(count) = count.parse::<u64>() else {
            snafu::whatever!("{}:{}: bad count", path.display(), index + 1);
        };
        rows.push((word.to_string(), count));
    }
    snafu::ensure_whatever!(!rows.is_empty(), "{}: no rows", path.display());
    Ok(rows)
}

/// `biquest tokenizer admit`: census counts through dictionary
/// membership into the admitted wordlist, the dictionary layer's order.
pub(crate) fn tokenizer_admit(args: &TokenizerAdmitArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let counts = read_counts(&args.counts)?;
    let words = read_words(&args.words)?;

    let census_occurrences: u64 = counts.iter().map(|(_, count)| count).sum();
    let mut admitted_rows: Vec<harness::nu::Value> = Vec::new();
    let mut admitted_occurrences = 0u64;
    for (word, count) in &counts {
        if *count < args.min_count || !words.contains(word) {
            continue;
        }
        admitted_occurrences += count;
        admitted_rows.push(harness::nu::Value::record(
            harness::nu::record! {
                "word" => v_str(word),
                "count" => v_int(*count as i64),
            },
            span(),
        ));
    }
    snafu::ensure_whatever!(!admitted_rows.is_empty(), "nothing admitted");
    let admitted_count = admitted_rows.len();
    harness::nu::save_value(&args.out, &harness::nu::Value::list(admitted_rows, span()))?;

    let provenance = harness::nu::Value::record(
        harness::nu::record! {
            "counts" => v_str(&args.counts.display().to_string()),
            "words" => v_str(&args.words.display().to_string()),
            "min_count" => v_int(args.min_count as i64),
            "census_types" => v_int(counts.len() as i64),
            "dictionary_words" => v_int(words.len() as i64),
            "admitted_types" => v_int(admitted_count as i64),
            "census_occurrences" => v_int(census_occurrences as i64),
            "admitted_occurrences" => v_int(admitted_occurrences as i64),
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "admitted_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&args.out.with_extension("provenance.nuon"), &provenance)?;

    let coverage = if census_occurrences == 0 {
        0.0
    } else {
        admitted_occurrences as f64 / census_occurrences as f64
    };
    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "census_types" => v_int(counts.len() as i64),
            "dictionary_words" => v_int(words.len() as i64),
            "admitted_types" => v_int(admitted_count as i64),
            "occurrence_coverage" => v_float(coverage),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// Read an admitted artifact back into its ordered wordlist.
pub(crate) fn read_admitted(path: &Path) -> BiquestResult<Vec<String>> {
    let value = harness::nu::load_value(path)?;
    let rows = match value.as_list() {
        Ok(rows) => rows,
        Err(e) => snafu::whatever!("{}: not a table: {e}", path.display()),
    };
    let mut words = Vec::with_capacity(rows.len());
    for row in rows {
        let record = match row.as_record() {
            Ok(record) => record,
            Err(e) => snafu::whatever!("admitted row: {e}"),
        };
        words.push(field_str(record, "word")?);
    }
    snafu::ensure_whatever!(!words.is_empty(), "{}: no rows", path.display());
    Ok(words)
}
