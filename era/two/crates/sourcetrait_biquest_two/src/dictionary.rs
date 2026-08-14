//! The dictionary layer's source: a wiktextract English dump, streamed.
use crate::*;

use std::io::BufRead;

use crate::lexer::PieceKind;
use crate::lexer::boundary_pieces;
use crate::ucd::CharacterTable;

/// The vendored English word set (folded, single-piece, sorted),
/// extracted from the wiktextract dump; the CC BY-SA attribution is
/// at docs/licenses/wiktionary/. Re-vendor via `tokenizer dictionary`.
const ENGLISH_WORDS: &str = include_str!("../data/words/english.txt");

/// The embedded English word list, file order (alphabetical).
pub(crate) fn embedded_words() -> Vec<String> {
    ENGLISH_WORDS
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

/// The fields read off one wiktextract entry; the rest is skipped.
#[derive(serde::Deserialize)]
struct Entry {
    #[serde(default)]
    word: Option<String>,
    #[serde(default)]
    lang_code: Option<String>,
    #[serde(default)]
    forms: Vec<Form>,
}

#[derive(serde::Deserialize)]
struct Form {
    #[serde(default)]
    form: Option<String>,
}

/// What one extraction pass counted.
#[derive(Default)]
struct DictionaryTally {
    entries_read: usize,
    non_english: usize,
    headwords_seen: usize,
    forms_seen: usize,
    not_single_piece: usize,
    refused_characters: usize,
}

/// Whether a candidate lexes as exactly one word piece, case-folded.
///
/// Multiword, hyphenated, and apostrophe-carrying entries fail here by
/// design: a token can never span a boundary, so such entries decompose
/// at lex time and get no dictionary row.
fn single_piece_folded(table: &CharacterTable, candidate: &str) -> Option<String> {
    let pieces = boundary_pieces(table, candidate).ok()?;
    if pieces.len() != 1 || pieces[0].kind != PieceKind::Word {
        return None;
    }
    table.fold_str(pieces[0].text).ok()
}

/// `biquest tokenizer dictionary`: dump to the folded word-set artifact.
pub(crate) fn tokenizer_dictionary(args: &TokenizerDictionaryArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let table = CharacterTable::embedded()?;
    let file = fs::File::open(&args.dump)?;
    let reader = io::BufReader::with_capacity(1 << 20, file);

    let mut words: HashSet<String> = HashSet::new();
    let mut tally = DictionaryTally::default();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: Entry = match serde_json::from_str(&line) {
            Ok(entry) => entry,
            Err(e) => snafu::whatever!("{}:{}: JSON parse failed: {e}", args.dump.display(), index + 1),
        };
        tally.entries_read += 1;
        if entry.lang_code.as_deref() != Some("en") {
            tally.non_english += 1;
            continue;
        }
        let mut candidates: Vec<&str> = Vec::new();
        if let Some(word) = &entry.word {
            tally.headwords_seen += 1;
            candidates.push(word);
        }
        for form in &entry.forms {
            if let Some(form) = &form.form {
                tally.forms_seen += 1;
                candidates.push(form);
            }
        }
        for candidate in candidates {
            match single_piece_folded(&table, candidate) {
                Some(folded) => {
                    words.insert(folded);
                }
                None => {
                    if candidate.chars().any(|c| table.index_of(c as u32).is_none()) {
                        tally.refused_characters += 1;
                    } else {
                        tally.not_single_piece += 1;
                    }
                }
            }
        }
    }
    snafu::ensure_whatever!(
        !words.is_empty(),
        "{}: no admissible words found",
        args.dump.display()
    );

    let mut sorted: Vec<&String> = words.iter().collect();
    sorted.sort();
    let mut payload = String::with_capacity(words.len() * 10);
    for word in &sorted {
        payload.push_str(word);
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
            "dump" => v_str(&args.dump.display().to_string()),
            "entries_read" => v_int(tally.entries_read as i64),
            "non_english" => v_int(tally.non_english as i64),
            "headwords_seen" => v_int(tally.headwords_seen as i64),
            "forms_seen" => v_int(tally.forms_seen as i64),
            "not_single_piece" => v_int(tally.not_single_piece as i64),
            "refused_characters" => v_int(tally.refused_characters as i64),
            "words" => v_int(words.len() as i64),
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "extracted_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&args.out.with_extension("provenance.nuon"), &provenance)?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "entries_read" => v_int(tally.entries_read as i64),
            "words" => v_int(words.len() as i64),
            "not_single_piece" => v_int(tally.not_single_piece as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// Read a words artifact back: one folded word per line.
pub(crate) fn read_words(path: &Path) -> BiquestResult<HashSet<String>> {
    let words: HashSet<String> = fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    snafu::ensure_whatever!(!words.is_empty(), "{}: no words", path.display());
    Ok(words)
}
