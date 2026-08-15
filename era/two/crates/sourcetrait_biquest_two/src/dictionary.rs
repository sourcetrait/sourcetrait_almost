//! The dictionary layer's source: the raw enwiktionary dump, derived
//! through the WikimediaDumpTool's own renderer.
use crate::*;

use crate::ucd::CharacterTable;
use crate::wikiderive::derive_dump;

/// The vendored English word set (folded, single-piece, sorted),
/// derived from the raw enwiktionary dump; the CC BY-SA attribution
/// is at docs/licenses/wiktionary/. Re-vendor via `tokenizer
/// dictionary`.
const ENGLISH_WORDS: &str = include_str!("../data/words/english.txt");

/// The embedded English word list, file order (alphabetical).
pub(crate) fn embedded_words() -> Vec<String> {
    ENGLISH_WORDS
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

/// `biquest tokenizer dictionary`: the raw dump to the folded
/// word-set artifact - English-bearing ns0 titles plus the head
/// engines' derived forms, each an admissible connected candidate.
pub(crate) fn tokenizer_dictionary(args: &TokenizerDictionaryArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let table = CharacterTable::embedded()?;
    let derivation = derive_dump(&table, &args.source, true)?;

    let mut sorted: Vec<&String> = derivation.words.iter().collect();
    sorted.sort();
    let mut payload = String::with_capacity(derivation.words.len() * 10);
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

    let tally = &derivation.tally;
    let provenance = harness::nu::Value::record(
        harness::nu::record! {
            "source" => v_str(&args.source.display().to_string()),
            "pages_read" => v_int(tally.pages_read as i64),
            "ns0_pages" => v_int(tally.ns0_pages as i64),
            "redirect_pages" => v_int(tally.redirect_pages as i64),
            "english_pages" => v_int(tally.english_pages as i64),
            "no_entry_pages" => v_int(tally.no_entry_pages as i64),
            "parse_failures" => v_int(tally.parse_failures as i64),
            "titles_admitted" => v_int(tally.titles_admitted as i64),
            "forms_seen" => v_int(tally.forms_seen as i64),
            "forms_admitted" => v_int(tally.forms_admitted as i64),
            "words" => v_int(derivation.words.len() as i64),
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "extracted_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&args.out.with_extension("provenance.nuon"), &provenance)?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "pages_read" => v_int(tally.pages_read as i64),
            "english_pages" => v_int(tally.english_pages as i64),
            "words" => v_int(derivation.words.len() as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// Read a words artifact in FILE ORDER: with the whole dictionary as
/// the vocabulary (TheUser), line position IS the dictionary-layer
/// id order.
pub(crate) fn read_words_ordered(path: &Path) -> BiquestResult<Vec<String>> {
    let words: Vec<String> = fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    snafu::ensure_whatever!(!words.is_empty(), "{}: no words", path.display());
    Ok(words)
}
