//! The compression ledger against cl100k, and the encode debug dump.
use crate::*;

use crate::census::corpus_files;
use crate::census::read_admitted;
use crate::census::refused_rows;
use crate::lexer::Layer;
use crate::lexer::Segmenter;
use crate::ucd::CharacterTable;

/// `biquest tokenizer ledger`: Quill tokens against the checkpoint's
/// cl100k-family tokenizer, per file and in total.
pub(crate) fn tokenizer_ledger(cli: &Cli, args: &TokenizerLedgerArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let config = llm::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let tokenizer = llm::load_tokenizer(&config.model_dir())?;
    llm::verify_token_map(&tokenizer)?;
    let table = CharacterTable::embedded()?;
    let admitted = read_admitted(&args.admitted)?;
    let segmenter = Segmenter::new(&table, &admitted);
    let files = corpus_files(&args.roots)?;

    let mut rows: Vec<harness::nu::Value> = Vec::new();
    let mut refused: Vec<(PathBuf, String)> = Vec::new();
    let mut quill_total = 0usize;
    let mut dictionary_total = 0usize;
    let mut character_total = 0usize;
    let mut cl100k_total = 0usize;
    let mut byte_total = 0usize;
    let mut measured_files = 0usize;
    for path in &files {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => snafu::whatever!("read {} failed (corpus is utf-8): {e}", path.display()),
        };
        // A refused file drops out of BOTH sides, keeping the
        // comparison honest over the same file set.
        let tokens = match segmenter.segment(&text) {
            Ok(tokens) => tokens,
            Err(e) => {
                refused.push((path.clone(), e.to_string()));
                continue;
            }
        };
        measured_files += 1;
        let dictionary_tokens =
            tokens.iter().filter(|token| token.layer == Layer::Dictionary).count();
        let character_tokens = tokens.len() - dictionary_tokens;
        let encoding = match tokenizer.encode(text.as_str(), false) {
            Ok(encoding) => encoding,
            Err(e) => snafu::whatever!("{}: cl100k encode failed: {e}", path.display()),
        };
        let cl100k_tokens = encoding.get_ids().len();
        quill_total += tokens.len();
        dictionary_total += dictionary_tokens;
        character_total += character_tokens;
        cl100k_total += cl100k_tokens;
        byte_total += text.len();
        rows.push(harness::nu::Value::record(
            harness::nu::record! {
                "file" => v_str(&path.display().to_string()),
                "text_bytes" => v_int(text.len() as i64),
                "quill_tokens" => v_int(tokens.len() as i64),
                "dictionary_tokens" => v_int(dictionary_tokens as i64),
                "character_tokens" => v_int(character_tokens as i64),
                "cl100k_tokens" => v_int(cl100k_tokens as i64),
            },
            span(),
        ));
    }

    let ratio = if cl100k_total == 0 {
        0.0
    } else {
        quill_total as f64 / cl100k_total as f64
    };
    let report = harness::nu::Value::record(
        harness::nu::record! {
            "roots" => harness::nu::Value::list(
                args.roots.iter().map(|p| v_str(&p.display().to_string())).collect(),
                span(),
            ),
            "admitted" => v_str(&args.admitted.display().to_string()),
            "files" => v_int(measured_files as i64),
            "refused_files" => refused_rows(&refused),
            "text_bytes" => v_int(byte_total as i64),
            "quill_tokens" => v_int(quill_total as i64),
            "dictionary_tokens" => v_int(dictionary_total as i64),
            "character_tokens" => v_int(character_total as i64),
            "cl100k_tokens" => v_int(cl100k_total as i64),
            "quill_over_cl100k" => v_float(ratio),
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "measured_at" => v_int(epoch_seconds()),
            "rows" => harness::nu::Value::list(rows, span()),
        },
        span(),
    );
    harness::nu::save_value(&args.out, &report)?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "files" => v_int(measured_files as i64),
            "refused_files" => v_int(refused.len() as i64),
            "quill_tokens" => v_int(quill_total as i64),
            "cl100k_tokens" => v_int(cl100k_total as i64),
            "quill_over_cl100k" => v_float(ratio),
            "dictionary_share" => v_float(if quill_total == 0 {
                0.0
            } else {
                dictionary_total as f64 / quill_total as f64
            }),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// `biquest tokenizer ucd`: report the embedded layer's shape.
pub(crate) fn tokenizer_ucd() -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let table = CharacterTable::embedded()?;
    let mut word_chars = 0usize;
    let mut unicode_chars = 0usize;
    let mut white_space_chars = 0usize;
    let mut folded = 0usize;
    for row in &table.rows {
        match table.class_of_row(row) {
            crate::ucd::CharClass::Word => word_chars += 1,
            _ => unicode_chars += 1,
        }
        if row.white_space {
            white_space_chars += 1;
        }
        if row.fold != row.code_point {
            folded += 1;
        }
    }
    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "assigned" => v_int(table.assigned_count() as i64),
            "word_chars" => v_int(word_chars as i64),
            "unicode_chars" => v_int(unicode_chars as i64),
            "white_space_chars" => v_int(white_space_chars as i64),
            "fold_pairs" => v_int(folded as i64),
            "decompositions" => v_int(table.decompositions.len() as i64),
            "numeric_values" => v_int(table.numeric_values.len() as i64),
            "uppercase_maps" => v_int(table.simple_uppercase.len() as i64),
            "lowercase_maps" => v_int(table.simple_lowercase.len() as i64),
            "scripts" => v_int(table.scripts.len() as i64),
            "general_categories" => v_int(table.general_categories.len() as i64),
            "dictionary_offset" => v_int(
                crate::lexer::CHARACTER_OFFSET as i64 + table.assigned_count() as i64
            ),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}
