//! Assembler locks: structural round trips, noun agreement, band
//! expansion, REPETITION's wire illegality, raw-string protection,
//! and indentation regeneration - over a hand table and a draft-shaped
//! syntax table.
use crate::assembler::Assembler;
use crate::assembler::SyntaxTable;
use crate::bucket::BucketTable;
use crate::lexer::KEYWORD_REPETITION;
use crate::lexer::Segmenter;
use crate::ucd::CharRow;
use crate::ucd::CharacterTable;

/// NULL plus the four heads plus a handful of nouns, draft order.
fn syntax() -> SyntaxTable {
    SyntaxTable::from_names(
        ["NULL", "OPEN", "CLOSE", "BEGIN", "END", "DIALOGUE", "CONFIG", "NUON", "NU"]
            .into_iter()
            .map(String::from)
            .collect(),
    )
    .expect("draft table binds")
}

/// Keyboard symbols, digits, newline, and d g o n u.
fn table() -> CharacterTable {
    let categories = vec![
        String::from("Ll"),
        String::from("Nd"),
        String::from("Zs"),
        String::from("Po"),
        String::from("Cc"),
    ];
    let row = |code_point: u32, category: u8, white_space: bool| CharRow {
        code_point,
        category,
        script: u16::MAX,
        white_space,
        fold: code_point,
    };
    let mut rows: Vec<CharRow> = vec![row(0x0A, 4, true)];
    for symbol in crate::bucket::KEYBOARD_CHARS {
        let white = symbol == ' ' || symbol == '\t';
        rows.push(row(symbol as u32, if white { 2 } else { 3 }, white));
    }
    for digit in '0'..='9' {
        rows.push(row(digit as u32, 1, false));
    }
    for letter in ['D', 'E', 'N', 'O', 'U', 'd', 'g', 'n', 'o', 'u'] {
        rows.push(row(letter as u32, 0, false));
    }
    rows.sort_by_key(|entry| entry.code_point);
    CharacterTable::from_rows(rows, categories)
}

struct Rig {
    table: CharacterTable,
    buckets: BucketTable,
    syntax: SyntaxTable,
    admitted: Vec<String>,
}

impl Rig {
    fn new() -> Self {
        Self {
            table: table(),
            buckets: BucketTable::new(),
            syntax: syntax(),
            admitted: vec![String::from("dog")],
        }
    }

    fn with<T>(&self, run: impl FnOnce(&Assembler<'_>) -> T) -> T {
        let segmenter = Segmenter::new(&self.table, &self.buckets, &self.admitted);
        let assembler = Assembler::new(
            &self.syntax,
            &segmenter,
            &self.table,
            &self.buckets,
            &self.admitted,
        );
        run(&assembler)
    }
}

#[test]
fn structure_round_trips_with_indentation_regenerated() {
    let rig = Rig::new();
    // Encoded flat: indentation is surface-only.
    let flat = "OPEN DIALOGUE\nOPEN CONFIG\nNU\nCLOSE CONFIG\nCLOSE DIALOGUE\n";
    let (wire, rendered) = rig.with(|assembler| {
        let wire = assembler.encode(flat).expect("encodes");
        let rendered = assembler.decode(&wire).expect("decodes");
        (wire, rendered)
    });
    // OPEN DIALOGUE OPEN CONFIG NU CLOSE CONFIG CLOSE DIALOGUE.
    assert_eq!(wire, [1, 5, 1, 6, 8, 2, 6, 2, 5]);
    assert_eq!(
        rendered,
        "OPEN DIALOGUE\n  OPEN CONFIG\n    NU\n  CLOSE CONFIG\nCLOSE DIALOGUE\n"
    );
    let rewire = rig.with(|assembler| assembler.encode(&rendered).expect("re-encodes"));
    assert_eq!(wire, rewire);
}

#[test]
fn serialization_interiors_tokenize_and_round_trip() {
    let rig = Rig::new();
    let text = "OPEN CONFIG\n  BEGIN NUON\n    dog\n  END NUON\nCLOSE CONFIG\n";
    let (wire, rendered) = rig.with(|assembler| {
        let wire = assembler.encode(text).expect("encodes");
        let rendered = assembler.decode(&wire).expect("decodes");
        (wire, rendered)
    });
    // The interior is one dictionary token between BEGIN/END pairs.
    let dictionary_offset =
        256 + rig.table.assigned_count() as u32 + rig.buckets.count() as u32;
    assert_eq!(wire, [1, 6, 3, 7, dictionary_offset, 4, 7, 2, 6]);
    assert_eq!(rendered, text);
}

#[test]
fn close_and_end_noun_agreement_fault() {
    let rig = Rig::new();
    rig.with(|assembler| {
        let mismatch = assembler.encode("OPEN DIALOGUE\nCLOSE CONFIG\n");
        assert!(mismatch.is_err());
        let stray = assembler.encode("CLOSE CONFIG\n");
        assert!(stray.is_err());
        let unended = assembler.encode("OPEN CONFIG\nBEGIN NUON\n");
        assert!(unended.is_err());
        // Decode-side: CLOSE against the wrong open block.
        let fault = assembler.decode(&[1, 5, 2, 6]).expect_err("mismatch faults");
        assert!(fault.to_string().contains("wire fault at token"), "got: {fault}");
    });
}

#[test]
fn bands_expand_and_repetition_is_wire_illegal() {
    let rig = Rig::new();
    let text = format!("OPEN CONFIG\n  BEGIN NUON\n    {}\n  END NUON\nCLOSE CONFIG\n", "u".repeat(12));
    let (wire, rendered) = rig.with(|assembler| {
        let wire = assembler.encode(&text).expect("encodes");
        let rendered = assembler.decode(&wire).expect("decodes");
        (wire, rendered)
    });
    // The 12-run brackets a two-digit count; the decode expands it.
    assert_eq!(rendered, text);
    assert!(wire.contains(&crate::lexer::KEYWORD_BEGIN_REPEAT));
    rig.with(|assembler| {
        let mut poisoned = wire.clone();
        let slot = poisoned
            .iter()
            .position(|&id| id == crate::lexer::KEYWORD_BEGIN_REPEAT)
            .expect("bracketed run");
        poisoned[slot] = KEYWORD_REPETITION;
        let fault = assembler.decode(&poisoned).expect_err("repetition faults");
        assert!(
            fault.to_string().contains("AbstractConceptMarker"),
            "got: {fault}"
        );
    });
}

#[test]
fn raw_strings_protect_structural_content_both_ways() {
    let rig = Rig::new();
    // Content whose lines would read as assembly: the decoder must
    // wrap it raw, and the wrapped form must re-encode identically.
    let text = "OPEN CONFIG\n  BEGIN NUON\n    #{\n    END NUON\n    }#\n  END NUON\nCLOSE CONFIG\n";
    let (wire, rendered) = rig.with(|assembler| {
        let wire = assembler.encode(text).expect("encodes");
        let rendered = assembler.decode(&wire).expect("decodes");
        (wire, rendered)
    });
    assert_eq!(rendered, text);
    let rewire = rig.with(|assembler| assembler.encode(&rendered).expect("re-encodes"));
    assert_eq!(wire, rewire);
}

#[test]
fn digits_never_expand_as_runs() {
    let rig = Rig::new();
    let text = "OPEN CONFIG\n  BEGIN NUON\n    4000019\n  END NUON\nCLOSE CONFIG\n";
    let (wire, rendered) = rig.with(|assembler| {
        let wire = assembler.encode(text).expect("encodes");
        let rendered = assembler.decode(&wire).expect("decodes");
        (wire, rendered)
    });
    assert_eq!(rendered, text);
    assert!(!wire.contains(&crate::lexer::KEYWORD_REPEAT));
    // A digit heading a run is a decode fault, not an expansion.
    rig.with(|assembler| {
        let digit_four = 256
            + rig.table.index_of('4' as u32).expect("digit row");
        let poisoned = [3u32, 7, digit_four, crate::lexer::KEYWORD_REPEAT];
        let fault = assembler.decode(&poisoned).expect_err("digit run faults");
        assert!(fault.to_string().contains("non-repeatable"), "got: {fault}");
    });
}

#[test]
fn content_outside_serialization_faults() {
    let rig = Rig::new();
    rig.with(|assembler| {
        let fault = assembler.decode(&[1, 5, 300]).expect_err("content faults");
        assert!(
            fault.to_string().contains("outside a serialization"),
            "got: {fault}"
        );
    });
}

/// The escape-capable draft: TRAIN/INPUT/ESCAPE/ESCAPED bound after
/// the base set. Ids: TRAIN 9, INPUT 10, ESCAPE 11, ESCAPED 12.
fn escape_rig() -> Rig {
    Rig {
        table: table(),
        buckets: BucketTable::new(),
        syntax: SyntaxTable::from_names(
            [
                "NULL", "OPEN", "CLOSE", "BEGIN", "END", "DIALOGUE", "CONFIG",
                "NUON", "NU", "TRAIN", "INPUT", "ESCAPE", "ESCAPED",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        )
        .expect("escape table binds"),
        admitted: vec![String::from("dog")],
    }
}

#[test]
fn escape_spans_round_trip_and_mention_ids() {
    let rig = escape_rig();
    let text = "OPEN TRAIN\n  OPEN INPUT\n    BEGIN NUON\n      dog <|ESCAPE|><|OPEN|> dog <|ESCAPED|> dog\n    END NUON\n  CLOSE INPUT\nCLOSE TRAIN\n";
    let (wire, rendered) = rig.with(|assembler| {
        let wire = assembler.encode(text).expect("encodes");
        let rendered = assembler.decode(&wire).expect("decodes");
        (wire, rendered)
    });
    assert_eq!(rendered, text);
    let escape_at = wire.iter().position(|&id| id == 11).expect("escape id");
    assert_eq!(wire[escape_at + 1], 1, "the OPEN mention rides as its real id");
    assert!(wire.contains(&12), "the span closes with the ESCAPED id");
    let rewire = rig.with(|assembler| assembler.encode(&rendered).expect("re-encodes"));
    assert_eq!(wire, rewire);
}

#[test]
fn bare_marker_spellings_are_invalid_outside_spans() {
    let rig = escape_rig();
    let text = "OPEN TRAIN\n  OPEN INPUT\n    BEGIN NUON\n      dog <|OPEN|> dog\n    END NUON\n  CLOSE INPUT\nCLOSE TRAIN\n";
    rig.with(|assembler| {
        let fault = assembler.encode(text).expect_err("bare spelling faults");
        assert!(fault.to_string().contains("outside an escape span"), "got: {fault}");
    });
}

#[test]
fn escape_spans_are_train_input_only() {
    let rig = escape_rig();
    let text = "OPEN DIALOGUE\n  OPEN CONFIG\n    BEGIN NUON\n      <|ESCAPE|><|OPEN|><|ESCAPED|>\n    END NUON\n  CLOSE CONFIG\nCLOSE DIALOGUE\n";
    rig.with(|assembler| {
        let fault = assembler.encode(text).expect_err("outside TRAIN faults");
        assert!(fault.to_string().contains("legal only inside TRAIN"), "got: {fault}");
        // Decode-side: an escape id in a non-TRAIN interior faults.
        let fault = assembler
            .decode(&[1, 6, 3, 7, 11, 1, 12, 4, 7, 2, 6])
            .expect_err("decode-side legality");
        assert!(fault.to_string().contains("legal only inside TRAIN"), "got: {fault}");
    });
}

#[test]
fn nested_unterminated_and_unknown_spellings_fault() {
    let rig = escape_rig();
    let prefix = "OPEN TRAIN\n  OPEN INPUT\n    BEGIN NUON\n      ";
    let suffix = "\n    END NUON\n  CLOSE INPUT\nCLOSE TRAIN\n";
    rig.with(|assembler| {
        let nested = format!("{prefix}<|ESCAPE|><|ESCAPE|>{suffix}");
        let fault = assembler.encode(&nested).expect_err("nesting faults");
        assert!(fault.to_string().contains("cannot nest"), "got: {fault}");
        let unterminated = format!("{prefix}<|ESCAPE|><|OPEN|> dog{suffix}");
        let fault = assembler.encode(&unterminated).expect_err("unterminated faults");
        assert!(fault.to_string().contains("never closes"), "got: {fault}");
        let unknown = format!("{prefix}<|ESCAPE|><|BOGUS|><|ESCAPED|>{suffix}");
        let fault = assembler.encode(&unknown).expect_err("unknown spelling faults");
        assert!(fault.to_string().contains("not a bound keyword"), "got: {fault}");
    });
}

#[test]
fn reign_lines_validate_and_produce_nothing() {
    let rig = escape_rig();
    let text = "REIGN human\n\nOPEN DIALOGUE\nCLOSE DIALOGUE\n";
    let wire = rig.with(|assembler| assembler.encode(text).expect("encodes"));
    assert_eq!(wire, [1, 5, 2, 5], "REIGN contributes no wire");
    rig.with(|assembler| {
        assert!(assembler.encode("REIGN bogus\n").is_err());
        assert!(assembler.encode("REIGN\n").is_err());
        assert!(assembler.encode("REIGN human extra\n").is_err());
    });
}
