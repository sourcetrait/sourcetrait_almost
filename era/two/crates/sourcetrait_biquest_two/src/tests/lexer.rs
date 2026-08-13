//! Lexer locks: boundary splitting, folding, layer resolution, ids,
//! buckets, REPEAT collapsing, and the ingestion refusals - over a
//! hand-built table.
use crate::bucket::BucketTable;
use crate::lexer::CHARACTER_OFFSET;
use crate::lexer::KEYWORD_REPEAT;
use crate::lexer::Layer;
use crate::lexer::PieceKind;
use crate::lexer::Segmenter;
use crate::lexer::boundary_pieces;
use crate::ucd::CharRow;
use crate::ucd::CharacterTable;

/// Ll, Lu (folding to Ll), Zs whitespace, Po punctuation, Cc
/// controls, Nd digits.
fn table() -> CharacterTable {
    let categories = vec![
        String::from("Ll"),
        String::from("Lu"),
        String::from("Zs"),
        String::from("Po"),
        String::from("Cc"),
        String::from("Nd"),
    ];
    let row = |code_point: u32, category: u8, white_space: bool, fold: u32| CharRow {
        code_point,
        category,
        script: u16::MAX,
        white_space,
        fold,
    };
    let mut rows = vec![
        row(0x0001, 4, false, 0x0001),
        row(0x0009, 4, true, 0x0009), // tab
        row(0x0020, 2, true, 0x0020), // space
        row(0x0023, 3, false, 0x0023), // #
        row(0x002E, 3, false, 0x002E), // .
        row(0x002F, 3, false, 0x002F), // /
    ];
    for digit in 0x0030..=0x0039 {
        rows.push(row(digit, 5, false, digit)); // 0-9
    }
    rows.push(row(0x0041, 1, false, 0x0061)); // A -> a
    rows.push(row(0x0044, 1, false, 0x0064)); // D -> d
    rows.push(row(0x0061, 0, false, 0x0061)); // a
    rows.push(row(0x0064, 0, false, 0x0064)); // d
    rows.push(row(0x0065, 0, false, 0x0065)); // e
    rows.push(row(0x0067, 0, false, 0x0067)); // g
    rows.push(row(0x0069, 0, false, 0x0069)); // i
    rows.push(row(0x006F, 0, false, 0x006F)); // o
    rows.push(row(0x0074, 0, false, 0x0074)); // t
    CharacterTable::from_rows(rows, categories)
}

/// The synthetic table's dense index for a code point.
fn char_index(table: &CharacterTable, c: char) -> u32 {
    table.index_of(c as u32).expect("in the synthetic table")
}

fn kinds<'a>(pieces: &'a [crate::lexer::Piece<'a>]) -> Vec<(&'a str, PieceKind)> {
    pieces.iter().map(|piece| (piece.text, piece.kind)).collect()
}

#[test]
fn text_splits_into_word_runs_and_single_unicode_pieces() {
    let table = table();
    let buckets = BucketTable::new();
    let pieces = boundary_pieces(&table, &buckets, "Dog ate. it").expect("lexes");
    assert_eq!(
        kinds(&pieces),
        [
            ("Dog", PieceKind::Word),
            (" ", PieceKind::Unicode),
            ("ate", PieceKind::Word),
            (".", PieceKind::Unicode),
            (" ", PieceKind::Unicode),
            ("it", PieceKind::Word),
        ]
    );
}

#[test]
fn bucket_sequences_match_longest_first() {
    let table = table();
    let buckets = BucketTable::new();
    // Two-space is bucket index 0, four-space index 1; heading depth
    // 2 is index 3; the comment entries close the enumeration.
    let pieces = boundary_pieces(&table, &buckets, "      dog   /**// ##").expect("lexes");
    assert_eq!(
        kinds(&pieces),
        [
            ("    ", PieceKind::Bucket(1)),
            ("  ", PieceKind::Bucket(0)),
            ("dog", PieceKind::Word),
            ("  ", PieceKind::Bucket(0)),
            (" ", PieceKind::Unicode),
            ("/**", PieceKind::Bucket(18)),
            ("//", PieceKind::Bucket(16)),
            (" ", PieceKind::Unicode),
            ("##", PieceKind::Bucket(3)),
        ]
    );
}

#[test]
fn dictionary_hit_is_case_insensitive_and_miss_splits_to_characters() {
    let table = table();
    let buckets = BucketTable::new();
    let admitted = vec![String::from("dog"), String::from("it")];
    let segmenter = Segmenter::new(&table, &buckets, &admitted);
    let tokens = segmenter.segment("Dog ate it").expect("segments");
    let layers: Vec<Layer> = tokens.iter().map(|token| token.layer).collect();
    assert_eq!(
        layers,
        [
            Layer::Dictionary, // Dog (folded to dog)
            Layer::Character,  // space
            Layer::Character,  // a
            Layer::Character,  // t
            Layer::Character,  // e
            Layer::Character,  // space
            Layer::Dictionary, // it
        ]
    );
}

#[test]
fn ids_stack_keywords_characters_buckets_then_dictionary() {
    let table = table();
    let buckets = BucketTable::new();
    let admitted = vec![String::from("dog"), String::from("it")];
    let segmenter = Segmenter::new(&table, &buckets, &admitted);
    let bucket_offset = CHARACTER_OFFSET + table.assigned_count() as u32;
    let dictionary_offset = bucket_offset + buckets.count() as u32;
    let tokens = segmenter.segment("it  dog").expect("segments");
    // it = admitted[1]; the double space = bucket index 0; dog =
    // admitted[0].
    assert_eq!(tokens[0].id, dictionary_offset + 1);
    assert_eq!(tokens[1].id, bucket_offset);
    assert_eq!(tokens[1].layer, Layer::Bucket);
    assert_eq!(tokens[2].id, dictionary_offset);
}

#[test]
fn runs_collapse_into_repeat_groups() {
    let table = table();
    let buckets = BucketTable::new();
    let segmenter = Segmenter::new(&table, &buckets, &[]);
    let bucket_offset = CHARACTER_OFFSET + table.assigned_count() as u32;

    // Twelve spaces: three four-space units, the ruled example -
    // |    ||REPEAT||3|.
    let tokens = segmenter.segment(&" ".repeat(12)).expect("segments");
    let shape: Vec<(u32, Layer)> = tokens.iter().map(|t| (t.id, t.layer)).collect();
    assert_eq!(
        shape,
        [
            (bucket_offset + 1, Layer::Bucket),
            (KEYWORD_REPEAT, Layer::Keyword),
            (CHARACTER_OFFSET + char_index(&table, '3'), Layer::Character),
        ]
    );

    // Eight spaces are two units: the plain form is cheaper, no
    // REPEAT.
    let tokens = segmenter.segment(&" ".repeat(8)).expect("segments");
    assert_eq!(tokens.len(), 2);
    assert!(tokens.iter().all(|t| t.layer == Layer::Bucket));

    // A four-tab run collapses over the tab character row.
    let tokens = segmenter.segment(&"\t".repeat(4)).expect("segments");
    let shape: Vec<(u32, Layer)> = tokens.iter().map(|t| (t.id, t.layer)).collect();
    assert_eq!(
        shape,
        [
            (CHARACTER_OFFSET + char_index(&table, '\t'), Layer::Character),
            (KEYWORD_REPEAT, Layer::Keyword),
            (CHARACTER_OFFSET + char_index(&table, '4'), Layer::Character),
        ]
    );
}

#[test]
fn repeat_count_is_one_digit_so_literal_digits_stay_literal() {
    let table = table();
    let buckets = BucketTable::new();
    let segmenter = Segmenter::new(&table, &buckets, &[]);
    // Sixteen spaces then a literal year: the count digit is exactly
    // one token, so 2024 survives as its own word-run character
    // split.
    let text = format!("{}2024", " ".repeat(16));
    let tokens = segmenter.segment(&text).expect("segments");
    let layers: Vec<Layer> = tokens.iter().map(|t| t.layer).collect();
    assert_eq!(
        layers,
        [
            Layer::Bucket,    // the four-space unit
            Layer::Keyword,   // REPEAT
            Layer::Character, // count digit 4
            Layer::Character, // 2
            Layer::Character, // 0
            Layer::Character, // 2
            Layer::Character, // 4
        ]
    );
    assert_eq!(tokens[2].id, CHARACTER_OFFSET + char_index(&table, '4'));
    assert_eq!(tokens[3].id, CHARACTER_OFFSET + char_index(&table, '2'));
}

#[test]
fn long_runs_chain_groups_of_nine() {
    let table = table();
    let buckets = BucketTable::new();
    let segmenter = Segmenter::new(&table, &buckets, &[]);
    // Eleven tabs: a group of nine, then a leftover pair stays plain.
    let tokens = segmenter.segment(&"\t".repeat(11)).expect("segments");
    let layers: Vec<Layer> = tokens.iter().map(|t| t.layer).collect();
    assert_eq!(
        layers,
        [
            Layer::Character, // tab unit
            Layer::Keyword,   // REPEAT
            Layer::Character, // count digit 9
            Layer::Character, // leftover tab
            Layer::Character, // leftover tab
        ]
    );
    assert_eq!(tokens[2].id, CHARACTER_OFFSET + char_index(&table, '9'));
}

#[test]
fn segment_pieces_carries_folded_words_and_characters() {
    let table = table();
    let buckets = BucketTable::new();
    let admitted = vec![String::from("dog")];
    let segmenter = Segmenter::new(&table, &buckets, &admitted);
    let pairs = segmenter.segment_pieces("Dog. a").expect("segments");
    let texts: Vec<(&str, Layer)> = pairs
        .iter()
        .map(|(token, text)| (text.as_str(), token.layer))
        .collect();
    assert_eq!(
        texts,
        [
            ("dog", Layer::Dictionary), // folded row identity, not "Dog"
            (".", Layer::Character),
            (" ", Layer::Character),
            ("a", Layer::Character),
        ]
    );
}

#[test]
fn unassigned_refuses_and_assigned_controls_lex_as_unicode() {
    let table = table();
    let buckets = BucketTable::new();
    let unassigned =
        boundary_pieces(&table, &buckets, "dog z").expect_err("z is unassigned");
    assert!(unassigned.to_string().contains("ingestion refusal"), "got: {unassigned}");
    // An assigned control is just a Unicode-class character: the floor
    // is total over the assigned set.
    let pieces = boundary_pieces(&table, &buckets, "dog\u{0001}").expect("control lexes");
    let last = pieces.last().expect("pieces");
    assert_eq!((last.text, last.kind), ("\u{0001}", PieceKind::Unicode));
}
