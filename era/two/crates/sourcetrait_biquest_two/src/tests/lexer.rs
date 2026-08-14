//! Lexer locks: boundary splitting, folding, layer resolution, ids,
//! the banded run encoder, digit exemption, and the ingestion
//! refusals - over a hand-built table.
use crate::bucket::BucketTable;
use crate::lexer::CHARACTER_OFFSET;
use crate::lexer::KEYWORD_BEGIN_REPEAT;
use crate::lexer::KEYWORD_END_REPEAT;
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

#[test]
fn text_splits_into_word_runs_and_single_unicode_pieces() {
    let table = table();
    let pieces = boundary_pieces(&table, "Dog ate. it").expect("lexes");
    let shape: Vec<(&str, PieceKind)> =
        pieces.iter().map(|piece| (piece.text, piece.kind)).collect();
    assert_eq!(
        shape,
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
fn digits_ride_word_runs_and_never_encode_as_repetition() {
    let table = table();
    let buckets = BucketTable::new();
    let segmenter = Segmenter::new(&table, &buckets, &[]);
    // The zeros are a run of five identical digit tokens and MUST
    // stay plain: a number is place-value content, not repetition.
    let tokens = segmenter.segment("4000019").expect("segments");
    assert_eq!(tokens.len(), 7);
    assert!(tokens.iter().all(|t| t.layer == Layer::Character));
    assert_eq!(tokens[1].id, CHARACTER_OFFSET + char_index(&table, '0'));
    assert_eq!(tokens[5].id, CHARACTER_OFFSET + char_index(&table, '1'));
}

#[test]
fn keyboard_doubles_and_triples_take_their_rows() {
    let table = table();
    let buckets = BucketTable::new();
    let segmenter = Segmenter::new(&table, &buckets, &[]);
    let bucket_offset = CHARACTER_OFFSET + table.assigned_count() as u32;

    let tokens = segmenter.segment("  ").expect("segments");
    let space_double = buckets.index_for(' ', 2).expect("row") as u32;
    assert_eq!(tokens.len(), 1);
    assert_eq!((tokens[0].id, tokens[0].layer), (bucket_offset + space_double, Layer::Bucket));

    let tokens = segmenter.segment("...").expect("segments");
    let dot_triple = buckets.index_for('.', 3).expect("row") as u32;
    assert_eq!(tokens.len(), 1);
    assert_eq!((tokens[0].id, tokens[0].layer), (bucket_offset + dot_triple, Layer::Bucket));
}

#[test]
fn non_keyboard_runs_skip_the_rows() {
    let table = table();
    let buckets = BucketTable::new();
    let segmenter = Segmenter::new(&table, &buckets, &[]);
    // The control character has no keyboard row: two stays plain,
    // three rides REPEAT.
    let tokens = segmenter.segment("\u{0001}\u{0001}").expect("segments");
    assert_eq!(tokens.len(), 2);
    assert!(tokens.iter().all(|t| t.layer == Layer::Character));
    let tokens = segmenter.segment(&"\u{0001}".repeat(3)).expect("segments");
    let layers: Vec<Layer> = tokens.iter().map(|t| t.layer).collect();
    assert_eq!(layers, [Layer::Character, Layer::Keyword, Layer::Character]);
    assert_eq!(tokens[1].id, KEYWORD_REPEAT);
    assert_eq!(tokens[2].id, CHARACTER_OFFSET + char_index(&table, '3'));
}

#[test]
fn mid_runs_ride_repeat_with_one_count_digit() {
    let table = table();
    let buckets = BucketTable::new();
    let segmenter = Segmenter::new(&table, &buckets, &[]);
    // Four through nine ride |unit||REPEAT||digit| - rows serve
    // exactly two and three, never composing.
    let tokens = segmenter.segment(&" ".repeat(4)).expect("segments");
    let shape: Vec<(u32, Layer)> = tokens.iter().map(|t| (t.id, t.layer)).collect();
    assert_eq!(
        shape,
        [
            (CHARACTER_OFFSET + char_index(&table, ' '), Layer::Character),
            (KEYWORD_REPEAT, Layer::Keyword),
            (CHARACTER_OFFSET + char_index(&table, '4'), Layer::Character),
        ]
    );
    // A count digit then literal digits: exactly one digit is the
    // count, so the year survives as content.
    let text = format!("{}2024", " ".repeat(7));
    let tokens = segmenter.segment(&text).expect("segments");
    assert_eq!(tokens.len(), 7);
    assert_eq!(tokens[2].id, CHARACTER_OFFSET + char_index(&table, '7'));
    assert_eq!(tokens[3].id, CHARACTER_OFFSET + char_index(&table, '2'));
}

#[test]
fn long_runs_bracket_a_multi_digit_count() {
    let table = table();
    let buckets = BucketTable::new();
    let segmenter = Segmenter::new(&table, &buckets, &[]);
    let tokens = segmenter.segment(&" ".repeat(128)).expect("segments");
    let shape: Vec<(u32, Layer)> = tokens.iter().map(|t| (t.id, t.layer)).collect();
    assert_eq!(
        shape,
        [
            (CHARACTER_OFFSET + char_index(&table, ' '), Layer::Character),
            (KEYWORD_BEGIN_REPEAT, Layer::Keyword),
            (CHARACTER_OFFSET + char_index(&table, '1'), Layer::Character),
            (CHARACTER_OFFSET + char_index(&table, '2'), Layer::Character),
            (CHARACTER_OFFSET + char_index(&table, '8'), Layer::Character),
            (KEYWORD_END_REPEAT, Layer::Keyword),
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
    // it = admitted[1]; the double space = its keyboard row; dog =
    // admitted[0].
    assert_eq!(tokens[0].id, dictionary_offset + 1);
    assert_eq!(tokens[1].layer, Layer::Bucket);
    assert_eq!(tokens[2].id, dictionary_offset);
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
    let unassigned = boundary_pieces(&table, "dog z").expect_err("z is unassigned");
    assert!(unassigned.to_string().contains("ingestion refusal"), "got: {unassigned}");
    // An assigned control is just a Unicode-class character: the floor
    // is total over the assigned set.
    let pieces = boundary_pieces(&table, "dog\u{0001}").expect("control lexes");
    let last = pieces.last().expect("pieces");
    assert_eq!((last.text, last.kind), ("\u{0001}", PieceKind::Unicode));
}
