//! Lexer locks: boundary splitting, folding, layer resolution, ids,
//! and the ingestion refusals - over a hand-built character table.
use crate::lexer::CHARACTER_OFFSET;
use crate::lexer::Layer;
use crate::lexer::PieceKind;
use crate::lexer::Segmenter;
use crate::lexer::boundary_pieces;
use crate::ucd::CharRow;
use crate::ucd::CharacterTable;

/// Ll, Lu (folding to Ll), Zs whitespace, Po punctuation, Cc other.
fn table() -> CharacterTable {
    let categories = vec![
        String::from("Ll"),
        String::from("Lu"),
        String::from("Zs"),
        String::from("Po"),
        String::from("Cc"),
    ];
    let row = |code_point: u32, category: u8, white_space: bool, fold: u32| CharRow {
        code_point,
        category,
        script: u16::MAX,
        white_space,
        fold,
    };
    let rows = vec![
        row(0x0001, 4, false, 0x0001),
        row(0x0020, 2, true, 0x0020),
        row(0x002E, 3, false, 0x002E),
        row(0x0041, 1, false, 0x0061), // A -> a
        row(0x0044, 1, false, 0x0064), // D -> d
        row(0x0061, 0, false, 0x0061), // a
        row(0x0064, 0, false, 0x0064), // d
        row(0x0065, 0, false, 0x0065), // e
        row(0x0067, 0, false, 0x0067), // g
        row(0x0069, 0, false, 0x0069), // i
        row(0x006F, 0, false, 0x006F), // o
        row(0x0074, 0, false, 0x0074), // t
    ];
    CharacterTable::from_rows(rows, categories)
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
fn dictionary_hit_is_case_insensitive_and_miss_splits_to_characters() {
    let table = table();
    let admitted = vec![String::from("dog"), String::from("it")];
    let segmenter = Segmenter::new(&table, &admitted);
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
fn ids_stack_keywords_characters_then_dictionary() {
    let table = table();
    let admitted = vec![String::from("dog"), String::from("it")];
    let segmenter = Segmenter::new(&table, &admitted);
    let offset = CHARACTER_OFFSET + table.assigned_count() as u32;
    let tokens = segmenter.segment("it dog").expect("segments");
    // it = admitted[1], the space = character index 1 (0x0020), dog =
    // admitted[0].
    assert_eq!(tokens[0].id, offset + 1);
    assert_eq!(tokens[1].id, CHARACTER_OFFSET + 1);
    assert_eq!(tokens[2].id, offset);
}

#[test]
fn segment_pieces_carries_folded_words_and_characters() {
    let table = table();
    let admitted = vec![String::from("dog")];
    let segmenter = Segmenter::new(&table, &admitted);
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
