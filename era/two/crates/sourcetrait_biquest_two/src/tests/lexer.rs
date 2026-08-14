//! Lexer locks: boundary splitting, folding, layer resolution, ids,
//! REPEAT collapsing, and the ingestion refusals - over a hand-built
//! table.
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
    let dictionary_offset = CHARACTER_OFFSET + table.assigned_count() as u32;
    let tokens = segmenter.segment("it  dog").expect("segments");
    // it = admitted[1]; a two-space run stays plain characters; dog =
    // admitted[0].
    assert_eq!(tokens[0].id, dictionary_offset + 1);
    assert_eq!(tokens[1].id, CHARACTER_OFFSET + char_index(&table, ' '));
    assert_eq!(tokens[2].id, CHARACTER_OFFSET + char_index(&table, ' '));
    assert_eq!(tokens[3].id, dictionary_offset);
}

#[test]
fn runs_collapse_into_repeat_groups() {
    let table = table();
    let segmenter = Segmenter::new(&table, &[]);

    // A four-space indent is one group: |sp||REPEAT||4|.
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

    // Two spaces stay plain: the group would cost three.
    let tokens = segmenter.segment(&" ".repeat(2)).expect("segments");
    assert_eq!(tokens.len(), 2);
    assert!(tokens.iter().all(|t| t.layer == Layer::Character));

    // Twelve spaces chain greedily: nine, then three.
    let tokens = segmenter.segment(&" ".repeat(12)).expect("segments");
    let layers: Vec<Layer> = tokens.iter().map(|t| t.layer).collect();
    assert_eq!(
        layers,
        [
            Layer::Character, // space unit
            Layer::Keyword,   // REPEAT
            Layer::Character, // 9
            Layer::Character, // space unit
            Layer::Keyword,   // REPEAT
            Layer::Character, // 3
        ]
    );
    assert_eq!(tokens[2].id, CHARACTER_OFFSET + char_index(&table, '9'));
    assert_eq!(tokens[5].id, CHARACTER_OFFSET + char_index(&table, '3'));

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
    let segmenter = Segmenter::new(&table, &[]);
    // Eight spaces then a literal year: the count digit is exactly
    // one token, so 2024 survives as its own word-run character
    // split.
    let text = format!("{}2024", " ".repeat(8));
    let tokens = segmenter.segment(&text).expect("segments");
    let layers: Vec<Layer> = tokens.iter().map(|t| t.layer).collect();
    assert_eq!(
        layers,
        [
            Layer::Character, // the space unit
            Layer::Keyword,   // REPEAT
            Layer::Character, // count digit 8
            Layer::Character, // 2
            Layer::Character, // 0
            Layer::Character, // 2
            Layer::Character, // 4
        ]
    );
    assert_eq!(tokens[2].id, CHARACTER_OFFSET + char_index(&table, '8'));
    assert_eq!(tokens[3].id, CHARACTER_OFFSET + char_index(&table, '2'));
}

#[test]
fn long_runs_chain_groups_of_nine() {
    let table = table();
    let segmenter = Segmenter::new(&table, &[]);
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
