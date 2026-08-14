//! Matrix locks: the layout arithmetic, the composition rule, jitter
//! identity, keyboard-row associations, and build determinism - over
//! a hand-built table.
use crate::bucket::BucketTable;
use crate::lexer::KEYWORD_REPETITION;
use crate::matrix::MatrixLayout;
use crate::matrix::MatrixSpec;
use crate::matrix::build_matrix;
use crate::matrix::word_seed;
use crate::ucd::CharRow;
use crate::ucd::CharacterTable;

const HIDDEN: usize = 16;
const SEED: u64 = 7;
const SIGMA: f64 = 0.02;

fn spec(reserve: usize, jitter_sigma: f64) -> MatrixSpec {
    MatrixSpec {
        hidden: HIDDEN,
        reserve,
        seed: SEED,
        sigma: SIGMA,
        jitter_sigma,
    }
}

fn reseeded_spec(reserve: usize, jitter_sigma: f64) -> MatrixSpec {
    MatrixSpec {
        seed: SEED + 1,
        ..spec(reserve, jitter_sigma)
    }
}

/// Every keyboard symbol (the build composes all 68 rows), the two
/// count digits, and d g o.
fn table() -> CharacterTable {
    let categories = vec![
        String::from("Ll"),
        String::from("Nd"),
        String::from("Zs"),
        String::from("Po"),
    ];
    let row = |code_point: u32, category: u8, white_space: bool| CharRow {
        code_point,
        category,
        script: u16::MAX,
        white_space,
        fold: code_point,
    };
    let mut rows: Vec<CharRow> = Vec::new();
    for symbol in crate::bucket::KEYBOARD_CHARS {
        let white = symbol == ' ' || symbol == '\t';
        rows.push(row(symbol as u32, if white { 2 } else { 3 }, white));
    }
    for digit in ['2', '3'] {
        rows.push(row(digit as u32, 1, false));
    }
    for lower in ['d', 'g', 'o'] {
        rows.push(row(lower as u32, 0, false));
    }
    rows.sort_by_key(|entry| entry.code_point);
    CharacterTable::from_rows(rows, categories)
}

fn char_row_index(table: &CharacterTable, layout: &MatrixLayout, c: char) -> usize {
    layout.character_offset + table.index_of(c as u32).expect("in the table") as usize
}

fn row(build: &crate::matrix::MatrixBuild, index: usize) -> &[f32] {
    &build.rows[index * HIDDEN..(index + 1) * HIDDEN]
}

#[test]
fn layout_offsets_stack_the_layers() {
    let table = table();
    let buckets = BucketTable::new();
    let layout = MatrixLayout::new(&table, &buckets, 2, 8, HIDDEN);
    assert_eq!(layout.character_offset, 256);
    assert_eq!(layout.keyboard_offset, 256 + table.assigned_count());
    assert_eq!(layout.dictionary_offset, layout.keyboard_offset + buckets.count());
    assert_eq!(layout.reserve_offset, layout.dictionary_offset + 2);
    assert_eq!(layout.vocab_size, layout.reserve_offset + 8);
}

#[test]
fn word_rows_are_scaled_sums_of_their_characters() {
    let table = table();
    let buckets = BucketTable::new();
    let admitted = vec![String::from("dog")];
    // jitter 0: the composition alone.
    let build =
        build_matrix(&table, &buckets, &admitted, spec(0, 0.0)).expect("builds");
    let layout = build.layout;
    let d = row(&build, char_row_index(&table, &layout, 'd'));
    let o = row(&build, char_row_index(&table, &layout, 'o'));
    let g = row(&build, char_row_index(&table, &layout, 'g'));
    let word = row(&build, layout.dictionary_offset);
    let scale = 1.0 / (3f64).sqrt();
    for i in 0..HIDDEN {
        let expected = ((d[i] as f64 + o[i] as f64 + g[i] as f64) * scale) as f32;
        assert!((word[i] - expected).abs() < 1e-6, "slot {i}");
    }
}

#[test]
fn jitter_separates_anagrams_and_is_word_stable() {
    let table = table();
    let buckets = BucketTable::new();
    let admitted = vec![String::from("dog"), String::from("god")];
    let build =
        build_matrix(&table, &buckets, &admitted, spec(0, 0.004)).expect("builds");
    let layout = build.layout;
    let dog = row(&build, layout.dictionary_offset);
    let god = row(&build, layout.dictionary_offset + 1);
    assert_ne!(dog, god);
    // The jitter keys on the word, not its admitted position.
    let reordered = vec![String::from("god"), String::from("dog")];
    let rebuilt =
        build_matrix(&table, &buckets, &reordered, spec(0, 0.004)).expect("builds");
    assert_eq!(dog, row(&rebuilt, rebuilt.layout.dictionary_offset + 1));
    assert_ne!(word_seed("dog"), word_seed("god"));
}

#[test]
fn keyboard_rows_compose_their_declared_associations() {
    let table = table();
    let buckets = BucketTable::new();
    let build =
        build_matrix(&table, &buckets, &[], spec(0, 0.0)).expect("builds");
    let layout = build.layout;
    let index = buckets.index_for(' ', 3).expect("space triple");
    let space = row(&build, char_row_index(&table, &layout, ' '));
    let three = row(&build, char_row_index(&table, &layout, '3'));
    let marker = row(&build, KEYWORD_REPETITION as usize);
    let composed = row(&build, layout.keyboard_offset + index);
    let scale = 1.0 / (3f64).sqrt();
    for i in 0..HIDDEN {
        let expected = ((space[i] as f64 + three[i] as f64 + marker[i] as f64) * scale) as f32;
        assert!((composed[i] - expected).abs() < 1e-6, "slot {i}");
    }
}

#[test]
fn builds_are_seed_deterministic() {
    let table = table();
    let buckets = BucketTable::new();
    let admitted = vec![String::from("dog")];
    let first =
        build_matrix(&table, &buckets, &admitted, spec(4, 0.004)).expect("builds");
    let second =
        build_matrix(&table, &buckets, &admitted, spec(4, 0.004)).expect("builds");
    assert_eq!(first.rows, second.rows);
    let reseeded =
        build_matrix(&table, &buckets, &admitted, reseeded_spec(4, 0.004))
            .expect("builds");
    assert_ne!(first.rows, reseeded.rows);
}

#[test]
fn every_row_is_seeded_nonzero() {
    let table = table();
    let buckets = BucketTable::new();
    let admitted = vec![String::from("dog")];
    let build =
        build_matrix(&table, &buckets, &admitted, spec(4, 0.004)).expect("builds");
    for index in 0..build.layout.vocab_size {
        assert!(
            row(&build, index).iter().any(|v| *v != 0.0),
            "row {index} is all zeros"
        );
    }
}
