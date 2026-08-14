//! Store locks: the concept table, gloss sanitizing, dump absorption
//! (word set, definition merging, inflection links, fold and
//! multi-piece drops), and the single-line NUON rendering guard.
use crate::associations::AbstractConcept;
use crate::associations::DumpSide;
use crate::associations::Entry;
use crate::associations::condensed_line;
use crate::associations::sanitize_gloss;
use crate::lexer::KEYWORD_REPETITION;
use crate::ucd::CharRow;
use crate::ucd::CharacterTable;
use crate::value::v_str;

/// Ll, Lu (folding to Ll), Zs whitespace, Pd hyphen.
fn table() -> CharacterTable {
    let categories = vec![
        String::from("Ll"),
        String::from("Lu"),
        String::from("Zs"),
        String::from("Pd"),
    ];
    let row = |code_point: u32, category: u8, white_space: bool, fold: u32| CharRow {
        code_point,
        category,
        script: u16::MAX,
        white_space,
        fold,
    };
    let mut rows = vec![
        row(0x0020, 2, true, 0x0020),  // space
        row(0x002D, 3, false, 0x002D), // hyphen
        row(0x0044, 1, false, 0x0064), // D -> d
    ];
    for lower in [0x0061, 0x0062, 0x0064, 0x0065, 0x0067, 0x006B, 0x006F, 0x0073] {
        rows.push(row(lower, 0, false, lower)); // a b d e g k o s
    }
    CharacterTable::from_rows(rows, categories)
}

fn entry(json: &str) -> Entry {
    serde_json::from_str(json).expect("test entry parses")
}

#[test]
fn repetition_concept_is_backed_by_its_marker() {
    assert_eq!(AbstractConcept::ALL.len(), 1);
    let concept = AbstractConcept::Repetition;
    assert_eq!(concept.name(), "repetition");
    assert_eq!(concept.marker(), KEYWORD_REPETITION);
    assert_eq!(concept.spelling(), "<|repetition|>");
}

#[test]
fn glosses_sanitize_to_single_line_single_spaced() {
    assert_eq!(sanitize_gloss("  a\n\tb   c\r\n"), "a b c");
    assert_eq!(sanitize_gloss("\n \t"), "");
}

#[test]
fn absorb_collects_words_definitions_and_form_links() {
    let table = table();
    let mut side = DumpSide::new(&table);
    side.absorb(&entry(
        r#"{"word":"dog","lang_code":"en","pos":"noun",
            "forms":[{"form":"dogs"},{"form":"dog-eared"}],
            "senses":[{"glosses":["A domesticated canid."]}]}"#,
    ));
    assert!(side.words.contains("dog") && side.words.contains("dogs"));
    // The hyphenated form is not single-piece: no word, no link.
    assert!(!side.words.contains("dog-eared"));
    assert_eq!(
        side.definitions[&(String::from("dog"), String::from("noun"))],
        [String::from("A domesticated canid.")]
    );
    assert!(side.links.contains(&(String::from("dogs"), String::from("dog"))));
    assert_eq!(side.tally.links_from_forms, 1);
}

#[test]
fn form_of_links_entry_to_lemma_and_keeps_the_gloss() {
    let table = table();
    let mut side = DumpSide::new(&table);
    side.absorb(&entry(
        r#"{"word":"book","lang_code":"en","pos":"verb",
            "senses":[{"glosses":["simple past of bake"],
                       "form_of":[{"word":"bake"}]}]}"#,
    ));
    assert!(side.links.contains(&(String::from("book"), String::from("bake"))));
    assert_eq!(side.tally.links_from_form_of, 1);
    assert_eq!(
        side.definitions[&(String::from("book"), String::from("verb"))],
        [String::from("simple past of bake")]
    );
}

#[test]
fn headwords_fold_and_multiword_headwords_drop() {
    let table = table();
    let mut side = DumpSide::new(&table);
    side.absorb(&entry(
        r#"{"word":"Dog","lang_code":"en","pos":"noun",
            "senses":[{"glosses":["A dog."]}]}"#,
    ));
    side.absorb(&entry(
        r#"{"word":"dog bed","lang_code":"en","pos":"noun",
            "senses":[{"glosses":["A bed."]}]}"#,
    ));
    assert!(side.words.contains("dog"));
    assert!(!side.words.iter().any(|word| word.contains(' ')));
    // The multiword headword carries no definition row either.
    assert_eq!(side.definitions.len(), 1);
    assert!(side.definitions.contains_key(&(String::from("dog"), String::from("noun"))));
}

#[test]
fn single_code_point_entries_are_excluded_whole() {
    let table = table();
    let mut side = DumpSide::new(&table);
    // The character layer already holds "d"; no word, no definition,
    // and a single-letter form earns no row or link either.
    side.absorb(&entry(
        r#"{"word":"d","lang_code":"en","pos":"noun",
            "senses":[{"glosses":["The letter d."]}],
            "forms":[{"form":"o"}]}"#,
    ));
    assert!(side.words.is_empty());
    assert!(side.definitions.is_empty());
    assert!(side.links.is_empty());
}

#[test]
fn non_english_entries_are_skipped_whole() {
    let table = table();
    let mut side = DumpSide::new(&table);
    side.absorb(&entry(
        r#"{"word":"dose","lang_code":"de","pos":"noun",
            "senses":[{"glosses":["can"]}]}"#,
    ));
    assert_eq!(side.tally.non_english, 1);
    assert!(side.words.is_empty() && side.definitions.is_empty());
}

#[test]
fn senses_take_the_most_specific_gloss_and_merge_across_entries() {
    let table = table();
    let mut side = DumpSide::new(&table);
    // wiktextract glosses refine parent-to-child: the last is the
    // sense's own.
    side.absorb(&entry(
        r#"{"word":"dog","lang_code":"en","pos":"noun",
            "senses":[{"glosses":["Base.","Refined."]}]}"#,
    ));
    side.absorb(&entry(
        r#"{"word":"dog","lang_code":"en","pos":"noun",
            "senses":[{"glosses":["Second entry."]}]}"#,
    ));
    let senses = &side.definitions[&(String::from("dog"), String::from("noun"))];
    assert_eq!(senses.len(), 2);
    assert_eq!(senses[0], "Refined.");
    assert_eq!(senses[1], "Second entry.");
}

#[test]
fn condensed_rows_render_single_line_and_raw_newlines_refuse() {
    let engine_state = nu_protocol::engine::EngineState::new();
    let clean = condensed_line(&engine_state, &v_str("a \"quoted\" gloss")).expect("renders");
    assert!(!clean.contains('\n'));
    condensed_line(&engine_state, &v_str("two\nlines")).expect_err("multi-line rows refuse");
}
