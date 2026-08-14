use crate::wikivocab::indefinite_article;
use crate::wikivocab::language_table;
use crate::wikivocab::place_table;
use crate::wikivocab::serial_join;
use crate::wikivocab::tag_table;
use crate::wikivocab::term_modifiers;

#[test]
fn language_codes_resolve_across_the_three_kinds() {
    let table = language_table();
    assert_eq!(table.name("en"), Some("English"));
    assert_eq!(table.name("enm"), Some("Middle English"));
    assert_eq!(table.name("la-med"), Some("Medieval Latin"));
    assert_eq!(table.name("gem"), Some("Germanic"));
    assert_eq!(table.name("gem-pro"), Some("Proto-Germanic"));
    assert_eq!(table.name("nope-nope"), None);
}

#[test]
fn inflection_tags_resolve_shortcuts_multiparts_and_lists() {
    let table = tag_table();
    assert_eq!(table.resolve("3"), vec!["third-person"]);
    assert_eq!(table.resolve("s"), vec!["singular"]);
    assert_eq!(table.resolve("1s"), vec!["first-person", "singular"]);
    assert_eq!(table.resolve("nom//acc"), vec!["nominative/accusative"]);
    assert_eq!(
        table.resolve("ed-form").join(" "),
        "simple past and past participle"
    );
    assert_eq!(table.resolve("made-up-tag"), vec!["made-up-tag"]);
}

#[test]
fn place_data_carries_the_render_keys() {
    let table = place_table();
    assert_eq!(table.expand_alias("c"), "country");
    assert_eq!(table.expand_alias("cont"), "continent");
    assert_eq!(table.placetype_resolved("county").preposition, "of");
    assert_eq!(table.placetype_resolved("city").preposition, "");
    assert!(table.placetype_resolved("oblast").affix_type == "Suf");
    assert!(table.holonym_takes_the("country", "United States"));
    assert!(table.holonym_takes_the("country", "Netherlands"));
    assert!(!table.holonym_takes_the("country", "France"));
    assert!(table.holonym_takes_the("river", "Amazon"));
    let usa = table.location("USA").expect("USA row");
    assert_eq!(usa.alias_of, "United States");
    assert!(usa.display_expand);
    assert_eq!(table.qualifier("largest").map(|(_, article)| article), Some("the"));
    assert_eq!(table.qualifier("several").map(|(_, article)| article), Some("none"));
}

#[test]
fn article_join_and_modifier_helpers_behave() {
    assert_eq!(indefinite_article("English surname"), "an");
    assert_eq!(indefinite_article("male given name"), "a");
    assert_eq!(indefinite_article("unisex name"), "a");
    assert_eq!(indefinite_article("occupation"), "an");
    assert_eq!(
        serial_join(
            &[String::from("a"), String::from("b"), String::from("c")],
            "and"
        ),
        "a, b and c"
    );
    let (base, modifiers) = term_modifiers("bjǫrn<t:bear><q:rare>");
    assert_eq!(base, "bjǫrn");
    assert_eq!(
        modifiers,
        vec![
            (String::from("t"), String::from("bear")),
            (String::from("q"), String::from("rare")),
        ]
    );
    let (base, modifiers) = term_modifiers("word<unc>");
    assert_eq!(base, "word");
    assert_eq!(modifiers, vec![(String::from("unc"), String::new())]);
}
