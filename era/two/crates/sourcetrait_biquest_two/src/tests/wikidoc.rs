use crate::wikidoc::regular_participle;
use crate::wikidoc::regular_past;
use crate::wikidoc::regular_plural;
use crate::wikidoc::render_word_document;
use crate::wikixml::WikiPage;

#[test]
fn regular_inflections_follow_the_default_rules() {
    assert_eq!(regular_plural("Microsoft"), "Microsofts");
    assert_eq!(regular_plural("church"), "churches");
    assert_eq!(regular_plural("fly"), "flies");
    assert_eq!(regular_plural("day"), "days");
    assert_eq!(regular_participle("Microsoft"), "Microsofting");
    assert_eq!(regular_participle("love"), "loving");
    assert_eq!(regular_participle("see"), "seeing");
    assert_eq!(regular_past("Microsoft"), "Microsofted");
    assert_eq!(regular_past("love"), "loved");
    assert_eq!(regular_past("try"), "tried");
    assert_eq!(regular_past("pass"), "passed");
}

const PAGE_TEXT: &str = "==English==\n\n===Etymology===\nA {{blend|en|alpha|beta|nocap=1}}.\n\n===Noun===\n{{en-noun}}\n\n# {{lb|en|informal}} A [[pet]], with [[friend]]s.\n#: {{syn|en|hound<q:x>|pooch}}\n#* {{quote-book|en|year=2001|title=Dogs|passage=the '''dog''' ran}}\n\n====Derived terms====\n{{col|en||dog days|dogged}}\n\n====Translations====\n{{trans-top|x}}\n* Finnish: {{t|fi|koira}}\n{{trans-bottom}}\n\n===Anagrams===\n* {{anagrams|en|a=dgo|god}}\n\n==Finnish==\n\n===Noun===\n{{fi-noun}}\n\n# ignored\n";

const EXPECTED: &str = "# dog\n\n## Etymology\n\nA blend of [alpha] + [beta].\n\n## Noun\n\n### dog\n(plural [dogs])\n\n1. (informal) A [pet], with [friends](friend).\n   - *Dogs*: \"the **dog** ran\"\n\n### Synonyms\n- [hound]\n- [pooch]\n\n### Derived terms\n- [dog days]\n- [dogged]\n\n## Anagrams\n- [god]\n";

fn page(title: &str, text: &str) -> WikiPage {
    WikiPage {
        title: title.to_string(),
        ns: 0,
        id: 1,
        redirect: None,
        text: text.to_string(),
    }
}

#[test]
fn a_page_renders_to_the_document_standard() {
    let document = render_word_document(&[page("dog", PAGE_TEXT)]).expect("render");
    assert_eq!(document.markdown, EXPECTED);
    let classes: Vec<&str> =
        document.audit.iter().map(|row| row.class.as_str()).collect();
    assert!(classes.contains(&"section_dropped"));
}

#[test]
fn only_the_english_subtree_renders() {
    let document = render_word_document(&[page("dog", PAGE_TEXT)]).expect("render");
    assert!(!document.markdown.contains("Finnish"));
    assert!(!document.markdown.contains("ignored"));
}

#[test]
fn multi_etymology_pages_render_the_uniform_levels() {
    let text = "==English==\n\n===Etymology 1===\nOne origin.\n\n====Noun====\n{{en-noun}}\n\n# A sense.\n\n=====Derived terms=====\n{{col|en|term}}\n\n===Etymology 2===\nAnother origin.\n\n====Verb====\n{{en-verb}}\n\n# To sense.\n";
    let document = render_word_document(&[page("bank", text)]).expect("render");
    let headings: Vec<&str> = document
        .markdown
        .lines()
        .filter(|line| line.starts_with('#'))
        .collect();
    assert_eq!(
        headings,
        vec![
            "# bank",
            "## Etymology 1",
            "## Noun",
            "### bank",
            "### Derived terms",
            "## Etymology 2",
            "## Verb",
            "### bank",
        ]
    );
}

#[test]
fn source_italics_flatten_and_bold_survives() {
    let text = "==English==\n\n===Etymology===\nFrom ''italic'' and '''bold''' use.\n";
    let document = render_word_document(&[page("x", text)]).expect("render");
    assert!(document.markdown.contains("From italic and **bold** use."));
}

#[test]
fn argless_etymology_templates_render_without_panic() {
    let text = "==English==\n\n===Etymology===\nA {{blend}} and {{af|nocap=1}} case.\n";
    let document = render_word_document(&[page("x", text)]).expect("render");
    assert!(document.markdown.contains("## Etymology"));
}

#[test]
fn audit_details_are_single_line() {
    let text = "==English==\n\n===Etymology===\nSee [[File:a\nb.png|x]] here.\n";
    let document = render_word_document(&[page("x", text)]).expect("render");
    assert!(!document.audit.is_empty());
    for row in &document.audit {
        assert!(!row.detail.contains('\n'), "multi-line detail: {:?}", row.detail);
    }
}

#[test]
fn typography_normalizes_in_rendered_text() {
    let text = "==English==\n\n===Etymology===\nFrom \u{201C}so\u{2014}called\u{201D} use.\n";
    let document = render_word_document(&[page("x", text)]).expect("render");
    assert!(document.markdown.contains("From \"so - called\" use."));
}
