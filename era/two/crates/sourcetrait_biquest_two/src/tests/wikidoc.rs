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

fn pos_doc(pos: &str, template_line: &str, title: &str) -> String {
    let text = format!("==English==\n\n==={pos}===\n{template_line}\n\n# A sense.\n");
    render_word_document(&[page(title, &text)]).expect("render").markdown
}

#[test]
fn noun_specs_derive_the_documented_plurals() {
    assert!(pos_doc("Noun", "{{en-noun|-}}", "awe").contains("(uncountable)"));
    assert!(pos_doc("Noun", "{{en-noun|-|+}}", "reality")
        .contains("(usually uncountable, plural [realities])"));
    assert!(pos_doc("Noun", "{{en-noun|~}}", "beer")
        .contains("(countable and uncountable, plural [beers])"));
    assert!(pos_doc("Noun", "{{en-noun|+|octopi|octopodes}}", "octopus")
        .contains("(plural [octopuses] or [octopi] or [octopodes])"));
    assert!(pos_doc("Noun", "{{en-noun|*}}", "sheep").contains("(plural [sheep])"));
    assert!(pos_doc("Noun", "{{en-noun|++}}", "quiz").contains("(plural [quizzes])"));
    assert!(pos_doc("Noun", "{{en-noun|+|treen<l:obsolete>}}", "tree")
        .contains("(plural [trees] or [treen] (obsolete))"));
    assert!(pos_doc("Noun", "{{en-noun|p|attr=pant}}", "pants")
        .contains("(plural only, attributive [pant])"));
}

#[test]
fn verb_slots_and_indicators_derive_the_documented_forms() {
    assert!(pos_doc("Verb", "{{en-verb}}", "flip").contains(
        "(third-person singular simple present [flips], present participle \
         [flipping], simple past and past participle [flipped])"
    ));
    assert!(pos_doc("Verb", "{{en-verb}}", "tie").contains("[tying]"));
    assert!(pos_doc("Verb", "{{en-verb|does|doing|did|done}}", "do")
        .contains("simple past [did], past participle [done]"));
    assert!(pos_doc("Verb", "{{en-verb|+|+|+,wrought<l:obsolete>}}", "work")
        .contains("simple past and past participle [worked] or [wrought] (obsolete)"));
    let can = pos_doc("Verb", "{{en-verb|can|-|could|-}}", "can");
    assert!(
        can.contains("(third-person singular simple present [can], simple past [could])"),
        "got: {can}"
    );
    let log_on = pos_doc("Verb", "{{en-verb|*}}", "log on");
    assert!(
        log_on.contains("[logs on]")
            && log_on.contains("[logging on]")
            && log_on.contains("[logged on]"),
        "got: {log_on}"
    );
    let grudge = pos_doc("Verb", "{{en-verb|hold<,,held> a grudge}}", "hold a grudge");
    assert!(
        grudge.contains("[holds a grudge]")
            && grudge.contains("simple past and past participle [held a grudge]"),
        "got: {grudge}"
    );
}

#[test]
fn graded_specs_derive_comparatives_and_superlatives() {
    assert!(pos_doc("Adjective", "{{en-adj|er}}", "hard")
        .contains("(comparative [harder], superlative [hardest])"));
    assert!(pos_doc("Adjective", "{{en-adj|-}}", "coal").contains("(not comparable)"));
    let avid = pos_doc("Adjective", "{{en-adj|more,avider<l:less common>}}", "avid");
    assert!(
        avid.contains("comparative [more avid] or [avider] (less common)"),
        "got: {avid}"
    );
    assert!(
        avid.contains("superlative [most avid] or [avidest] (less common)"),
        "got: {avid}"
    );
    let fitting = pos_doc("Adjective", "{{en-adj|+first}}", "loose-fitting");
    assert!(
        fitting.contains("(comparative [looser-fitting], superlative [loosest-fitting])"),
        "got: {fitting}"
    );
    assert!(pos_doc("Adverb", "{{en-adv|-}}", "solely").contains("(not comparable)"));
}

#[test]
fn head_template_pairs_and_pointers_render() {
    let lemma = pos_doc("Noun", "{{head|en|noun|plural|lemmas|or|lemmata}}", "lemma");
    assert!(lemma.contains("(plural [lemmas] or [lemmata])"), "got: {lemma}");
    let milk = pos_doc(
        "Noun",
        "{{head|en|noun|countable and uncountable||plural|milks}}",
        "milk",
    );
    assert!(
        milk.contains("(countable and uncountable, plural [milks])"),
        "got: {milk}"
    );
    // A bare head carries no inflections and is handled, not audited.
    let text = "==English==\n\n===Verb===\n{{head|en|verb form}}\n\n# A sense.\n";
    let document = render_word_document(&[page("runs", text)]).expect("render");
    assert!(
        document.audit.iter().all(|row| !row.class.starts_with("head_")),
        "audit: {:?}",
        document.audit
    );
    // The wikipedia pointer routes to its own audit class and the
    // head template still lands.
    let text = "==English==\n\n===Noun===\n{{wp}}\n\n{{en-noun}}\n\n# A sense.\n";
    let document = render_word_document(&[page("dog", text)]).expect("render");
    assert!(document.markdown.contains("(plural [dogs])"));
    assert!(document.audit.iter().any(|row| row.class == "wikipedia_pointer"));
    // head= overrides the headword line.
    let intj = pos_doc("Interjection", "{{en-intj|head=not!}}", "not");
    assert!(intj.contains("### not!"), "got: {intj}");
}

#[test]
fn accent_codes_resolve_against_the_label_data() {
    let text = "==English==\n\n===Pronunciation===\n* {{IPA|en|/x/|a=GenAm,SSB,Scotland,square-nurse,non-rhotic,nMmmm,æ-tensing,non-æ-tensing}}\n\n===Noun===\n{{en-noun}}\n\n# A sense.\n";
    let document = render_word_document(&[page("x", text)]).expect("render");
    assert!(
        document.markdown.contains(
            "IPA (General American, Standard Southern British, Scotland, \
             fair-fur merger, non-rhotic, without the Mary-marry-merry merger, \
             æ-raising, without æ-raising): /x/"
        ),
        "got: {}",
        document.markdown
    );
    assert!(document.audit.iter().all(|row| row.class != "accent_code_unknown"));
}

#[test]
fn numbered_variants_and_pron_pairs_render() {
    let flied = pos_doc("Verb", "{{en-verb|flies|flying|flied|past2=flyed}}", "fly");
    assert!(
        flied.contains("simple past and past participle [flied] or [flyed]"),
        "got: {flied}"
    );
    let munchies = pos_doc("Noun", "{{en-noun|p|sg=munchie|sg2=munchy}}", "munchies");
    assert!(
        munchies.contains("(plural only, singular [munchie] or [munchy])"),
        "got: {munchies}"
    );
    let numbered = pos_doc("Noun", "{{en-noun|1=-}}", "awe");
    assert!(numbered.contains("(uncountable)"), "got: {numbered}");
    let pron = pos_doc(
        "Pronoun",
        "{{en-pron|nominative|thou|reflexive|thyself|desc=second-person singular}}",
        "thee",
    );
    assert!(
        pron.contains("(nominative [thou], reflexive [thyself], second-person singular)"),
        "got: {pron}"
    );
    let text = "==English==\n\n===Preposition===\n{{en-head|prep}}\n\n# A sense.\n";
    let document = render_word_document(&[page("at", text)]).expect("render");
    assert!(
        document.audit.iter().all(|row| !row.class.starts_with("head_")),
        "audit: {:?}",
        document.audit
    );
}

#[test]
fn form_of_definitions_render() {
    let text = "==English==\n\n===Noun===\n{{en-noun|-}}\n\n# {{synonym of|en|DDR}}\n# {{plural of|en|word}}\n# {{alt form|en|colour}}\n";
    let document = render_word_document(&[page("2DR", text)]).expect("render");
    assert!(
        document.markdown.contains("1. Synonym of [DDR]"),
        "got: {}",
        document.markdown
    );
    assert!(document.markdown.contains("2. Plural of [word]"));
    assert!(document.markdown.contains("3. Alternative form of [colour]"));
    // The headword override renders plain: no anchors in headings.
    let text =
        "==English==\n\n===Noun===\n{{en-noun|head=[[μ]]-[[scope]]|s}}\n\n# A sense.\n";
    let document = render_word_document(&[page("μ-scope", text)]).expect("render");
    assert!(
        document.markdown.contains("### μ-scope"),
        "got: {}",
        document.markdown
    );
}

#[test]
fn ipa_separators_and_label_connectors_render_clean() {
    let text = "==English==\n\n===Pronunciation===\n* {{IPA|en|/a/|;|/b/|~|a=<<GA>> <<Scotland>>,AU!Australia}}\n\n===Noun===\n{{en-noun|+|sheeps<l:nonstandard,humorous,or,childish>}}\n\n# A sense.\n";
    let document = render_word_document(&[page("x", text)]).expect("render");
    assert!(
        document
            .markdown
            .contains("IPA (General American, Scotland, Australia): /a/, /b/"),
        "got: {}",
        document.markdown
    );
    assert!(
        document.markdown.contains("(nonstandard, humorous or childish)"),
        "got: {}",
        document.markdown
    );
}

#[test]
fn typography_normalizes_in_rendered_text() {
    let text = "==English==\n\n===Etymology===\nFrom \u{201C}so\u{2014}called\u{201D} use.\n";
    let document = render_word_document(&[page("x", text)]).expect("render");
    assert!(document.markdown.contains("From \"so - called\" use."));
}
