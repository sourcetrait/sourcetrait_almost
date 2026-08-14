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
fn en_degree_forms_render_and_the_en_family_stays_out_of_the_shape_rule() {
    let text = "==English==\n\n===Adjective===\n{{head|en|superlative adjective}}\n\n# {{en-superlative of|scrungy}}\n";
    let document = render_word_document(&[page("scrungiest", text)]).expect("render");
    assert!(
        document
            .markdown
            .contains("1. Superlative form of [scrungy]: most [scrungy]"),
        "got: {}",
        document.markdown
    );
    let text = "==English==\n\n===Adjective===\n{{head|en|comparative adjective}}\n\n# {{en-comparative of|hard}}\n";
    let document = render_word_document(&[page("harder", text)]).expect("render");
    assert!(document.markdown.contains("1. Comparative form of [hard]: more [hard]"));
    // Any other en-prefixed " of" template audits rather than
    // rendering through the shifted generic slots.
    let text = "==English==\n\n===Verb===\n{{head|en|verb form}}\n\n# {{en-archaic second-person singular of|do}}\n";
    let document = render_word_document(&[page("dost", text)]).expect("render");
    assert!(!document.markdown.contains("[]"), "got: {}", document.markdown);
    assert!(document
        .audit
        .iter()
        .any(|row| row.class == "template_unhandled"));
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

fn ety_doc(line: &str, title: &str) -> String {
    let text = format!("==English==\n\n===Etymology===\n{line}\n");
    render_word_document(&[page(title, &text)]).expect("render").markdown
}

#[test]
fn etymology_references_render_language_and_term() {
    assert!(ety_doc("From {{der|en|la|verbum}}.", "word")
        .contains("From Latin [verbum]."));
    assert!(ety_doc("{{bor+|en|fr|mot}}.", "word")
        .contains("Borrowed from French [mot]."));
    assert!(ety_doc("{{inh+|en|enm|two}}.", "two")
        .contains("Inherited from Middle English [two]."));
    assert!(ety_doc("From {{uder|en|la|cubitus||elbow}}.", "cubit")
        .contains("From Latin [cubitus] (\"elbow\")."));
    assert!(ety_doc("{{lbor|en|la|aberratio|t=wandering}}.", "aberration")
        .contains("Learned borrowing from Latin [aberratio] (\"wandering\")."));
    assert!(ety_doc("Compare {{cog|ca,oc,pt,es|salar}}.", "saler").contains(
        "Compare Catalan, Occitan, Portuguese and Spanish [salar]."
    ));
    assert!(ety_doc("Cognate with the {{cog|en|-}} name {{m|en|Leonidas}}.", "x")
        .contains("Cognate with the English name [Leonidas]."));
    assert!(ety_doc("From {{der|en|la|femina|alt=fēmina}}.", "female")
        .contains("From Latin [fēmina](femina)."));
    assert!(ety_doc("From {{m+|ang|earc}}.", "ark")
        .contains("From Old English [earc]."));
}

#[test]
fn unknown_language_codes_render_verbatim_and_audit() {
    let text = "==English==\n\n===Etymology===\nFrom {{der|en|LL.|word}}.\n";
    let document = render_word_document(&[page("x", text)]).expect("render");
    assert!(document.markdown.contains("From LL. [word]."));
    assert!(document
        .audit
        .iter()
        .any(|row| row.class == "language_code_unknown" && row.detail == "LL."));
}

#[test]
fn doublet_unknown_and_formation_statements_render() {
    assert!(ety_doc("{{doublet|en|fire}}.", "pyre").contains("Doublet of [fire]."));
    assert!(ety_doc("{{doublet|en|advoke|avouch|avow}}.", "avocate")
        .contains("Doublet of [advoke], [avouch] and [avow]."));
    assert!(ety_doc("{{doublet|en|frail|t1=weak}}.", "fragile")
        .contains("Doublet of [frail] (\"weak\")."));
    assert!(ety_doc("{{unk|en}}.", "x").contains("Unknown."));
    assert!(ety_doc("Origin {{unk|en|nocap=1}}.", "x").contains("Origin unknown."));
    assert!(ety_doc("{{unc|en}}.", "x").contains("Uncertain."));
    assert!(ety_doc("{{back-form|en|editor}}.", "edit")
        .contains("Back-formation from [editor]."));
    assert!(ety_doc("{{surf|en|ignore|-ance}}.", "ignorance")
        .contains("By surface analysis, [ignore] + [-ance]."));
    assert!(ety_doc("{{calque|en|de|Weltanschauung}}.", "worldview")
        .contains("Calque of German [Weltanschauung]."));
}

#[test]
fn etymon_renders_nothing_bare_and_one_step_with_text() {
    let bare = "==English==\n\n===Etymology===\n{{etymon|en|id=x|inh|enm:fader}}\nFrom prose.\n";
    let document = render_word_document(&[page("father", bare)]).expect("render");
    assert!(!document.markdown.contains("fader"));
    assert!(document.audit.iter().all(|row| row.class != "template_unhandled"));
    assert!(ety_doc("{{etymon|en|id=b|:bor|fr:bouquet|text=+}}", "bouquet")
        .contains("Borrowed from French [bouquet]"));
    assert!(ety_doc("{{ety|en|id=b|:af|un-|happy|text=+}}", "unhappy")
        .contains("[un-] + [happy]"));
}

#[test]
fn inflection_of_renders_resolved_tags() {
    let text = "==English==\n\n===Verb===\n{{head|en|verb form}}\n\n# {{infl of|en|amar||3|s|pres|act|ind}}\n";
    let document = render_word_document(&[page("aman", text)]).expect("render");
    assert!(
        document
            .markdown
            .contains("1. Third-person singular present active indicative of [amar]"),
        "got: {}",
        document.markdown
    );
    let text = "==English==\n\n===Noun===\n{{head|en|noun form}}\n\n# {{noun form of|en|word||p}}\n";
    let document = render_word_document(&[page("words", text)]).expect("render");
    assert!(
        document.markdown.contains("1. Plural of [word]"),
        "got: {}",
        document.markdown
    );
    let text = "==English==\n\n===Noun===\n{{head|en|noun form}}\n\n# {{infl of|en|путь||gen//dat|s|;|nom//acc|p}}\n";
    let document = render_word_document(&[page("пути", text)]).expect("render");
    assert!(
        document
            .markdown
            .contains("1. Genitive/dative singular; nominative/accusative plural of [путь]"),
        "got: {}",
        document.markdown
    );
}

fn sense_doc(line: &str, title: &str) -> String {
    let text =
        format!("==English==\n\n===Noun===\n{{{{en-noun}}}}\n\n{line}\n");
    render_word_document(&[page(title, &text)]).expect("render").markdown
}

#[test]
fn surname_and_given_name_definitions_render() {
    assert!(sense_doc("# {{surname|en}}.", "Weber").contains("1. A surname."));
    assert!(sense_doc("# {{surname|en|English}}.", "Weber")
        .contains("1. An English surname."));
    assert!(sense_doc("# {{surname|en|from=patronymics}}.", "Johnson")
        .contains("1. A surname originating as a patronymic."));
    assert!(sense_doc("# {{surname|en|g=m|from=Irish}}.", "Murphy")
        .contains("1. A male surname from Irish."));
    assert!(sense_doc("# {{surname|en|from=Slavic languages}}.", "Halkin")
        .contains("1. A surname from the Slavic languages."));
    assert!(sense_doc("# {{given name|en|male}}.", "Amber")
        .contains("1. A male given name."));
    assert!(sense_doc("# {{given name|en|female|from=Hebrew|m=Daniel}}.", "Danielle")
        .contains("1. A female given name from Hebrew, masculine equivalent [Daniel]."));
    assert!(
        sense_doc("# {{given name|en|male|dimof=Barnabas,Bernard}}.", "Barney")
            .contains("1. A diminutive of the male given names [Barnabas] and [Bernard]."),
    );
    assert!(sense_doc("# {{given name|en|unisex|from=surnames}}.", "Taylor")
        .contains("1. A unisex given name transferred from the surname."));
}

#[test]
fn place_definitions_render_the_documented_shapes() {
    assert!(sense_doc("# {{place|en|city|p/Ontario|c/Canada}}.", "Toronto")
        .contains("1. A city in [Ontario], [Canada]."));
    assert!(sense_doc("# {{place|en|country|cont/Europe}}.", "Germany")
        .contains("1. A country in [Europe]."));
    assert!(sense_doc("# {{place|en|city|c/Netherlands}}.", "Amsterdam")
        .contains("1. A city in the [Netherlands]."));
    assert!(sense_doc("# {{place|en|county|s/Virginia|c/United States}}.", "Fairfax")
        .contains("1. A county of [Virginia], [United States]."));
    assert!(sense_doc("# {{place|en|country|in central|cont/Europe}}.", "Germany")
        .contains("1. A country in central [Europe]."));
    assert!(sense_doc("# {{place|en|river|c/USA|and|c/Canada}}.", "Columbia")
        .contains("1. A river in the [United States] and [Canada]."));
    assert!(sense_doc("# {{place|en|river|c/Ukraine,Belarus,Poland}}.", "Bug")
        .contains("1. A river in [Ukraine], [Belarus] and [Poland]."));
    assert!(
        sense_doc("# {{place|en|city|in northeastern|s/Pennsylvania|c/United States}}.", "Scranton")
            .contains("1. A city in northeastern [Pennsylvania], [United States]."),
    );
    assert!(sense_doc("# {{place|en|capital city|c:pref/Georgia}}.", "Tbilisi")
        .contains("1. A capital city of the country of [Georgia]."));
    assert!(
        sense_doc("# {{place|en|country|in southern|cont/Europe|caplc=Rome}}.", "Italy")
            .contains(
                "1. A country in southern [Europe]; capital and largest city: [Rome]."
            ),
    );
    assert!(sense_doc(
        "# {{place|en|A <<neighborhood>> in <<city/Istanbul>>, <<c/Turkey>>}}.",
        "Fener"
    )
    .contains("1. A neighborhood in [Istanbul], [Turkey]."));
    let two = sense_doc(
        "# {{place|en|city/state capital|s/Rio de Janeiro|c/Brazil|;|former capital city|of|c/Brazil}}.",
        "Rio",
    );
    assert!(
        two.contains(
            "1. A city, state capital in [Rio de Janeiro], [Brazil]; \
             a former capital city of [Brazil]."
        ),
        "got: {two}"
    );
}

#[test]
fn sense_tax_and_inline_helpers_render() {
    let text = "==English==\n\n===Noun===\n{{en-noun}}\n\n# A fish.\n\n====Synonyms====\n* {{sense|an oath}} {{l|en|promise}}\n";
    let document = render_word_document(&[page("word", text)]).expect("render");
    assert!(
        document.markdown.contains("- (an oath): [promise]"),
        "got: {}",
        document.markdown
    );
    assert!(sense_doc("# The pike, {{taxfmt|Esox lucius|species}}.", "pike")
        .contains("1. The pike, [Esox lucius]."));
    assert!(sense_doc("# A {{taxlink|Felis silvestris|species}} cat.", "cat")
        .contains("1. A Felis silvestris cat."));
    assert!(sense_doc("# The {{vern|northern pike}}.", "pike")
        .contains("1. The [northern pike]."));
    assert!(sense_doc("# See {{cap|autumn}}.", "fall")
        .contains("1. See [Autumn](autumn)."));
    assert!(sense_doc("# {{n-g|A placeholder name.}}", "thing")
        .contains("1. A placeholder name."));
    assert!(sense_doc("# {{lang|fr|mot}} use.", "word").contains("1. mot use."));
}

#[test]
fn quotation_helpers_and_examples_render() {
    let text = "==English==\n\n===Noun===\n{{en-noun}}\n\n# A pet.\n#: {{ux|en|The '''dog''' barked.}}\n#* {{quote-book|en|title=Dogs|passage=a {{...}} dog}}\n";
    let document = render_word_document(&[page("dog", text)]).expect("render");
    assert!(
        document.markdown.contains("   - \"The **dog** barked.\""),
        "got: {}",
        document.markdown
    );
    assert!(
        document.markdown.contains("   - *Dogs*: \"a ... dog\""),
        "got: {}",
        document.markdown
    );
}

#[test]
fn derived_aliases_and_the_definition_rescue_arms_render() {
    // The 401k catch: nstd sp resolves through the derived alias
    // table (a template redirect the hand-grown list missed).
    assert!(sense_doc("# {{nstd sp|en|401(k)}}.", "401k")
        .contains("1. Nonstandard spelling of [401(k)]."));
    assert!(sense_doc("# {{obs form|en|word}}.", "worde")
        .contains("1. Obsolete form of [word]."));
    assert!(sense_doc("# {{alt case|en|internet}}.", "Internet")
        .contains("1. Alternative case form of [internet]."));
    assert!(sense_doc("# {{short for|en|gonna,going to|nocap=1}}.", "gon")
        .contains("1. short for [gonna], [going to]."));
    assert!(sense_doc("# {{only used in|en|get the drop on}}.", "drop")
        .contains("1. Only used in [get the drop on]."));
    assert!(sense_doc("# {{&lit|en|kick|the bucket}}.", "kick the bucket")
        .contains("1. Used other than figuratively or idiomatically: see [kick], [the bucket]."));
    assert!(sense_doc("# {{demonym-noun|en|the <<city:pref/Alexandria>>, <<c/Egypt>>}}.", "Alexandrian")
        .contains("1. A native or inhabitant of the city of [Alexandria], [Egypt]."));
    assert!(sense_doc("# {{demonym-adj|en|Arizona}}.", "Arizonan")
        .contains("1. Of, from, or relating to Arizona."));
    assert!(sense_doc("# {{SI-unit|en|micro|meter}}.", "micrometer").contains(
        "1. (metrology) An SI unit of length equal to 10^-6 [meter]s. Symbol: μm."
    ));
    assert!(sense_doc("# {{staco|Cleveland|CLE|Ohio}}.", "CLE").contains(
        "1. (rail transport) The station code of [CLE](Cleveland) in Ohio."
    ));
    assert!(sense_doc("# {{tcl|en|Belgium|id=Q31}}.", "Belgien")
        .contains("1. See [Belgium]."));
    assert!(
        sense_doc("# {{name translit|en|ru|Иван|type=male given name}}.", "Ivan")
            .contains("1. Transliteration of the Russian male given name [Иван]."),
    );
    // A label carrying a template renders it rather than leaking it.
    assert!(sense_doc("# {{lb|en|preceded by {{m|en|the}}}} A topic.", "talk")
        .contains("1. (preceded by [the]) A topic."));
}

#[test]
fn nested_subsenses_render_with_depth_numbering() {
    let text = "==English==\n\n===Noun===\n{{en-noun}}\n\n# A [[gentle]] incline.\n## {{lb|en|geomorphology}} A sloping landform.\n## {{lb|en|military}}\n### A fortification incline.\n###: {{syn|en|talus}}\n### An armour plate.\n## {{lb|en|post}} A mail sorter.\n# A second sense.\n";
    let document = render_word_document(&[page("glacis", text)]).expect("render");
    let expected = [
        "1. A [gentle] incline.",
        "   1. (geomorphology) A sloping landform.",
        "   2. (military)",
        "      1. A fortification incline.",
        "      2. An armour plate.",
        "   3. (post) A mail sorter.",
        "2. A second sense.",
    ];
    for line in expected {
        assert!(
            document.markdown.contains(line),
            "missing {line:?} in: {}",
            document.markdown
        );
    }
    assert!(document.markdown.contains("- [talus]"));
    assert!(document
        .audit
        .iter()
        .all(|row| row.class != "list_line_dropped"));
}

#[test]
fn silent_metadata_files_no_audit_anywhere() {
    let text = "==English==\n\n===Etymology===\n{{root|en|ine-pro|*bher-}}\nFrom use.\n\n===Noun===\n{{en-noun}}\n\n# A sense. {{C|en|Dogs}}\n\n{{cln|en|nouns}}\n\n====Synonyms====\n{{topics|en|animals}}\n* {{l|en|hound}}\n";
    let document = render_word_document(&[page("dog", text)]).expect("render");
    assert!(
        document
            .audit
            .iter()
            .all(|row| row.class != "template_unhandled"),
        "audit: {:?}",
        document.audit
    );
    assert!(!document.markdown.contains("ine-pro"));
}
