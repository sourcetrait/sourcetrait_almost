use crate::wikitext::normalize_typography;
use crate::wikitext::parse_blocks;
use crate::wikitext::parse_inline_text;
use crate::wikitext::Block;
use crate::wikitext::Emphasis;
use crate::wikitext::Inline;

fn text_of(inlines: &[Inline]) -> String {
    inlines
        .iter()
        .filter_map(|inline| match inline {
            Inline::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn typography_normalizes_the_ruled_set() {
    assert_eq!(normalize_typography("don\u{2019}t"), "don't");
    assert_eq!(normalize_typography("\u{2018}quoted\u{2019}"), "'quoted'");
    assert_eq!(normalize_typography("\u{201C}say\u{201D}"), "\"say\"");
    assert_eq!(normalize_typography("1990\u{2013}95"), "1990-95");
}

#[test]
fn typography_normalizes_the_extended_set() {
    assert_eq!(normalize_typography("1\u{2044}2"), "1/2");
    assert_eq!(normalize_typography("3\u{2212}1"), "3-1");
    assert_eq!(normalize_typography("so\u{2026}"), "so...");
    assert_eq!(normalize_typography("\u{2022} item"), "- item");
    assert_eq!(normalize_typography("\u{00AB}x\u{00BB} \u{201E}y\u{201D}"), "\"x\" \"y\"");
    assert_eq!(normalize_typography("\u{2039}z\u{203A} \u{201A}w\u{2019}"), "'z' 'w'");
    assert_eq!(normalize_typography("it\u{00B4}s"), "it's");
    assert_eq!(normalize_typography("a\u{00A0}b\u{2009}c\u{202F}d"), "a b c d");
    assert_eq!(normalize_typography("a\u{00AD}b\u{200B}c\u{200E}d"), "abcd");
    assert_eq!(normalize_typography("a\u{2015}b 5\u{2012}6"), "a - b 5-6");
}

#[test]
fn semantic_symbols_keep_their_rows() {
    assert_eq!(normalize_typography("hy\u{00B7}phen"), "hy\u{00B7}phen");
    assert_eq!(normalize_typography("2\u{00D7}4"), "2\u{00D7}4");
    assert_eq!(normalize_typography("Hawai\u{02BC}i"), "Hawai\u{02BC}i");
    assert_eq!(normalize_typography("5\u{2032} 10\u{2033}"), "5\u{2032} 10\u{2033}");
}

#[test]
fn em_dash_carries_its_missing_spacing() {
    assert_eq!(normalize_typography("war\u{2014}peace"), "war - peace");
    assert_eq!(normalize_typography("war \u{2014} peace"), "war - peace");
    assert_eq!(normalize_typography("war\u{2014} peace"), "war - peace");
    assert_eq!(normalize_typography("war \u{2014}peace"), "war - peace");
    assert_eq!(normalize_typography("\u{2014}peace"), "- peace");
    assert_eq!(normalize_typography("war\u{2014}"), "war -");
}

#[test]
fn blocks_classify_by_line_shape() {
    let text = "==English==\n\n===Noun===\n{{en-noun}}\n\n# A gloss.\n#* A quote line.\n----\n";
    let blocks = parse_blocks(text).expect("parse");
    match &blocks[0] {
        Block::Heading { level, content } => {
            assert_eq!(*level, 2);
            assert_eq!(text_of(content), "English");
        }
        other => panic!("expected heading, got {other:?}"),
    }
    assert_eq!(blocks[1], Block::Blank);
    assert!(matches!(&blocks[2], Block::Heading { level: 3, .. }));
    assert!(matches!(&blocks[3], Block::Paragraph { .. }));
    match &blocks[5] {
        Block::ListItem { markers, content } => {
            assert_eq!(markers, "#");
            assert_eq!(text_of(content), " A gloss.");
        }
        other => panic!("expected list item, got {other:?}"),
    }
    match &blocks[6] {
        Block::ListItem { markers, .. } => assert_eq!(markers, "#*"),
        other => panic!("expected list item, got {other:?}"),
    }
    assert_eq!(blocks[7], Block::HorizontalRule);
}

#[test]
fn the_microsoft_gloss_line_parses_whole() {
    let text = "# {{lb|en|slang|transitive}} To [[render]] more like Microsoft \
                with regards to [[business]] [[practice]]s and [[tactic]]s.";
    let blocks = parse_blocks(text).expect("parse");
    let Block::ListItem { markers, content } = &blocks[0] else {
        panic!("expected list item");
    };
    assert_eq!(markers, "#");
    let templates: Vec<_> = content
        .iter()
        .filter_map(|inline| match inline {
            Inline::Template(template) => Some(template),
            _ => None,
        })
        .collect();
    assert_eq!(templates.len(), 1);
    assert_eq!(templates[0].name, "lb");
    assert_eq!(templates[0].positional, vec!["en", "slang", "transitive"]);
    let links: Vec<_> = content
        .iter()
        .filter_map(|inline| match inline {
            Inline::Link(link) => Some((link.target.as_str(), link.trail.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        links,
        vec![("render", ""), ("business", ""), ("practice", "s"), ("tactic", "s")]
    );
}

#[test]
fn templates_split_named_and_positional_nesting_aware() {
    let inlines = parse_inline_text(
        "{{quote-journal|en|author=w:Robert Metcalfe|title=Web [[war|Wars]]|passage=the '''Microsoft''' of it}}",
    )
    .expect("parse");
    let Inline::Template(template) = &inlines[0] else {
        panic!("expected template");
    };
    assert_eq!(template.name, "quote-journal");
    assert_eq!(template.positional, vec!["en"]);
    assert_eq!(
        template.named,
        vec![
            (String::from("author"), String::from("w:Robert Metcalfe")),
            (String::from("title"), String::from("Web [[war|Wars]]")),
            (String::from("passage"), String::from("the '''Microsoft''' of it")),
        ]
    );
}

#[test]
fn templates_cross_lines_and_nest() {
    let text = "# gloss\n#* {{quote-book|en|year=2001\n|passage=nested {{w|Corp}} here}}\nplain";
    let blocks = parse_blocks(text).expect("parse");
    let Block::ListItem { content, .. } = &blocks[1] else {
        panic!("expected the quote list item");
    };
    let Some(Inline::Template(template)) = content
        .iter()
        .find(|inline| matches!(inline, Inline::Template(_)))
    else {
        panic!("expected a template");
    };
    assert_eq!(template.name, "quote-book");
    assert_eq!(
        template.named,
        vec![
            (String::from("year"), String::from("2001")),
            (String::from("passage"), String::from("nested {{w|Corp}} here")),
        ]
    );
    assert!(matches!(&blocks[2], Block::Paragraph { .. }));
}

#[test]
fn links_carry_display_and_colon_targets() {
    let inlines =
        parse_inline_text("[[practice|practices]] and [[:Category:Microsoft]]")
            .expect("parse");
    let links: Vec<_> = inlines
        .iter()
        .filter_map(|inline| match inline {
            Inline::Link(link) => Some(link),
            _ => None,
        })
        .collect();
    assert_eq!(links[0].target, "practice");
    assert_eq!(links[0].display.as_deref(), Some("practices"));
    assert_eq!(links[0].trail, "");
    assert_eq!(links[1].target, ":Category:Microsoft");
}

#[test]
fn uppercase_never_blends_into_a_trail() {
    let inlines = parse_inline_text("[[dog]]s [[dog]]Houses").expect("parse");
    let links: Vec<_> = inlines
        .iter()
        .filter_map(|inline| match inline {
            Inline::Link(link) => Some(link.trail.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(links, vec!["s", ""]);
}

#[test]
fn emphasis_toggles_by_run_length() {
    let inlines = parse_inline_text("''i'' '''b''' '''''bi''''' can't").expect("parse");
    let toggles: Vec<_> = inlines
        .iter()
        .filter_map(|inline| match inline {
            Inline::Emphasis(emphasis) => Some(*emphasis),
            _ => None,
        })
        .collect();
    assert_eq!(
        toggles,
        vec![
            Emphasis::Italic,
            Emphasis::Italic,
            Emphasis::Bold,
            Emphasis::Bold,
            Emphasis::BoldItalic,
            Emphasis::BoldItalic,
        ]
    );
    assert!(text_of(&inlines).contains("can't"));
}

#[test]
fn comments_strip_and_entities_resolve() {
    let inlines = parse_inline_text("a<!-- hidden | {{x}} -->b &amp; &#65; &#x42; &ndash; &unknown; c")
        .expect("parse");
    assert_eq!(text_of(&inlines), "ab & A B \u{2013} &unknown; c");
}

#[test]
fn refs_nowiki_and_html_tags_tokenize() {
    let inlines = parse_inline_text(
        "x<ref name=a/>y<ref>cited {{tpl}}</ref><nowiki>{{raw}}</nowiki><br/>z<sub>1</sub>",
    )
    .expect("parse");
    let kinds: Vec<&str> = inlines
        .iter()
        .map(|inline| match inline {
            Inline::Text(_) => "text",
            Inline::Ref { content: None, .. } => "ref_void",
            Inline::Ref { content: Some(_), .. } => "ref",
            Inline::Nowiki(_) => "nowiki",
            Inline::Html { .. } => "html",
            _ => "other",
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            "text", "ref_void", "text", "ref", "nowiki", "html", "text", "html",
            "text", "html",
        ]
    );
    let Inline::Ref { content: Some(content), .. } = &inlines[3] else {
        panic!("expected the paired ref");
    };
    assert_eq!(content, "cited {{tpl}}");
    let Inline::Nowiki(raw) = &inlines[4] else {
        panic!("expected nowiki");
    };
    assert_eq!(raw, "{{raw}}");
}

#[test]
fn external_links_split_url_and_label() {
    let inlines = parse_inline_text("[https://a.example one two] [not a url]")
        .expect("parse");
    let Inline::ExternalLink { url, label } = &inlines[0] else {
        panic!("expected external link");
    };
    assert_eq!(url, "https://a.example");
    assert_eq!(label.as_deref(), Some("one two"));
    assert!(text_of(&inlines).contains("[not a url]"));
}

#[test]
fn tables_capture_raw_and_nested() {
    let text = "before\n{| class=x\n|-\n| cell\n{| inner\n|}\n|}\nafter";
    let blocks = parse_blocks(text).expect("parse");
    let Block::Table { source } = &blocks[1] else {
        panic!("expected table, got {:?}", blocks[1]);
    };
    assert!(source.starts_with("{| class=x"));
    assert!(source.ends_with("|}"));
    assert!(source.contains("{| inner"));
    assert!(matches!(&blocks[2], Block::Paragraph { .. }));
}

#[test]
fn an_ampersand_before_a_straddling_multibyte_char_stays_literal() {
    let inlines = parse_inline_text("&0123456789\u{2014}xxxx tail").expect("parse");
    assert!(text_of(&inlines).starts_with("&0123456789"));
}

#[test]
fn multibyte_content_scans_safely() {
    let text = "* {{IPA|en|/\u{02C8}ma\u{026A}k\u{0279}\u{0259}\u{02CC}s\u{0252}ft/|a=RP}}";
    let blocks = parse_blocks(text).expect("parse");
    let Block::ListItem { content, .. } = &blocks[0] else {
        panic!("expected list item");
    };
    let Some(Inline::Template(template)) = content
        .iter()
        .find(|inline| matches!(inline, Inline::Template(_)))
    else {
        panic!("expected the IPA template");
    };
    assert_eq!(template.name, "IPA");
    assert_eq!(template.positional.len(), 2);
    assert!(template.positional[1].starts_with("/\u{02C8}"));
    assert_eq!(template.named, vec![(String::from("a"), String::from("RP"))]);
}

#[test]
fn unterminated_structure_degrades_to_literal_text() {
    // The MediaWiki behavior: an unterminated construct renders as
    // literal text and the page survives - one broken construct must
    // never fail a whole page (the here-page class).
    let blocks = parse_blocks("a {{never closes").expect("parse");
    let Block::Paragraph { content } = &blocks[0] else {
        panic!("expected paragraph");
    };
    assert_eq!(text_of(content), "a {{never closes");
    let blocks = parse_blocks("[[never closes").expect("parse");
    let Block::Paragraph { content } = &blocks[0] else {
        panic!("expected paragraph");
    };
    assert_eq!(text_of(content), "[[never closes");
    let blocks = parse_blocks("<ref>never closes").expect("parse");
    let Block::Paragraph { content } = &blocks[0] else {
        panic!("expected paragraph");
    };
    assert_eq!(text_of(content), "<ref>never closes");
    let blocks = parse_blocks("x <nowiki>never closes").expect("parse");
    let Block::Paragraph { content } = &blocks[0] else {
        panic!("expected paragraph");
    };
    assert_eq!(text_of(content), "x <nowiki>never closes");
    // An unterminated table swallows to the end as a table block.
    let blocks = parse_blocks("{| never closes\n| cell").expect("parse");
    let Block::Table { source } = &blocks[0] else {
        panic!("expected table, got {:?}", blocks[0]);
    };
    assert!(source.contains("| cell"));
}
