use crate::wikixml::parse_index_line;
use crate::wikixml::PageReader;

/// A rootless page sequence, as a multistream block carries it: a
/// content page with entities in title and text, then a redirect.
const FRAGMENT: &str = r#"  <page>
    <title>AT&amp;T</title>
    <ns>0</ns>
    <id>42</id>
    <revision>
      <id>777</id>
      <timestamp>2026-08-01T00:00:00Z</timestamp>
      <contributor>
        <username>Box</username>
        <id>9</id>
      </contributor>
      <model>wikitext</model>
      <format>text/x-wiki</format>
      <text bytes="43" xml:space="preserve">==English==
A &lt;ref&gt;cited&lt;/ref&gt; {{lb|en|x}} line.</text>
    </revision>
  </page>
  <page>
    <title>Micro$oft</title>
    <ns>0</ns>
    <id>77</id>
    <redirect title="Microsoft &amp; Co" />
    <revision>
      <id>778</id>
      <text bytes="31" xml:space="preserve">#REDIRECT [[Microsoft &amp; Co]]</text>
    </revision>
  </page>
"#;

#[test]
fn pages_parse_with_entities_positions_and_redirect() {
    let mut reader = PageReader::new(FRAGMENT.as_bytes());
    let first = reader.next_page().expect("parse").expect("first page");
    assert_eq!(first.title, "AT&T");
    assert_eq!(first.ns, 0);
    assert_eq!(first.id, 42);
    assert_eq!(first.redirect, None);
    assert_eq!(
        first.text,
        "==English==\nA <ref>cited</ref> {{lb|en|x}} line."
    );
    let second = reader.next_page().expect("parse").expect("second page");
    assert_eq!(second.title, "Micro$oft");
    assert_eq!(second.id, 77);
    assert_eq!(second.redirect.as_deref(), Some("Microsoft & Co"));
    assert!(reader.next_page().expect("parse").is_none());
}

#[test]
fn index_lines_keep_title_colons() {
    let row = parse_index_line("654:9007:Appendix:Glossary of chess").expect("parse");
    assert_eq!(row.offset, 654);
    assert_eq!(row.page_id, 9007);
    assert_eq!(row.title, "Appendix:Glossary of chess");
}

#[test]
fn index_line_shape_is_enforced() {
    assert!(parse_index_line("nonsense").is_err());
    assert!(parse_index_line("12:x:Title").is_err());
}
