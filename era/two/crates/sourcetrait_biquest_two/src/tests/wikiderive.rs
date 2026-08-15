//! Derivation locks: the one-pass dump derivation over an in-memory
//! page stream - word-set assembly from titles plus head-derived
//! forms, per-POS sense collection, form and form-of links, and the
//! page filters (namespace, redirect, English presence, no-entry,
//! admission).
use crate::ucd::CharacterTable;
use crate::wikiderive::derive_pages;
use crate::wikixml::PageReader;

const FIXTURE: &str = r#"<mediawiki>
  <page>
    <title>dog</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <id>100</id>
      <text>==English==

===Noun===
{{en-noun}}

# A domestic animal.
## A male canine.

===Verb===
{{en-verb}}

# To follow persistently.
</text>
    </revision>
  </page>
  <page>
    <title>dogs</title>
    <ns>0</ns>
    <id>2</id>
    <revision>
      <id>101</id>
      <text>==English==

===Noun===
{{head|en|noun form}}

# {{plural of|en|dog}}
</text>
    </revision>
  </page>
  <page>
    <title>doggo</title>
    <ns>0</ns>
    <id>3</id>
    <revision>
      <id>102</id>
      <text>==English==

===Noun===
{{en-noun}}

# {{alt form|en|dog}}
</text>
    </revision>
  </page>
  <page>
    <title>red</title>
    <ns>0</ns>
    <id>4</id>
    <revision>
      <id>103</id>
      <text>==English==

===Adjective===
{{en-adj|er}}

# Having red color.
</text>
    </revision>
  </page>
  <page>
    <title>espy</title>
    <ns>0</ns>
    <id>5</id>
    <revision>
      <id>104</id>
      <text>== English ==

===Noun===
{{en-noun}}

# A test of the spaced heading.
</text>
    </revision>
  </page>
  <page>
    <title>Hund</title>
    <ns>0</ns>
    <id>6</id>
    <revision>
      <id>105</id>
      <text>==German==

===Noun===

# a dog
</text>
    </revision>
  </page>
  <page>
    <title>cat</title>
    <ns>0</ns>
    <id>7</id>
    <redirect title="dog"/>
    <revision>
      <id>106</id>
      <text>#REDIRECT [[dog]]</text>
    </revision>
  </page>
  <page>
    <title>Wiktionary:About</title>
    <ns>4</ns>
    <id>8</id>
    <revision>
      <id>107</id>
      <text>==English==
project page
</text>
    </revision>
  </page>
  <page>
    <title>d</title>
    <ns>0</ns>
    <id>9</id>
    <revision>
      <id>108</id>
      <text>==English==

===Letter===
{{head|en|letter}}

# The fourth letter of the alphabet.
</text>
    </revision>
  </page>
  <page>
    <title>booby-trap</title>
    <ns>0</ns>
    <id>10</id>
    <revision>
      <id>109</id>
      <text>==English==

===Verb===
{{en-verb}}

# To rig with a concealed trap.
</text>
    </revision>
  </page>
  <page>
    <title>well-aimed</title>
    <ns>0</ns>
    <id>11</id>
    <revision>
      <id>110</id>
      <text>==English==

===Adjective===
{{en-adj|+first}}

# Aimed with precision.
</text>
    </revision>
  </page>
  <page>
    <title>whiskey</title>
    <ns>0</ns>
    <id>12</id>
    <revision>
      <id>111</id>
      <text>==English==

===Noun===
{{en-noun|ies}}

# A distilled spirit.
</text>
    </revision>
  </page>
  <page>
    <title>atto-candela</title>
    <ns>0</ns>
    <id>13</id>
    <revision>
      <id>112</id>
      <text>{{also|attocandela}}
==English==
{{no entry|en|[[Appendix:SI units]]}}
</text>
    </revision>
  </page>
  <page>
    <title>aamof</title>
    <ns>0</ns>
    <id>14</id>
    <revision>
      <id>113</id>
      <text>==English==

===Prepositional phrase===
{{head|en|prepositional phrase}}

# As a matter of fact.
</text>
    </revision>
  </page>
  <page>
    <title>chessylite</title>
    <ns>0</ns>
    <id>15</id>
    <revision>
      <id>114</id>
      <text>==English==

===Noun===
{{en-noun|-}}

# {{synonym of|en|azurite}}
</text>
    </revision>
  </page>
  <page>
    <title>abc</title>
    <ns>0</ns>
    <id>16</id>
    <revision>
      <id>115</id>
      <text>==English==

===Noun===
{{head|en|noun}}

# {{initialism of|en|[[alpha]] [[beta]] [[crab]]}}
</text>
    </revision>
  </page>
</mediawiki>"#;

fn derive_fixture() -> crate::wikiderive::DumpDerivation {
    let table = CharacterTable::embedded().expect("embedded table");
    let mut reader = PageReader::new(std::io::Cursor::new(FIXTURE.as_bytes()));
    derive_pages(&table, &mut reader, false).expect("derivation runs")
}

#[test]
fn words_assemble_from_titles_and_head_derived_forms() {
    let derivation = derive_fixture();
    let expected = [
        "dog", "dogs", "dogging", "dogged", "doggo", "doggos", "red",
        "redder", "reddest", "espy", "espies", "booby-trap", "booby-traps",
        "booby-trapping", "booby-trapped", "well-aimed", "better-aimed",
        "best-aimed", "whiskey", "whiskies", "aamof", "chessylite", "abc",
    ];
    for word in expected {
        assert!(derivation.words.contains(word), "missing {word}");
    }
    assert_eq!(derivation.words.len(), expected.len());
}

#[test]
fn senses_collect_per_pos_including_subsenses() {
    let derivation = derive_fixture();
    let key = |word: &str, pos: &str| (String::from(word), String::from(pos));
    assert_eq!(
        derivation.definitions[&key("dog", "noun")],
        ["A domestic animal.", "A male canine."]
    );
    assert_eq!(
        derivation.definitions[&key("dog", "verb")],
        ["To follow persistently."]
    );
    assert_eq!(derivation.definitions[&key("dogs", "noun")], ["Plural of [dog]"]);
    assert_eq!(
        derivation.definitions[&key("doggo", "noun")],
        ["Alternative form of [dog]"]
    );
    assert_eq!(derivation.definitions[&key("red", "adjective")], ["Having red color."]);
    assert_eq!(
        derivation.definitions[&key("espy", "noun")],
        ["A test of the spaced heading."]
    );
    assert_eq!(
        derivation.definitions[&key("aamof", "prepositional phrase")],
        ["As a matter of fact."]
    );
    assert_eq!(
        derivation.definitions[&key("chessylite", "noun")],
        ["Synonym of [azurite]"]
    );
    assert_eq!(
        derivation.definitions[&key("abc", "noun")],
        ["Initialism of [alpha] [beta] [crab]"]
    );
    assert_eq!(derivation.definitions.len(), 12);
}

#[test]
fn links_ride_forms_and_form_of_targets() {
    let derivation = derive_fixture();
    let link = |form: &str, lemma: &str| (String::from(form), String::from(lemma));
    for (form, lemma) in [
        ("dogs", "dog"),
        ("dogging", "dog"),
        ("dogged", "dog"),
        ("doggos", "doggo"),
        ("redder", "red"),
        ("reddest", "red"),
        ("espies", "espy"),
        ("booby-traps", "booby-trap"),
        ("booby-trapping", "booby-trap"),
        ("booby-trapped", "booby-trap"),
        ("better-aimed", "well-aimed"),
        ("best-aimed", "well-aimed"),
        ("whiskies", "whiskey"),
        ("doggo", "dog"),
    ] {
        assert!(
            derivation.links.contains(&link(form, lemma)),
            "missing link {form} -> {lemma}"
        );
    }
    // The dogs -> dog pair arrived through the forms first; its
    // plural-of render does not double-count.
    assert_eq!(derivation.tally.links_from_forms, 13);
    assert_eq!(derivation.tally.links_from_form_of, 1);
    // A synonym is a semantic relation, not a form variant, and a
    // multiword wikilinked target is no lemma at all.
    assert!(!derivation
        .links
        .contains(&link("chessylite", "azurite")));
    assert!(!derivation.links.iter().any(|(form, _)| form == "abc"));
}

#[test]
fn page_filters_hold() {
    let derivation = derive_fixture();
    let tally = &derivation.tally;
    assert_eq!(tally.pages_read, 16);
    assert_eq!(tally.ns0_pages, 15);
    assert_eq!(tally.redirect_pages, 1);
    assert_eq!(tally.english_pages, 13);
    // The no-entry soft redirect leaves the vocabulary whole.
    assert_eq!(tally.no_entry_pages, 1);
    assert!(!derivation.words.contains("atto-candela"));
    // The d title is a single code point: the character layer holds
    // that row, so no word, no senses.
    assert_eq!(tally.titles_admitted, 11);
    assert!(!derivation.words.contains("d"));
    assert!(!derivation
        .definitions
        .keys()
        .any(|(word, _)| word == "d" || word == "hund"));
    assert_eq!(tally.parse_failures, 0);
}
