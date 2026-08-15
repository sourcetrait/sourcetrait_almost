//! The wiktionary document renderer: a word's page set to the one
//! markdown document, per the cleaning rules and the microsoft
//! standard.
use crate::*;

use crate::wikitext::normalize_typography;
use crate::wikitext::parse_blocks;
use crate::wikitext::parse_inline_text;
use crate::wikitext::Block;
use crate::wikitext::Emphasis;
use crate::wikitext::Inline;
use crate::wikitext::Template;
use crate::wikixml::WikiPage;

/// One audit row: material the render dropped or could not handle.
/// Nothing unhandled ever leaks into the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AuditRow {
    pub(crate) page: String,
    pub(crate) class: String,
    pub(crate) detail: String,
}

/// A rendered word document plus its audit trail.
pub(crate) struct Document {
    pub(crate) markdown: String,
    pub(crate) audit: Vec<AuditRow>,
}

/// The part-of-speech section names of the English section taxonomy.
const POS_NAMES: [&str; 26] = [
    "Adjective", "Adverb", "Article", "Conjunction", "Contraction",
    "Determiner", "Infix", "Interjection", "Letter", "Noun", "Number",
    "Numeral", "Particle", "Phrase", "Postposition", "Prefix",
    "Preposition", "Prepositional phrase", "Pronoun", "Proper noun",
    "Proverb", "Punctuation mark", "Suffix", "Symbol", "Verb",
    "Interfix",
];

/// Sections dropped whole, audited. Gallery rides the list because it
/// is an image container and images are expected-absent by the
/// criteria; its filename|caption lines are not prose.
const DROP_SECTIONS: [&str; 7] = [
    "Translations", "Descendants", "References", "Further reading",
    "See also", "Statistics", "Gallery",
];

/// POS subsections rendered as anchored lists.
const LIST_SECTIONS: [&str; 12] = [
    "Synonyms", "Antonyms", "Derived terms", "Related terms",
    "Collocations", "Coordinate terms", "Hypernyms", "Hyponyms",
    "Meronyms", "Holonyms", "Troponyms", "Paronyms",
];

/// Third-party data carried in, never hardcoded (TheUser's ruling):
/// English Wiktionary's accent-code vocabulary, vendored under a
/// dump-date version directory and re-derived on a dump bump.
const ACCENTS_DATA: &str = include_str!("../data/accents/20260801/accents.nuon");

/// The parsed accent map: codes whose display differs, and labels
/// that name themselves.
struct AccentTable {
    names: Vec<(String, String)>,
    verbatim: Vec<String>,
}

/// The embedded accent map, parsed once; the vendored file is
/// gate-locked, so a malformed edit fails every accent test.
fn accent_table() -> &'static AccentTable {
    static TABLE: std::sync::OnceLock<AccentTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let value = harness::nu::from_nuon_text(ACCENTS_DATA)
            .expect("the vendored accents.nuon parses");
        let record = value.as_record().expect("accents.nuon is a record");
        let names = record
            .get("names")
            .expect("accents.nuon carries names")
            .as_list()
            .expect("names is a table")
            .iter()
            .map(|row| {
                let row = row.as_record().expect("names row is a record");
                (
                    field_str(row, "code").expect("names row carries code"),
                    field_str(row, "display").expect("names row carries display"),
                )
            })
            .collect();
        let verbatim = record
            .get("verbatim")
            .expect("accents.nuon carries verbatim")
            .as_list()
            .expect("verbatim is a list")
            .iter()
            .map(|item| item.as_str().expect("verbatim code is a string").to_string())
            .collect();
        AccentTable { names, verbatim }
    })
}

// --------------------------------------------------------------------
// The default inflection engine, mirroring Module:en-utilities'
// add_suffix (read from the pinned dump): case-sensitive lowercase
// phonology, y as a vowel, qu (and sometimes gu) collapsing to the
// bare consonant, and doubling read off the final segment.

/// The module's vowel test: y counts as a vowel, and the ligature
/// vowels ride the set (pæan reads vowel-final).
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'æ' | 'ø' | 'œ')
}

/// The module's normalize: lowercase, diacritics stripped (crêpe
/// reads crepe before the phonology), with `qu` (and `gu` when
/// `collapse_gu`) before a vowel collapsing to the bare consonant.
/// `followed` supplies the context character(s) after the word so a
/// word-final `qu` still collapses against its suffix.
fn normalize_stem(word: &str, followed: &str, collapse_gu: bool) -> String {
    use unicode_normalization::char::is_combining_mark;
    use unicode_normalization::UnicodeNormalization;
    let lower: String = format!("{}{}", word.to_lowercase(), followed)
        .nfd()
        .filter(|&c| !is_combining_mark(c))
        .collect();
    let chars: Vec<char> = lower.chars().collect();
    let mut out = String::new();
    let mut index = 0usize;
    while index < chars.len() {
        let c = chars[index];
        let collapsible = c == 'q' || (collapse_gu && c == 'g');
        if collapsible
            && chars.get(index + 1) == Some(&'u')
            && chars.get(index + 2).copied().is_some_and(is_vowel)
        {
            out.push(c);
            index += 2;
            continue;
        }
        out.push(c);
        index += 1;
    }
    for _ in 0..followed.chars().count() {
        out.pop();
    }
    out
}

/// The y-to-i stem for s/d suffixes, or None to keep the word whole.
/// `ey_to_i` enables the -er/-est rules: -eey keeps its e, -ey
/// converts iff the base is polysyllabic (a vowel or digit anywhere:
/// cliquey -> cliqui, grey stays), and gu collapses so roguy -> rogui.
fn final_y_to_i(word: &str, ey_to_i: bool) -> Option<String> {
    let base_y = word.strip_suffix('y')?;
    if base_y.ends_with("ee") {
        return Some(format!("{base_y}i"));
    }
    if ey_to_i && let Some(base) = base_y.strip_suffix('e') {
        if base.ends_with('-') {
            return Some(format!("{base}i"));
        }
        let normalized = normalize_stem(base, "ey", true);
        let polysyllabic = match normalized.chars().last() {
            Some('y') => normalized
                .chars()
                .rev()
                .nth(1)
                .is_some_and(|c| c.is_alphanumeric()),
            _ => normalized
                .chars()
                .any(|c| is_vowel(c) || c.is_ascii_digit()),
        };
        return if polysyllabic { Some(format!("{base}i")) } else { None };
    }
    if base_y.ends_with('-') {
        return Some(format!("{base_y}i"));
    }
    let normalized = normalize_stem(base_y, "y", ey_to_i);
    match normalized.chars().last() {
        Some(c)
            if !is_vowel(c)
                && !c.is_whitespace()
                && !c.is_ascii_punctuation() =>
        {
            Some(format!("{base_y}i"))
        }
        _ => None,
    }
}

/// Whether the plural/3sg takes epenthetic -es (the module's
/// sibilant set, case-sensitive): s/x/z, [csz]h+, dg, consonant-j,
/// and a word-initial u (double-u -> double-ues).
fn takes_es(word: &str) -> bool {
    if word.ends_with('s')
        || word.ends_with('x')
        || word.ends_with('z')
        || word.ends_with('ß')
    {
        return true;
    }
    if word.ends_with('h') {
        let trimmed = word.trim_end_matches('h');
        return trimmed.ends_with('c')
            || trimmed.ends_with('s')
            || trimmed.ends_with('z');
    }
    if word.ends_with('g') && word.len() >= 2 && word.as_bytes()[word.len() - 2] == b'd'
    {
        return true;
    }
    let mut rev = word.chars().rev();
    if rev.next() == Some('j') {
        return rev.next().is_some_and(|c| c.is_alphabetic() && !is_vowel(c));
    }
    if word == "u" {
        return true;
    }
    if word.ends_with('u') {
        let before = word.chars().rev().nth(1);
        return before.is_some_and(|c| !c.is_alphanumeric() && c != '\'');
    }
    false
}

/// The module's double_final_consonant shape: the raw final char is
/// a lowercase doubling consonant, and the normalized stem's final
/// segment (after a space or hyphen) reads [initial][vowel] where
/// initial is empty, exactly y, or a vowel-free run of letters and
/// punctuation. An uppercase segment never doubles.
fn doubles_final(word: &str) -> bool {
    let Some(last) = word.chars().last() else { return false };
    if !matches!(
        last,
        'b' | 'c' | 'd' | 'f' | 'g' | 'j' | 'k' | 'l' | 'm' | 'n' | 'p' | 'q'
            | 'r' | 's' | 't' | 'v' | 'z'
    ) {
        return false;
    }
    let stem = &word[..word.len() - last.len_utf8()];
    let raw_segment = stem.rsplit(['-', ' ']).next().unwrap_or(stem);
    if raw_segment.chars().any(|c| c.is_uppercase()) {
        return false;
    }
    let normalized = normalize_stem(stem, &last.to_string(), true);
    let segment = normalized
        .rsplit(['-', ' '])
        .next()
        .unwrap_or(normalized.as_str());
    let chars: Vec<char> = segment.chars().collect();
    let Some((&vowel, head)) = chars.split_last() else { return false };
    if !is_vowel(vowel) {
        return false;
    }
    if head.is_empty() || head == ['y'] {
        return true;
    }
    head.iter().all(|&c| {
        (c.is_alphabetic() || c.is_ascii_punctuation()) && !is_vowel(c)
    })
}

/// The word with its final consonant doubled before a suffix.
fn doubled(word: &str, suffix: &str) -> String {
    match word.chars().last() {
        Some(last) => format!("{word}{last}{suffix}"),
        None => String::from(suffix),
    }
}

/// The s-form engine behind the plural and the third-person
/// singular: y-to-i takes -es, the sibilant set takes -es (the verb
/// doubling first: quiz -> quizzes), else -s. A proper noun keeps
/// its -y (the Gettys).
fn s_form(word: &str, proper: bool, verb: bool) -> String {
    // The plural of a possessive pluralizes the base and re-adds the
    // possessive (greengrocer's -> greengrocers').
    if !verb && let Some(base) = word.strip_suffix("'s") {
        let plural = s_form(base, proper, false);
        let possessive = if plural.ends_with('s') { "'" } else { "'s" };
        return format!("{plural}{possessive}");
    }
    if !proper && let Some(stem) = final_y_to_i(word, false) {
        return format!("{stem}es");
    }
    if takes_es(word) {
        if verb && doubles_final(word) {
            return doubled(word, "es");
        }
        return format!("{word}es");
    }
    format!("{word}s")
}

/// The regular noun plural (the module's s.plural: no doubling).
pub(crate) fn regular_plural(word: &str) -> String {
    s_form(word, false, false)
}

/// The regular third-person singular (s.verb: doubling on -es).
pub(crate) fn verb_s_form(word: &str) -> String {
    s_form(word, false, true)
}

/// The regular present participle (the module's ing rules): -ie to
/// -ying unless after y, silent e drops after ue or vowel+consonant,
/// the doubling shape doubles, else +ing.
pub(crate) fn regular_participle(word: &str) -> String {
    if let Some(stem) = word.strip_suffix("ie") {
        let before = word.chars().rev().nth(2);
        if before.is_some_and(|c| {
            c != 'y' && c != 'Y' && !c.is_whitespace() && !c.is_ascii_punctuation()
        }) {
            return format!("{stem}ying");
        }
        return format!("{word}ing");
    }
    if let Some(base) = word.strip_suffix('e') {
        let silent = word.ends_with("ue") || {
            let normalized = normalize_stem(base, "e", true);
            let chars: Vec<char> = normalized.chars().collect();
            let trailing_consonants =
                chars.iter().rev().take_while(|&&c| !is_vowel(c)).count();
            trailing_consonants >= 1
                && chars
                    .get(chars.len().wrapping_sub(trailing_consonants + 1))
                    .copied()
                    .is_some_and(is_vowel)
        };
        if silent {
            return format!("{base}ing");
        }
        return format!("{word}ing");
    }
    if doubles_final(word) {
        return doubled(word, "ing");
    }
    format!("{word}ing")
}

/// The regular past (the module's d rules): y-to-i takes -ied, a
/// final e takes -d, the doubling shape doubles, else +ed.
pub(crate) fn regular_past(word: &str) -> String {
    if let Some(stem) = final_y_to_i(word, false) {
        return format!("{stem}ed");
    }
    if word.ends_with('e') {
        return format!("{word}d");
    }
    if doubles_final(word) {
        return doubled(word, "ed");
    }
    format!("{word}ed")
}

/// The irregular graded pairs the module hardcodes; a selector
/// grading "well" must yield better/best, never weller.
fn irregular_graded(word: &str, suffix: &str) -> Option<String> {
    let (comparative, superlative) = match word.to_lowercase().as_str() {
        "well" | "good" => ("better", "best"),
        "bad" | "badly" => ("worse", "worst"),
        "far" => ("further", "furthest"),
        _ => return None,
    };
    Some(String::from(if suffix == "er" { comparative } else { superlative }))
}

/// An -er/-est graded form (the module's r/st.superlative rules):
/// irregulars first, then y/ey-to-i (+ier/+iest), a final e takes
/// the bare -r/-st, the doubling shape doubles, else -er/-est.
fn graded_form(word: &str, suffix: &str) -> String {
    if let Some(irregular) = irregular_graded(word, suffix) {
        return irregular;
    }
    if let Some(stem) = final_y_to_i(word, true) {
        return format!("{stem}{suffix}");
    }
    if word.ends_with('e') {
        let bare = if suffix == "er" { "r" } else { "st" };
        return format!("{word}{bare}");
    }
    if doubles_final(word) {
        return doubled(word, suffix);
    }
    format!("{word}{suffix}")
}

/// One inflected form: its text plus rendered label text, from the
/// en-headword inline-modifier grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SpecForm {
    text: String,
    labels: Vec<String>,
}

/// Embedded wikilinks flattened to their display text.
fn flatten_wikilinks(text: &str) -> String {
    let mut text = text.to_string();
    while let Some(open) = text.find("[[") {
        let Some(close) = text[open..].find("]]").map(|c| open + c) else {
            break;
        };
        let inner = &text[open + 2..close];
        let display = match inner.rsplit_once('|') {
            Some((_, display)) => display,
            None => inner,
        };
        text = format!("{}{display}{}", &text[..open], &text[close + 2..]);
    }
    text
}

impl SpecForm {
    fn plain(text: String) -> Self {
        Self { text, labels: Vec::new() }
    }

    /// The document rendering: an anchored form plus its label
    /// parenthetical. Embedded wikilinks flatten to their display
    /// text - the whole form is the anchor.
    fn rendered(&self) -> String {
        let text = flatten_wikilinks(&self.text);
        if self.labels.is_empty() {
            format!("[{text}]")
        } else {
            format!("[{text}] ({})", self.labels.join(", "))
        }
    }
}

/// Render one label body: `l:`/`q:` class prefixes drop, `ref:`
/// bodies drop whole, comma lists join with the `_` separator
/// suppressor removed.
fn label_text(body: &str) -> Option<String> {
    let value = if let Some(rest) = body.split_once(':') {
        if rest.0 == "ref" {
            return None;
        }
        rest.1
    } else {
        body
    };
    let parts: Vec<&str> = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty() && *part != "_")
        .collect();
    if parts.is_empty() {
        return None;
    }
    // `or` and `and` are label-list connectors, joined inline
    // rather than as list elements.
    let mut out = String::new();
    let mut connector: Option<&str> = None;
    for part in parts {
        if part == "or" || part == "and" {
            connector = Some(part);
            continue;
        }
        if out.is_empty() {
            out.push_str(part);
        } else if let Some(word) = connector.take() {
            out.push_str(&format!(" {word} {part}"));
        } else {
            out.push_str(&format!(", {part}"));
        }
    }
    Some(out.replace("<<", "").replace(">>", ""))
}

/// Split a spec on top-level commas (inline `<...>` modifiers shield
/// theirs), strip each form's modifiers into labels, and substitute
/// `~` with the lemma. Empty pieces drop.
fn spec_forms(spec: &str, title: &str) -> Vec<SpecForm> {
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    for c in spec.chars() {
        match c {
            '<' => {
                depth += 1;
                current.push(c);
            }
            '>' => {
                depth = depth.saturating_sub(1);
                current.push(c);
            }
            ',' if depth == 0 => pieces.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    pieces.push(current);
    pieces
        .into_iter()
        .map(|piece| piece.trim().to_string())
        .filter(|piece| !piece.is_empty())
        .map(|piece| {
            let mut text = piece;
            let mut labels = Vec::new();
            while text.ends_with('>')
                && let Some(open) = text.rfind('<')
            {
                let body = text[open + 1..text.len() - 1].to_string();
                text.truncate(open);
                if let Some(label) = label_text(&body) {
                    labels.insert(0, label);
                }
            }
            // A lone `~` is a marker (noun countability), never a
            // substitution site; anywhere else `~` takes the lemma.
            let text = if text == "~" { text } else { text.replace('~', title) };
            SpecForm { text, labels }
        })
        .collect()
}

/// What a POS section's head template contributes: an optional
/// headword display override and an optional inflection
/// parenthetical.
#[derive(Default)]
pub(crate) struct HeadLine {
    pub(crate) headword: Option<String>,
    pub(crate) parenthetical: Option<String>,
}

/// Whether a template is an English headword-line template.
fn is_head_family(name: &str) -> bool {
    name == "head" || name.starts_with("en-")
}

/// A named key's base with trailing digits split off ("past2" to
/// "past"); an all-digit key stays whole.
fn key_base(key: &str) -> &str {
    let trimmed = key.trim_end_matches(|c: char| c.is_ascii_digit());
    if trimmed.is_empty() { key } else { trimmed }
}

/// A named list parameter's values: the base key plus its numbered
/// variants, in numeric order (meaning, meaning2, ...).
fn named_family(template: &Template, base: &str) -> Vec<String> {
    let mut rows: Vec<(usize, String)> = Vec::new();
    for (key, value) in &template.named {
        if key_base(key) == base && !value.trim().is_empty() {
            let number = key[base.len()..].parse::<usize>().unwrap_or(1);
            rows.push((number, value.clone()));
        }
    }
    rows.sort_by_key(|(number, _)| *number);
    rows.into_iter().map(|(_, value)| value).collect()
}

/// Split a value on commas outside `<...>` modifier blocks (the
/// inline-modifier convention shared by the name and form-of
/// families).
fn split_modifier_commas(value: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    for c in value.chars() {
        match c {
            '<' => {
                depth += 1;
                current.push(c);
            }
            '>' => {
                depth = depth.saturating_sub(1);
                current.push(c);
            }
            ',' if depth == 0 => parts.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current);
    }
    parts
}

/// Join resolved tag displays with the documented punctuation
/// spacing: a closer attaches left, an opener attaches right, a
/// slash or hyphen attaches both sides.
fn join_tags(tokens: &[String]) -> String {
    let mut out = String::new();
    let mut suppress_space = true;
    for token in tokens {
        match token.as_str() {
            "," | ")" | "]" | ":" => {
                out.push_str(token);
                suppress_space = false;
            }
            "(" | "[" => {
                if !suppress_space {
                    out.push(' ');
                }
                out.push_str(token);
                suppress_space = true;
            }
            "/" | "-" => {
                out.push_str(token);
                suppress_space = true;
            }
            _ => {
                if !suppress_space {
                    out.push(' ');
                }
                out.push_str(token);
                suppress_space = false;
            }
        }
    }
    out
}

/// Fold numeric named arguments into their positional slots - the
/// MediaWiki |1=x| equivalence - so the engines see one shape.
fn normalize_numbered(template: &Template) -> Template {
    let mut normalized = Template {
        name: template.name.clone(),
        positional: template.positional.clone(),
        named: Vec::new(),
    };
    for (key, value) in &template.named {
        if let Ok(position) = key.parse::<usize>()
            && position >= 1
        {
            while normalized.positional.len() < position {
                normalized.positional.push(String::new());
            }
            normalized.positional[position - 1] = value.clone();
            continue;
        }
        normalized.named.push((key.clone(), value.clone()));
    }
    normalized
}

/// A template's compact call signature for the audit: name, then
/// positionals, then named args, pipe-joined and capped - the head
/// families grow audit-driven, and that needs the arguments, not
/// just the name.
fn template_signature(template: &Template) -> String {
    let mut out = template.name.clone();
    for positional in &template.positional {
        out.push('|');
        out.push_str(positional);
    }
    for (key, value) in &template.named {
        out.push('|');
        out.push_str(key);
        out.push('=');
        out.push_str(value);
    }
    if out.chars().count() > 200 {
        let mut capped: String = out.chars().take(200).collect();
        capped.push_str("...");
        return capped;
    }
    out
}

/// Third-party data carried in: the form-of template alias
/// vocabulary, vendored under a dump-date version directory.
const FORM_OF_DATA: &str = include_str!("../data/form_of/20260801/form_of.nuon");

/// The parsed form-of aliases, loaded once like the accent map.
fn form_of_aliases() -> &'static Vec<(String, String)> {
    static TABLE: std::sync::OnceLock<Vec<(String, String)>> =
        std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let value = harness::nu::from_nuon_text(FORM_OF_DATA)
            .expect("the vendored form_of.nuon parses");
        let record = value.as_record().expect("form_of.nuon is a record");
        record
            .get("aliases")
            .expect("form_of.nuon carries aliases")
            .as_list()
            .expect("aliases is a table")
            .iter()
            .map(|row| {
                let row = row.as_record().expect("aliases row is a record");
                (
                    field_str(row, "name").expect("aliases row carries name"),
                    field_str(row, "full").expect("aliases row carries full"),
                )
            })
            .collect()
    })
}

/// " of"-shaped templates whose relation is semantic rather than a
/// form variant: they render through the shape rule but their
/// targets are not inflection-link lemmas.
const SEMANTIC_OF_LABELS: [&str; 10] = [
    "synonym of", "antonym of", "hypernym of", "hyponym of",
    "cohyponym of", "coordinate of", "homophone of",
    "female equivalent of", "male equivalent of", "gender equivalent of",
];

/// The full "<label> of" name a definitional form-of template
/// renders under: an alias expansion, or the name itself when it
/// already ends in " of" (the generic shape; `form of` itself
/// carries its label as an argument and is handled separately).
/// The en- prefixed family is excluded: those templates bake their
/// language, so the lemma sits one slot earlier and the generic
/// shape mangles them - the handled pair has its own arm and the
/// rest audit.
fn form_of_name(name: &str) -> Option<String> {
    if let Some((_, full)) =
        form_of_aliases().iter().find(|(alias, _)| alias == name)
    {
        return Some(full.clone());
    }
    if name.ends_with(" of") && name != "form of" && !name.starts_with("en-") {
        return Some(name.to_string());
    }
    None
}

/// Templates that carry no document meaning anywhere they appear:
/// categorization, sense ids, maintenance requests, and data-only
/// carriers. They render nothing and file no audit row.
const SILENT_TEMPLATES: [&str; 22] = [
    "C", "c", "topics", "cln", "catlangname", "senseid", "sid", "defdate",
    "rfe", "rfd", "rfv", "rfdef", "rfex", "rfquote", "attention", "anchor",
    "etystub", "dercat", "etymid", "root", "wikidata lexeme", "rfc",
];

/// Whether a template renders to nothing by design (no audit).
fn is_silent_template(name: &str) -> bool {
    SILENT_TEMPLATES.contains(&name)
}

/// Anchored text flattened to plain display: `[d](t)` and `[x]`
/// lose their markup - headword lines carry no anchors.
fn plain_anchor_text(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        out.push_str(&rest[..open]);
        let Some(close) = rest[open..].find(']').map(|c| open + c) else {
            out.push_str(&rest[open..]);
            return out;
        };
        out.push_str(&rest[open + 1..close]);
        rest = &rest[close + 1..];
        if rest.starts_with('(')
            && let Some(paren) = rest.find(')')
        {
            rest = &rest[paren + 1..];
        }
    }
    out.push_str(rest);
    out
}

/// An anchor per the display/resource rule: bare when the display IS
/// the resource name, display form otherwise. Empty inputs produce
/// no anchor - "[]" is damage, never output.
fn anchor(display: &str, target: &str) -> String {
    if display.is_empty() && target.is_empty() {
        return String::new();
    }
    if display.is_empty() {
        return format!("[{target}]");
    }
    if display == target || target.is_empty() {
        format!("[{display}]")
    } else {
        format!("[{display}]({target})")
    }
}

/// The renderer over one page's English subtree. Beside the markdown
/// it collects the derived DATA the render presents - inflected
/// forms, per-POS senses, and form-of lemma targets - so the
/// provenance derivation reads exactly what the document shows.
struct Renderer<'a> {
    page_title: &'a str,
    lines: Vec<String>,
    audit: Vec<AuditRow>,
    /// Every inflected form the head engines put in a parenthetical.
    forms: Vec<String>,
    /// (pos section name, rendered gloss) per numbered sense line.
    senses: Vec<(String, String)>,
    /// The lemma targets of definitional form-of renders.
    form_of_lemmas: Vec<String>,
}

impl<'a> Renderer<'a> {
    fn new(page_title: &'a str) -> Self {
        Self {
            page_title,
            lines: Vec::new(),
            audit: Vec::new(),
            forms: Vec::new(),
            senses: Vec::new(),
            form_of_lemmas: Vec::new(),
        }
    }

    /// One document line's text cleaned for the line-structured
    /// artifacts: any embedded newline flattens to a space, and any
    /// character with no vocabulary row (unassigned, or the excluded
    /// private-use class) strips - both audited. Raw template
    /// positionals legally span lines and several render paths take
    /// them without the inline renderer, so the guard sits here, the
    /// one point every pushed line shares. The corpus is controlled:
    /// content the tokenizer refuses by design never enters it.
    fn flat_line(&mut self, text: String) -> String {
        let mut text = text;
        if text.contains('\n') || text.contains('\r') {
            text = text.replace("\r\n", " ").replace(['\n', '\r'], " ");
            let sample: String = text.chars().take(120).collect();
            self.audit("line_embedded_newline", sample);
        }
        if !text.is_ascii() {
            let table = crate::ucd::embedded_table();
            if text.chars().any(|c| table.index_of(c as u32).is_none()) {
                let mut dropped: Vec<String> = text
                    .chars()
                    .filter(|c| table.index_of(*c as u32).is_none())
                    .map(|c| format!("U+{:04X}", c as u32))
                    .collect();
                dropped.dedup();
                dropped.truncate(16);
                text = text
                    .chars()
                    .filter(|c| table.index_of(*c as u32).is_some())
                    .collect();
                self.audit("character_unlexable", dropped.join(" "));
            }
        }
        text
    }

    /// One prose paragraph: a blank separator, then the line.
    fn push_prose(&mut self, text: String) {
        let text = self.flat_line(text);
        self.blank();
        self.lines.push(text);
    }

    /// One list item at a nesting depth, as a dash line.
    fn push_item(&mut self, depth: usize, text: String) {
        let text = self.flat_line(text);
        self.lines
            .push(format!("{}- {text}", "  ".repeat(depth)));
    }

    /// One numbered sense line at its 1-based depth.
    fn push_sense(&mut self, depth: usize, number: usize, gloss: String) {
        let gloss = self.flat_line(gloss);
        let indent = "   ".repeat(depth - 1);
        self.lines.push(format!("{indent}{number}. {gloss}"));
    }

    /// One quotation or usage-example line under the last sense.
    fn push_quote(&mut self, depth: usize, text: String) {
        let text = self.flat_line(text);
        let indent = "   ".repeat(depth);
        self.lines.push(format!("{indent}- {text}"));
    }

    /// A POS section's word line at its output level.
    fn push_head_word(&mut self, level: usize, form: String) {
        let form = self.flat_line(form);
        self.lines
            .push(format!("{} {form}", "#".repeat(level)));
    }

    /// The word's parenthetical line plus its separator.
    fn push_head_line(&mut self, text: String) {
        let text = self.flat_line(text);
        self.lines.push(text);
        self.lines.push(String::new());
    }

    /// Record one presented inflected form, wikilinks flattened.
    fn record_form(&mut self, text: &str) {
        let flat = flatten_wikilinks(text);
        let flat = flat.trim();
        if !flat.is_empty() && !flat.contains("{{") {
            self.forms.push(flat.to_string());
        }
    }

    /// Record a definitional form-of render's lemma target: inline
    /// modifiers strip, every wikilink flattens to its TARGET (the
    /// resource is the lemma), a section suffix strips; w:-referents
    /// and template-bearing targets are not lemmas. A multiword
    /// result survives here and drops at the admission filter -
    /// truncating it to its first word would fabricate a link.
    fn record_form_of(&mut self, raw: &str) {
        let (base, _) = term_modifiers(raw);
        let base = base.trim();
        if base.is_empty() || base.starts_with("w:") || base.contains("{{") {
            return;
        }
        let mut text = String::new();
        let mut rest = base;
        while let Some(open) = rest.find("[[") {
            text.push_str(&rest[..open]);
            let Some(close) = rest[open..].find("]]").map(|c| open + c) else {
                text.push_str(&rest[open..]);
                rest = "";
                break;
            };
            let inner = &rest[open + 2..close];
            let target = match inner.split_once('|') {
                Some((target, _)) => target,
                None => inner,
            };
            text.push_str(target);
            rest = &rest[close + 2..];
        }
        text.push_str(rest);
        let text = text.split('#').next().unwrap_or_default().trim();
        if !text.is_empty() {
            self.form_of_lemmas.push(text.to_string());
        }
    }

    /// Details sanitize to single NUON-line-safe lines here, the one
    /// point every consumer shares: raw wikitext rides into details
    /// (a link target legally spans lines), and the audit files are
    /// NUON-lines artifacts.
    fn audit(&mut self, class: &str, detail: String) {
        self.audit.push(AuditRow {
            page: self.page_title.to_string(),
            class: class.to_string(),
            detail: crate::associations::sanitize_gloss(&detail),
        });
    }

    /// A blank separator, never doubled, never leading.
    fn blank(&mut self) {
        if matches!(self.lines.last(), Some(last) if !last.is_empty()) {
            self.lines.push(String::new());
        }
    }

    fn heading(&mut self, level: usize, text: &str) {
        let text = self.flat_line(text.to_string());
        self.blank();
        self.lines.push(format!("{} {text}", "#".repeat(level)));
    }

    /// Render inlines to markdown text: anchors, emphasis, templates
    /// by family, typography normalized. Anchor targets stay verbatim
    /// (resource addresses); only prose text normalizes.
    fn inline_text(&mut self, inlines: &[Inline]) -> String {
        let mut out = String::new();
        for inline in inlines {
            match inline {
                Inline::Text(text) => out.push_str(&normalize_typography(text)),
                Inline::Nowiki(text) => out.push_str(&normalize_typography(text)),
                // Emphasis is reserved (TheUser's cleanup is the
                // ruling): source italics flatten - the renderer's
                // own citation titles are the only italics - and
                // source bold survives (the quote target).
                Inline::Emphasis(emphasis) => out.push_str(match emphasis {
                    Emphasis::Italic => "",
                    Emphasis::Bold | Emphasis::BoldItalic => "**",
                }),
                Inline::Link(link) => {
                    let target = link.target.trim();
                    if let Some(rest) = target.strip_prefix("w:") {
                        let display = match link.display.as_deref() {
                            Some(raw) if raw.contains("{{") => {
                                let raw = raw.to_string();
                                self.argument_text(&raw)
                            }
                            Some(raw) => raw.to_string(),
                            None => rest.to_string(),
                        };
                        out.push_str(&anchor(&display, rest));
                        continue;
                    }
                    // A same-page section link carries no
                    // cross-reference: its display text stands alone.
                    if let Some(section) = target.strip_prefix('#') {
                        let display = link.display.as_deref().unwrap_or(section);
                        out.push_str(&normalize_typography(display));
                        out.push_str(&link.trail);
                        continue;
                    }
                    if target.contains(':') {
                        self.audit("link_dropped", target.to_string());
                        continue;
                    }
                    // A section suffix addresses within the page; the
                    // page name is the resource. A display half can
                    // itself carry templates - re-render, never leak.
                    let page = match target.split_once('#') {
                        Some((page, _)) if !page.is_empty() => page,
                        _ => target,
                    };
                    let base = match link.display.as_deref() {
                        Some(raw) if raw.contains("{{") => {
                            let raw = raw.to_string();
                            self.argument_text(&raw)
                        }
                        Some(raw) => raw.to_string(),
                        None => page.to_string(),
                    };
                    let display = format!("{base}{}", link.trail);
                    out.push_str(&anchor(&display, page));
                }
                Inline::ExternalLink { url, label } => {
                    // A labeled external link keeps its label as
                    // plain prose (the printed form); a bare URL
                    // drops. The label-less drop inside literal
                    // brackets was the "[]" damage class.
                    match label {
                        Some(label) => out.push_str(&normalize_typography(label)),
                        None => self.audit("external_link_dropped", url.clone()),
                    }
                }
                Inline::Template(template) => {
                    let rendered = self.template_text(template);
                    out.push_str(&rendered);
                }
                Inline::Ref { attrs, .. } => {
                    self.audit("ref_dropped", attrs.clone());
                }
                Inline::Html { tag, .. } => {
                    if tag == "br" {
                        out.push(' ');
                    } else {
                        self.audit("html_tag_dropped", tag.clone());
                    }
                }
            }
        }
        // A template argument legally spans lines; rendered into one
        // document line an embedded newline would split into a raw
        // pseudo-line (a wikitext `#` lands as a false heading). HTML
        // renders such newlines as spaces - so does this, audited so
        // the class stays measurable (MediaWiki expands templates
        // before block parsing; this parse-first pipeline cannot, so
        // a template spanning list lines glues them).
        if out.contains('\n') || out.contains('\r') {
            out = out.replace("\r\n", " ").replace(['\n', '\r'], " ");
            let sample: String = out.chars().take(120).collect();
            self.audit("inline_embedded_newline", sample);
        }
        out.trim().to_string()
    }

    /// Re-parse and render a raw argument's wikitext.
    fn argument_text(&mut self, raw: &str) -> String {
        match parse_inline_text(raw) {
            Ok(inlines) => self.inline_text(&inlines),
            Err(_) => {
                self.audit("argument_unparsed", raw.to_string());
                normalize_typography(raw)
            }
        }
    }

    /// An anchored word argument: text carrying links or templates
    /// renders as it is; a bare word bare-anchors.
    fn anchored_argument(&mut self, raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.contains("[[") || trimmed.contains("{{") {
            self.argument_text(trimmed)
        } else if trimmed.is_empty() {
            String::new()
        } else {
            format!("[{trimmed}]")
        }
    }

    /// The template families, per the criteria: a handled family
    /// renders its document meaning; anything else contributes
    /// nothing and files an audit row.
    fn template_text(&mut self, template: &Template) -> String {
        let name = template.name.as_str();
        let positional: Vec<&str> =
            template.positional.iter().map(String::as_str).collect();
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        match name {
            "lb" | "lbl" | "label" => {
                // A label can itself carry a template or link (the
                // leak class); rendered ones pass through the inline
                // renderer, plain vocabulary stays verbatim.
                let qualifiers: Vec<String> = positional
                    .iter()
                    .skip(1)
                    .filter(|q| !q.is_empty() && **q != "_")
                    .map(|q| {
                        if q.contains("{{") || q.contains("[[") {
                            self.argument_text(q)
                        } else {
                            q.to_string()
                        }
                    })
                    .collect();
                if qualifiers.is_empty() {
                    String::new()
                } else {
                    format!("({})", qualifiers.join(", "))
                }
            }
            "q" | "qualifier" | "i" | "gloss" | "gl" => {
                let parts: Vec<String> = positional
                    .iter()
                    .filter(|part| !part.is_empty())
                    .map(|part| self.argument_text(part))
                    .collect();
                if parts.is_empty() {
                    String::new()
                } else {
                    format!("({})", parts.join(", "))
                }
            }
            "w" => {
                let target = positional.first().copied().unwrap_or_default();
                let display = positional.get(1).copied().unwrap_or(target);
                anchor(display, target)
            }
            "l" | "m" | "ll" => {
                let word = positional.get(1).copied().unwrap_or_default();
                let display = positional
                    .get(2)
                    .filter(|d| !d.is_empty())
                    .copied()
                    .unwrap_or(word);
                anchor(display, word)
            }
            "alter" | "alt" => {
                // Words after the language, stopping at the empty
                // positional that separates words from qualifiers.
                let words: Vec<String> = positional
                    .iter()
                    .skip(1)
                    .take_while(|part| !part.is_empty())
                    .map(|part| format!("[{part}]"))
                    .collect();
                words.join(", ")
            }
            "blend" => {
                let head = if named("nocap").is_some() { "blend of" } else { "Blend of" };
                let parts = self.plus_joined(positional.get(1..).unwrap_or(&[]));
                format!("{head} {parts}")
            }
            "suffix" | "suf" => {
                let base = positional.get(1).copied().unwrap_or_default();
                let suffix = positional.get(2).copied().unwrap_or_default();
                let suffix = if suffix.starts_with('-') {
                    suffix.to_string()
                } else {
                    format!("-{suffix}")
                };
                format!("{} + [{suffix}]", self.anchored_argument(base))
            }
            "prefix" | "pre" => {
                let prefix = positional.get(1).copied().unwrap_or_default();
                let base = positional.get(2).copied().unwrap_or_default();
                let prefix = if prefix.ends_with('-') {
                    prefix.to_string()
                } else {
                    format!("{prefix}-")
                };
                format!("[{prefix}] + {}", self.anchored_argument(base))
            }
            "affix" | "af" | "compound" | "com" => {
                self.plus_joined(positional.get(1..).unwrap_or(&[]))
            }
            "confix" => {
                // A confix is prefix + ... + suffix: the first part
                // hyphenates trailing, the last leading.
                let parts: Vec<&str> = positional
                    .get(1..)
                    .unwrap_or(&[])
                    .iter()
                    .filter(|part| !part.is_empty())
                    .copied()
                    .collect();
                let last = parts.len().saturating_sub(1);
                let rendered: Vec<String> = parts
                    .iter()
                    .enumerate()
                    .map(|(index, part)| {
                        let affixed = if index == 0 && !part.ends_with('-') {
                            format!("{part}-")
                        } else if index == last && !part.starts_with('-') {
                            format!("-{part}")
                        } else {
                            (*part).to_string()
                        };
                        self.anchored_argument(&affixed)
                    })
                    .collect();
                rendered.join(" + ")
            }
            // The page-title magic word, not a template.
            "PAGENAME" => self.page_title.to_string(),
            "short for" => self.multi_term_form_of(template, "Short for"),
            "only used in" | "only in" => {
                self.multi_term_form_of(template, "Only used in")
            }
            "&lit" => {
                let terms: Vec<String> = positional
                    .iter()
                    .skip(1)
                    .filter(|part| !part.is_empty())
                    .map(|part| self.anchored_argument(part))
                    .collect();
                format!(
                    "Used other than figuratively or idiomatically: see {}.",
                    terms.join(", ")
                )
            }
            "demonym-noun" | "demonym-adj" => {
                self.demonym_text(template, name == "demonym-noun")
            }
            "SI-unit" => self.si_unit_text(template),
            "staco" | "station code" => {
                let article = positional.first().copied().unwrap_or_default();
                let display = positional
                    .get(1)
                    .filter(|part| !part.is_empty())
                    .copied()
                    .unwrap_or(article);
                let place = positional.get(2).copied().unwrap_or_default();
                if article.is_empty() {
                    self.audit("template_unhandled", template_signature(template));
                    String::new()
                } else {
                    let place = self.argument_text(place);
                    let tail = if place.is_empty() {
                        String::from(".")
                    } else {
                        format!(" in {place}.")
                    };
                    format!(
                        "(rail transport) The station code of {}{tail}",
                        anchor(display, article)
                    )
                }
            }
            // A transcluded definition lives on another page; the
            // honest single-page rendering is the cross-reference.
            "tcl" | "transclude" | "transclude sense" => {
                let target = positional.get(1).copied().unwrap_or_default();
                if target.is_empty() {
                    self.audit("template_unhandled", template_signature(template));
                    String::new()
                } else {
                    format!("See {}.", self.anchored_argument(target))
                }
            }
            "name translit" => self.name_translit_text(template, "transliteration"),
            "name respelling" => self.name_translit_text(template, "respelling"),
            "name obor" => {
                self.name_translit_text(template, "orthographic borrowing")
            }
            // Pure metadata: categories, sense ids, dates, and
            // maintenance stubs carry no document meaning.
            name if is_silent_template(name) => String::new(),
            // The etymology reference family: language name plus
            // anchored term, complete-wording variants prefixed.
            "der" | "derived" | "uder" | "undefined derivation" => {
                self.ety_reference(template, 1, "")
            }
            "bor" | "borrowed" | "inh" | "inherited" => self.ety_reference(template, 1, ""),
            "der+" => self.ety_reference(template, 1, "Derived from "),
            "bor+" => self.ety_reference(template, 1, "Borrowed from "),
            "inh+" => self.ety_reference(template, 1, "Inherited from "),
            "lbor" | "learned borrowing" => {
                self.ety_reference(template, 1, "Learned borrowing from ")
            }
            "ubor" | "unadapted borrowing" => {
                self.ety_reference(template, 1, "Unadapted borrowing from ")
            }
            "slbor" | "semi-learned borrowing" => {
                self.ety_reference(template, 1, "Semi-learned borrowing from ")
            }
            "obor" | "orthographic borrowing" => {
                self.ety_reference(template, 1, "Orthographic borrowing from ")
            }
            "semantic loan" => self.ety_reference(template, 1, "Semantic loan from "),
            "calque" | "cal" | "clq" => self.ety_reference(template, 1, "Calque of "),
            "partial calque" | "pcal" => {
                self.ety_reference(template, 1, "Partial calque of ")
            }
            "cog" | "cognate" | "noncog" | "noncognate" | "ncog" | "m+" => {
                self.ety_reference(template, 0, "")
            }
            "doublet" | "dbt" => self.doublet_text(template, "Doublet of "),
            "piecewise doublet" | "pw dbt" | "pwdbt" | "pwd" => {
                self.doublet_text(template, "Piecewise doublet of ")
            }
            "unk" | "unknown" => self.origin_statement(template, "Unknown"),
            "unc" | "uncertain" => self.origin_statement(template, "Uncertain"),
            "back-form" | "back-formation" | "bf" => {
                let flag = |key: &str| {
                    template.named.iter().any(|(name, _)| name == key)
                };
                let term_text = match template.positional.get(1) {
                    Some(term) if !term.is_empty() && term != "-" => {
                        let term = term.clone();
                        let alt = template.positional.get(2).cloned().unwrap_or_default();
                        self.ety_term_anchor(&term, &alt)
                    }
                    _ => String::new(),
                };
                if flag("notext") {
                    return term_text;
                }
                let text = if term_text.is_empty() {
                    String::from("Back-formation")
                } else {
                    format!("Back-formation from {term_text}")
                };
                if flag("nocap") { lcfirst(&text) } else { text }
            }
            "surf" | "surface analysis" | "surface etymology" => {
                if template
                    .positional
                    .get(1)
                    .is_some_and(|part| part.starts_with('+'))
                {
                    self.audit("template_unhandled", template_signature(template));
                    return String::new();
                }
                let parts: Vec<String> = template
                    .positional
                    .iter()
                    .skip(1)
                    .filter(|part| !part.trim().is_empty())
                    .map(|part| term_modifiers(part).0)
                    .collect();
                let joined: Vec<String> = parts
                    .iter()
                    .map(|part| self.anchored_argument(part))
                    .collect();
                let named = |key: &str| {
                    template.named.iter().any(|(name, _)| name == key)
                };
                let text = format!("By surface analysis, {}", joined.join(" + "));
                if named("nocap") { lcfirst(&text) } else { text }
            }
            "etymon" | "ety" => self.etymon_text(template),
            "inflection of" | "infl of" | "noun form of" | "verb form of"
            | "adj form of" => self.inflection_of_text(template),
            "surname" => self.surname_text(template),
            "given name" => self.given_name_text(template),
            "place" => self.place_text(template),
            "sense" | "s" => {
                let parts: Vec<String> = positional
                    .iter()
                    .filter(|part| !part.is_empty())
                    .map(|part| self.argument_text(part))
                    .collect();
                if parts.is_empty() {
                    String::new()
                } else {
                    format!("({}):", parts.join(", "))
                }
            }
            "taxfmt" => {
                let taxon = positional.first().copied().unwrap_or_default();
                let display = positional
                    .get(2)
                    .filter(|part| !part.is_empty())
                    .copied()
                    .unwrap_or(taxon);
                if taxon.is_empty() {
                    String::new()
                } else {
                    anchor(display, taxon)
                }
            }
            "taxlink" => {
                let taxon = positional.first().copied().unwrap_or_default();
                let display = positional
                    .get(2)
                    .filter(|part| !part.is_empty())
                    .copied()
                    .unwrap_or(taxon);
                normalize_typography(display)
            }
            "vern" => {
                let name = positional.first().copied().unwrap_or_default();
                let display = positional
                    .get(1)
                    .filter(|part| !part.is_empty())
                    .copied()
                    .unwrap_or(name);
                let plural = named("pl").unwrap_or_default();
                if name.is_empty() {
                    String::new()
                } else {
                    anchor(&format!("{display}{plural}"), name)
                }
            }
            "glossary" | "lg" => {
                let term = positional.first().copied().unwrap_or_default();
                let display = positional
                    .get(1)
                    .filter(|part| !part.is_empty())
                    .copied()
                    .unwrap_or(term);
                self.argument_text(display)
            }
            "cap" | "U" => {
                let term = positional.first().copied().unwrap_or_default();
                if term.is_empty() {
                    String::new()
                } else {
                    anchor(&ucfirst(term), term)
                }
            }
            "..." | "nb..." => String::from("..."),
            "," => {
                if positional.first() == Some(&"and") {
                    String::from(", and")
                } else {
                    String::from(",")
                }
            }
            "'" => String::from("'"),
            "sic" | "SIC" => String::from("(sic)"),
            "smallcaps" | "smc" | "sup" | "sub" => {
                let text = positional.first().copied().unwrap_or_default();
                self.argument_text(text)
            }
            "IPAchar" => {
                let parts: Vec<String> = positional
                    .iter()
                    .filter(|part| !part.is_empty())
                    .map(|part| normalize_typography(part))
                    .collect();
                parts.join(", ")
            }
            "lang" => {
                let text = positional.get(1).copied().unwrap_or_default();
                self.argument_text(text)
            }
            "non-gloss" | "non-gloss definition" | "n-g" | "ng" | "ngd" => {
                let text = positional.first().copied().unwrap_or_default();
                self.argument_text(text)
            }
            "ux" | "uxi" | "usex" => {
                let text = positional.get(1).copied().unwrap_or_default();
                let rendered = self.argument_text(text);
                if rendered.is_empty() {
                    String::new()
                } else {
                    format!("\"{rendered}\"")
                }
            }
            name if name.starts_with("quote-") => {
                self.citation_text(template).unwrap_or_default()
            }
            // The English degree pair bakes lang=en, so the lemma is
            // the first positional; each renders its own "<degree>
            // <lemma>" tail, the template's own output shape.
            "en-superlative of" | "en-comparative of" => {
                let term = positional.first().copied().unwrap_or_default();
                if term.is_empty() {
                    self.audit("template_unhandled", template_signature(template));
                    return String::new();
                }
                let alt = positional
                    .get(1)
                    .filter(|d| !d.is_empty())
                    .copied()
                    .unwrap_or(term);
                self.record_form_of(term);
                let anchored = self.form_of_anchor(term, alt);
                let (label, degree) = if name == "en-superlative of" {
                    ("Superlative form of", "most")
                } else {
                    ("Comparative form of", "more")
                };
                format!("{label} {anchored}: {degree} {anchored}")
            }
            "form of" => {
                let label = positional.get(1).copied().unwrap_or_default();
                let target = positional.get(2).copied().unwrap_or_default();
                let display = positional
                    .get(3)
                    .filter(|d| !d.is_empty())
                    .copied()
                    .unwrap_or(target);
                self.record_form_of(target);
                let anchored = self.form_of_anchor(target, display);
                format!("{} of {}", ucfirst(label), anchored)
            }
            name if form_of_name(name).is_some() => {
                let label = form_of_name(name).expect("checked above");
                let target = positional.get(1).copied().unwrap_or_default();
                let display = positional
                    .get(2)
                    .filter(|d| !d.is_empty())
                    .copied()
                    .unwrap_or(target);
                if !SEMANTIC_OF_LABELS.contains(&label.as_str()) {
                    self.record_form_of(target);
                }
                let anchored = self.form_of_anchor(target, display);
                let mut text = format!("{label} {anchored}");
                if named("nocap").is_none() {
                    text = ucfirst(&text);
                }
                if let Some(gloss) = named("t") {
                    let gloss = self.argument_text(gloss);
                    if !gloss.is_empty() {
                        text.push_str(&format!(" ({gloss})"));
                    }
                }
                text
            }
            _ => {
                self.audit("template_unhandled", template.name.clone());
                String::new()
            }
        }
    }

    /// A form-of target's anchor: embedded markup renders as it is,
    /// a w: target strips to its page name, a plain pair anchors.
    fn form_of_anchor(&mut self, target: &str, display: &str) -> String {
        if target.contains("[[") || target.contains("{{") {
            return self.argument_text(target);
        }
        if let Some(rest) = target.strip_prefix("w:") {
            let display = display.strip_prefix("w:").unwrap_or(display);
            return anchor(display, rest);
        }
        anchor(display, target)
    }

    /// Record a name-list value's terms (the surname and given-name
    /// variant families) as form-of lemmas: comma pieces, English or
    /// bare terms only, inline modifiers stripped.
    fn record_name_form_of(&mut self, value: &str) {
        for piece in split_modifier_commas(value) {
            let (base, _) = term_modifiers(piece.trim());
            if base.is_empty() {
                continue;
            }
            match base.split_once(':') {
                Some((code, term)) if language_table().name(code).is_some() => {
                    if code == "en" {
                        self.record_form_of(term);
                    }
                }
                _ => self.record_form_of(&base),
            }
        }
    }

    /// Record a name-list value's terms as variant FORMS of this
    /// word (the varform and dimform families run the reverse
    /// direction: the listed name is a form of the entry).
    fn record_name_forms(&mut self, value: &str) {
        for piece in split_modifier_commas(value) {
            let (base, _) = term_modifiers(piece.trim());
            if base.is_empty() {
                continue;
            }
            match base.split_once(':') {
                Some((code, term)) if language_table().name(code).is_some() => {
                    if code == "en" {
                        self.record_form(term);
                    }
                }
                _ => self.record_form(&base),
            }
        }
    }

    /// Anchored parts joined with " + ", the etymology convention.
    fn plus_joined(&mut self, parts: &[&str]) -> String {
        let anchored: Vec<String> = parts
            .iter()
            .filter(|part| !part.is_empty())
            .map(|part| self.anchored_argument(part))
            .collect();
        anchored.join(" + ")
    }

    /// The comma-listed source languages of an etymology reference,
    /// resolved through the vendored code map; an unknown code
    /// renders verbatim and audits - the growth signal.
    fn ety_language_names(&mut self, codes: &str, conjunction: &str) -> String {
        let names: Vec<String> = codes
            .split(',')
            .map(str::trim)
            .filter(|code| !code.is_empty())
            .map(|code| match language_table().name(code) {
                Some(name) => name.to_string(),
                None => {
                    self.audit("language_code_unknown", code.to_string());
                    code.to_string()
                }
            })
            .collect();
        serial_join(&names, conjunction)
    }

    /// An anchored term reference: embedded wikilinks render as they
    /// are; an alt displays against the term as target; inline
    /// modifiers strip first.
    fn ety_term_anchor(&mut self, term: &str, alt: &str) -> String {
        let (base, _) = term_modifiers(term);
        if base.contains("[[") || base.contains("{{") {
            return self.argument_text(&base);
        }
        let (alt_base, _) = term_modifiers(alt);
        if alt_base.is_empty() {
            anchor(&base, &base)
        } else if alt_base.contains("[[") || alt_base.contains("{{") {
            self.argument_text(&alt_base)
        } else {
            anchor(&alt_base, &base)
        }
    }

    /// A rendered gloss parenthetical: ` ("gloss")`, or nothing.
    fn gloss_text(&mut self, gloss: &str) -> String {
        let rendered = self.argument_text(gloss);
        if rendered.is_empty() {
            String::new()
        } else {
            format!(" (\"{rendered}\")")
        }
    }

    /// Wording applied around an etymology body: the prefix drops
    /// under notext= and the whole lowercases under nocap=.
    fn ety_wording(body: String, prefix: &str, notext: bool, nocap: bool) -> String {
        let text = if notext || prefix.is_empty() {
            body
        } else {
            format!("{prefix}{body}")
        };
        if nocap { lcfirst(&text) } else { text }
    }

    /// The der/bor/inh/cog family: "<Language> [term] ("gloss")",
    /// with the complete-wording variants carrying a prefix.
    /// langs_slot is 1 for the derivation family (slot 0 is the
    /// entry language) and 0 for cog and m+.
    fn ety_reference(
        &mut self,
        template: &Template,
        langs_slot: usize,
        prefix: &str,
    ) -> String {
        let template = normalize_numbered(template);
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let positional: Vec<&str> =
            template.positional.iter().map(String::as_str).collect();
        let langs = positional.get(langs_slot).copied().unwrap_or_default();
        let conjunction = named("conj").unwrap_or("and").to_string();
        let mut out = self.ety_language_names(langs, &conjunction);
        let term = positional.get(langs_slot + 1).copied().unwrap_or_default();
        let alt = positional
            .get(langs_slot + 2)
            .copied()
            .filter(|value| !value.is_empty())
            .or(named("alt"))
            .unwrap_or_default();
        if !term.is_empty() && term != "-" {
            let anchored = self.ety_term_anchor(term, alt);
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&anchored);
        }
        let gloss = positional
            .get(langs_slot + 3)
            .copied()
            .filter(|value| !value.is_empty())
            .or(named("t"))
            .or(named("gloss"))
            .unwrap_or_default();
        out.push_str(&self.gloss_text(gloss));
        Self::ety_wording(out, prefix, named("notext").is_some(), named("nocap").is_some())
    }

    /// The doublet family: anchored terms with per-index alt/gloss,
    /// serial-joined, behind the given wording.
    fn doublet_text(&mut self, template: &Template, prefix: &str) -> String {
        let template = normalize_numbered(template);
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let mut terms: Vec<String> = Vec::new();
        for (index, raw) in template.positional.iter().enumerate().skip(1) {
            if raw.trim().is_empty() {
                continue;
            }
            let alt = named(&format!("alt{index}")).unwrap_or_default().to_string();
            let gloss = named(&format!("t{index}")).unwrap_or_default().to_string();
            let mut rendered = self.ety_term_anchor(raw, &alt);
            rendered.push_str(&self.gloss_text(&gloss));
            terms.push(rendered);
        }
        let body = serial_join(&terms, "and");
        Self::ety_wording(body, prefix, named("notext").is_some(), named("nocap").is_some())
    }

    /// The unknown/uncertain statements: a fixed word, title=
    /// override, notext and nocap honored.
    fn origin_statement(&mut self, template: &Template, default: &str) -> String {
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        if named("notext").is_some() {
            return String::new();
        }
        let text = match named("title") {
            Some(title) => self.argument_text(title),
            None => String::from(default),
        };
        if named("nocap").is_some() { lcfirst(&text) } else { text }
    }

    /// The etymon template: a data carrier that renders nothing
    /// without text= (its own on-page behavior); with text=, the one
    /// step the page itself carries renders - keyword wording plus
    /// its etymons.
    fn etymon_text(&mut self, template: &Template) -> String {
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        if named("text").is_none() {
            if named("tree").is_some() {
                self.audit("etymon_tree_dropped", template_signature(template));
            }
            return String::new();
        }
        let entry_lang = template
            .positional
            .first()
            .map(String::as_str)
            .unwrap_or_default()
            .to_string();
        enum Wording {
            Prefix(&'static str),
            Join,
            Skip,
        }
        let wording_of = |keyword: &str| -> Option<Wording> {
            Some(match keyword {
                "from" => Wording::Prefix("From "),
                "der" | "derived" | "uder" => Wording::Prefix("Derived from "),
                "inh" | "inherited" => Wording::Prefix("Inherited from "),
                "bor" | "borrowed" => Wording::Prefix("Borrowed from "),
                "lbor" => Wording::Prefix("Learned borrowing from "),
                "slbor" => Wording::Prefix("Semi-learned borrowing from "),
                "obor" => Wording::Prefix("Orthographic borrowing from "),
                "ubor" => Wording::Prefix("Unadapted borrowing from "),
                "calque" | "cal" | "clq" => Wording::Prefix("Calque of "),
                "partial calque" | "pcal" => Wording::Prefix("Partial calque of "),
                "semantic loan" | "sl" => Wording::Prefix("Semantic loan from "),
                "influence" => Wording::Prefix("Influenced by "),
                "blend" => Wording::Prefix("Blend of "),
                "reduplication" | "redup" => Wording::Prefix("Reduplication of "),
                "abbreviation" | "abbr" => Wording::Prefix("Abbreviation of "),
                "syllabic abbreviation" | "sylabbr" => {
                    Wording::Prefix("Syllabic abbreviation of ")
                }
                "acronym" | "acro" => Wording::Prefix("Acronym of "),
                "initialism" | "init" => Wording::Prefix("Initialism of "),
                "clipping" | "clip" => Wording::Prefix("Clipping of "),
                "ellipsis" | "ellip" => Wording::Prefix("Ellipsis of "),
                "univerbation" | "univ" => Wording::Prefix("Univerbation of "),
                "back-formation" | "bf" => Wording::Prefix("Back-formation from "),
                "deverbal" => Wording::Prefix("Deverbal from "),
                "denominal" | "denom" => Wording::Prefix("Denominal from "),
                "affix" | "af" => Wording::Join,
                "afeq" | "root" => Wording::Skip,
                _ => return None,
            })
        };
        let mut groups: Vec<String> = Vec::new();
        let mut keyword = String::from("from");
        let mut keyword_uncertain = false;
        let mut conjunction = String::from("or");
        let mut terms: Vec<String> = Vec::new();
        let mut flush = |renderer: &mut Self,
                         keyword: &str,
                         uncertain: bool,
                         conjunction: &str,
                         terms: &mut Vec<String>| {
            if terms.is_empty() {
                return;
            }
            let taken = std::mem::take(terms);
            match wording_of(keyword) {
                Some(Wording::Prefix(prefix)) => {
                    let joined = serial_join(&taken, conjunction);
                    let text = if uncertain {
                        format!("Possibly {}{joined}", lcfirst(prefix))
                    } else {
                        format!("{prefix}{joined}")
                    };
                    groups.push(text);
                }
                Some(Wording::Join) => groups.push(taken.join(" + ")),
                Some(Wording::Skip) => {}
                None => {
                    renderer.audit("etymon_keyword_unknown", keyword.to_string());
                }
            }
        };
        for raw in template.positional.iter().skip(1) {
            if let Some(rest) = raw.strip_prefix(':') {
                flush(self, &keyword, keyword_uncertain, &conjunction, &mut terms);
                let (base, modifiers) = term_modifiers(rest);
                keyword = base;
                keyword_uncertain = modifiers.iter().any(|(name, _)| name == "unc");
                conjunction = modifiers
                    .iter()
                    .find(|(name, _)| name == "conj")
                    .map(|(_, value)| value.clone())
                    .unwrap_or_else(|| String::from("or"));
                continue;
            }
            if raw.trim().is_empty() {
                continue;
            }
            let (base, modifiers) = term_modifiers(raw);
            let modifier = |key: &str| -> Option<&str> {
                modifiers
                    .iter()
                    .find(|(name, _)| name == key)
                    .map(|(_, value)| value.as_str())
            };
            let (code, term) = match base.split_once(':') {
                Some((code, term)) if language_table().name(code).is_some() => {
                    (code.to_string(), term.to_string())
                }
                _ => (entry_lang.clone(), base.clone()),
            };
            let mut rendered = String::new();
            if code != entry_lang {
                rendered = self.ety_language_names(&code, "and");
            }
            if term != "-" && !term.is_empty() {
                let anchored =
                    self.ety_term_anchor(&term, modifier("alt").unwrap_or_default());
                if !rendered.is_empty() {
                    rendered.push(' ');
                }
                rendered.push_str(&anchored);
            }
            if let Some(gloss) = modifier("t") {
                let gloss = gloss.to_string();
                rendered.push_str(&self.gloss_text(&gloss));
            }
            if !rendered.is_empty() {
                terms.push(rendered);
            }
        }
        flush(self, &keyword, keyword_uncertain, &conjunction, &mut terms);
        groups.join(", ")
    }

    /// The inflection-of engine: grammar tags resolved through the
    /// vendored map, "of [lemma]" closing; `//` multiparts, the
    /// punctuation tags, and `;` set breaks per the documented
    /// grammar.
    fn inflection_of_text(&mut self, template: &Template) -> String {
        let template = normalize_numbered(template);
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        if named("enclitic").is_some() {
            self.audit("form_of_arguments_unhandled", template_signature(&template));
        }
        let positional: Vec<&str> =
            template.positional.iter().map(String::as_str).collect();
        let lemmas_raw = positional.get(1).copied().unwrap_or_default();
        let alt = positional
            .get(2)
            .copied()
            .filter(|value| !value.is_empty())
            .or(named("alt"))
            .unwrap_or_default()
            .to_string();
        let mut lemmas: Vec<String> = Vec::new();
        let pieces = split_modifier_commas(lemmas_raw);
        for (index, piece) in pieces.iter().enumerate() {
            if piece.trim().is_empty() {
                continue;
            }
            self.record_form_of(piece);
            let (_, modifiers) = term_modifiers(piece);
            let piece_alt = if index == 0 && pieces.len() == 1 && !alt.is_empty() {
                alt.clone()
            } else {
                modifiers
                    .iter()
                    .find(|(name, _)| name == "alt")
                    .map(|(_, value)| value.clone())
                    .unwrap_or_default()
            };
            let mut rendered = self.ety_term_anchor(piece, &piece_alt);
            if let Some((_, gloss)) =
                modifiers.iter().find(|(name, _)| name == "t")
            {
                let gloss = gloss.clone();
                rendered.push_str(&self.gloss_text(&gloss));
            }
            lemmas.push(rendered);
        }
        let mut sets: Vec<Vec<String>> = vec![Vec::new()];
        for raw_tag in positional.iter().skip(3) {
            if raw_tag.trim().is_empty() {
                continue;
            }
            if *raw_tag == ";" {
                sets.push(Vec::new());
                continue;
            }
            sets.last_mut()
                .expect("sets starts non-empty")
                .extend(tag_table().resolve(raw_tag.trim()));
        }
        let joined_sets: Vec<String> = sets
            .iter()
            .filter(|set| !set.is_empty())
            .map(|set| join_tags(set))
            .collect();
        let mut out = joined_sets.join("; ");
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str("of ");
        out.push_str(&lemmas.join(", "));
        if let Some(gloss) = named("t").or(named("gloss")) {
            let gloss = gloss.to_string();
            out.push_str(&self.gloss_text(&gloss));
        }
        if named("nocap").is_some() { out } else { ucfirst(&out) }
    }

    /// The comma-multi form-of shape shared by short for and only
    /// used in: "<Label> [a], [b]" with inline modifiers and the
    /// template-level gloss.
    fn multi_term_form_of(&mut self, template: &Template, label: &str) -> String {
        let template = normalize_numbered(template);
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let terms_raw = template
            .positional
            .get(1)
            .cloned()
            .unwrap_or_default();
        let mut rendered: Vec<String> = Vec::new();
        for piece in split_modifier_commas(&terms_raw) {
            let (base, modifiers) = term_modifiers(piece.trim());
            if base.is_empty() {
                continue;
            }
            let alt = modifiers
                .iter()
                .find(|(name, _)| name == "alt")
                .map(|(_, value)| value.clone())
                .unwrap_or_default();
            let mut text = self.ety_term_anchor(&base, &alt);
            if let Some((_, gloss)) = modifiers.iter().find(|(name, _)| name == "t") {
                let gloss = gloss.clone();
                text.push_str(&self.gloss_text(&gloss));
            }
            rendered.push(text);
        }
        if rendered.is_empty() {
            self.audit("template_unhandled", template_signature(&template));
            return String::new();
        }
        let mut out = format!("{label} {}", rendered.join(", "));
        if let Some(gloss) = named("t").or(named("gloss")) {
            let gloss = gloss.to_string();
            out.push_str(&self.gloss_text(&gloss));
        }
        if named("nocap").is_some() { lcfirst(&out) } else { out }
    }

    /// The demonym definitions: toponyms may embed place single-spec
    /// markers; a w:-led toponym anchors its article name.
    fn demonym_text(&mut self, template: &Template, noun: bool) -> String {
        let template = normalize_numbered(template);
        let toponyms: Vec<String> = template
            .positional
            .iter()
            .skip(1)
            .filter(|part| !part.trim().is_empty())
            .map(|part| match part.trim().strip_prefix("w:") {
                Some(rest) => anchor(rest, rest),
                None => self.place_single_spec(part.trim()),
            })
            .collect();
        if toponyms.is_empty() {
            self.audit("template_unhandled", template_signature(&template));
            return String::new();
        }
        if noun {
            format!(
                "A native or inhabitant of {}",
                toponyms.join(", or of ")
            )
        } else {
            format!("Of, from, or relating to {}", toponyms.join(", or "))
        }
    }

    /// The SI-unit definition, the template's own switch tables:
    /// "(metrology) An SI unit of <quantity> equal to 10^<n>
    /// [<base>]s. Symbol: <prefix><base symbol>".
    fn si_unit_text(&mut self, template: &Template) -> String {
        const PREFIXES: [(&str, &str, &str); 24] = [
            ("quecto", "-30", "q"),
            ("ronto", "-27", "r"),
            ("yocto", "-24", "y"),
            ("zepto", "-21", "z"),
            ("atto", "-18", "a"),
            ("femto", "-15", "f"),
            ("pico", "-12", "p"),
            ("nano", "-9", "n"),
            ("micro", "-6", "μ"),
            ("milli", "-3", "m"),
            ("centi", "-2", "c"),
            ("deci", "-1", "d"),
            ("deca", "1", "da"),
            ("hecto", "2", "h"),
            ("kilo", "3", "k"),
            ("mega", "6", "M"),
            ("giga", "9", "G"),
            ("tera", "12", "T"),
            ("peta", "15", "P"),
            ("exa", "18", "E"),
            ("zetta", "21", "Z"),
            ("yotta", "24", "Y"),
            ("ronna", "27", "R"),
            ("quetta", "30", "Q"),
        ];
        const BASES: [(&str, &str, &str); 17] = [
            ("ampere", "current", "A"),
            ("candela", "luminous intensity", "cd"),
            ("kelvin", "temperature", "K"),
            ("gram", "mass", "g"),
            ("gramme", "mass", "g"),
            ("meter", "length", "m"),
            ("metre", "length", "m"),
            ("mole", "amount of substance", "mol"),
            ("second", "time", "s"),
            ("coulomb", "charge", "C"),
            ("farad", "capacitance", "F"),
            ("hertz", "frequency", "Hz"),
            ("joule", "energy", "J"),
            ("newton", "force", "N"),
            ("ohm", "resistance", "Ω"),
            ("pascal", "pressure", "Pa"),
            ("watt", "power", "W"),
        ];
        let template = normalize_numbered(template);
        let prefix = template.positional.get(1).cloned().unwrap_or_default();
        let base = template.positional.get(2).cloned().unwrap_or_default();
        let quantity_override = template
            .positional
            .get(3)
            .filter(|value| !value.is_empty())
            .cloned();
        let Some(&(_, exponent, prefix_symbol)) =
            PREFIXES.iter().find(|(name, _, _)| *name == prefix)
        else {
            self.audit("template_unhandled", template_signature(&template));
            return String::new();
        };
        let known_base = BASES.iter().find(|(name, _, _)| *name == base);
        let quantity = match (&quantity_override, known_base) {
            (Some(quantity), _) => quantity.clone(),
            (None, Some(&(_, quantity, _))) => quantity.to_string(),
            (None, None) => {
                self.audit("template_unhandled", template_signature(&template));
                return String::new();
            }
        };
        let base_symbol = known_base.map(|&(_, _, symbol)| symbol).unwrap_or_default();
        format!(
            "(metrology) An SI unit of {quantity} equal to 10^{exponent} \
             [{base}]s. Symbol: {prefix_symbol}{base_symbol}"
        )
    }

    /// The name-translit family (Module:names' entry point): a
    /// transliterated, respelled, or orthographically borrowed name,
    /// "of the <Language> <type> [name]" or "of a <Language> <type>"
    /// when no name is given.
    fn name_translit_text(&mut self, template: &Template, desctext: &str) -> String {
        let template = normalize_numbered(template);
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let sources = template.positional.get(1).cloned().unwrap_or_default();
        let language_names = self.ety_language_names(&sources, "or");
        let types = named("type").unwrap_or("patronymic").to_string();
        let mut type_text = types.split(',').map(str::trim).collect::<Vec<&str>>().join(", ");
        if named("dim").is_some() {
            type_text.push_str(" diminutive");
        } else if named("aug").is_some() {
            type_text.push_str(" augmentative");
        }
        let mut names: Vec<String> = Vec::new();
        for raw in template.positional.iter().skip(2) {
            let (base, modifiers) = term_modifiers(raw.trim());
            if base.is_empty() {
                continue;
            }
            let mut text = self.ety_term_anchor(&base, "");
            for (key, value) in &modifiers {
                match key.as_str() {
                    "t" => {
                        let gloss = value.clone();
                        text.push_str(&self.gloss_text(&gloss));
                    }
                    "xlit" => text.push_str(&format!(", {}", anchor(value, value))),
                    "eq" => {
                        text.push_str(&format!(", equivalent to {}", anchor(value, value)));
                    }
                    _ => {}
                }
            }
            names.push(text);
        }
        let mut out = format!("{desctext} of ");
        if names.is_empty() {
            out.push_str(&format!(
                "{} {language_names} {type_text}",
                indefinite_article(&language_names)
            ));
        } else {
            out.push_str(&format!(
                "the {language_names} {type_text} {}",
                names.join(" or ")
            ));
        }
        if let Some(addl) = named("addl") {
            let addl = addl.to_string();
            let rendered = self.argument_text(&addl);
            if !rendered.is_empty() {
                out.push_str(&format!(", {rendered}"));
            }
        }
        if named("nocap").is_some() { out } else { ucfirst(&out) }
    }

    /// A comma-separated name-list value: each term optionally
    /// "code:term" with inline modifiers; a foreign term renders
    /// behind its language name, and include_language forces the
    /// name even for English (the eq= convention).
    fn name_list(
        &mut self,
        value: &str,
        include_language: bool,
        conjunction: &str,
    ) -> (String, usize) {
        let mut rendered: Vec<String> = Vec::new();
        for piece in split_modifier_commas(value) {
            if piece.trim().is_empty() {
                continue;
            }
            let (base, modifiers) = term_modifiers(piece.trim());
            let (code, term) = match base.split_once(':') {
                Some((code, term)) if language_table().name(code).is_some() => {
                    (code.to_string(), term.to_string())
                }
                _ => (String::from("en"), base.clone()),
            };
            let alt = modifiers
                .iter()
                .find(|(name, _)| name == "alt")
                .map(|(_, value)| value.clone())
                .unwrap_or_default();
            let mut text = String::new();
            if include_language || code != "en" {
                text.push_str(&self.ety_language_names(&code, "and"));
                text.push(' ');
            }
            text.push_str(&self.ety_term_anchor(&term, &alt));
            if let Some((_, gloss)) = modifiers.iter().find(|(name, _)| name == "t") {
                let gloss = gloss.clone();
                text.push_str(&self.gloss_text(&gloss));
            }
            rendered.push(text);
        }
        let count = rendered.len();
        (serial_join(&rendered, conjunction), count)
    }

    /// One from= piece's (prefix, suffix) wording, the Module:names
    /// grammar: the category keywords, languages and families by
    /// name, and code:term references.
    fn name_from_piece(&mut self, piece: &str) -> (String, String) {
        let trimmed = piece.trim();
        match trimmed {
            "surnames" | "given names" | "nicknames" | "place names"
            | "common nouns" | "month names" => (
                String::from("transferred from the "),
                trimmed.trim_end_matches('s').to_string(),
            ),
            "patronymics" | "matronymics" | "coinages" => (
                String::from("originating "),
                format!("as a {}", trimmed.trim_end_matches('s')),
            ),
            "occupations" | "ethnonyms" => (
                String::from("originating "),
                format!("as an {}", trimmed.trim_end_matches('s')),
            ),
            "the Bible" => {
                (String::from("originating "), String::from("from the Bible"))
            }
            _ => {
                if trimmed.contains(':') {
                    let (text, _) = self.name_list(trimmed, true, "and");
                    (String::from("from "), text)
                } else if trimmed.ends_with(" languages")
                    || trimmed.ends_with(" Languages")
                    || trimmed.ends_with(" lects")
                    || trimmed.ends_with(" Lects")
                {
                    (String::from("from "), format!("the {trimmed}"))
                } else {
                    (String::from("from "), trimmed.to_string())
                }
            }
        }
    }

    /// A from= value rendered whole: comma pieces group under shared
    /// prefixes, and a " < " chain renders its tail as parenthesized
    /// in-turn steps.
    fn name_from_text(&mut self, values: &[String]) -> String {
        let mut segments: Vec<(String, Vec<String>)> = Vec::new();
        for value in values {
            for piece in split_modifier_commas(value) {
                if piece.trim().is_empty() {
                    continue;
                }
                let chain: Vec<&str> = piece.split(" < ").collect();
                if chain.len() == 1 {
                    let (prefix, suffix) = self.name_from_piece(chain[0]);
                    match segments.last_mut() {
                        Some((last_prefix, suffixes)) if *last_prefix == prefix => {
                            suffixes.push(suffix);
                        }
                        _ => segments.push((prefix, vec![suffix])),
                    }
                    continue;
                }
                let mut steps: Vec<String> = Vec::new();
                for step in &chain {
                    let (prefix, suffix) = self.name_from_piece(step);
                    steps.push(format!("{prefix}{suffix}"));
                }
                let full = format!(
                    "{} (in turn {})",
                    steps[0],
                    steps[1..].join(", in turn ")
                );
                segments.push((String::new(), vec![full]));
            }
        }
        let rendered: Vec<String> = segments
            .into_iter()
            .map(|(prefix, suffixes)| {
                format!("{prefix}{}", serial_join(&suffixes, "or"))
            })
            .collect();
        serial_join(&rendered, "or")
    }

    /// The surname definition line, the Module:names assembly:
    /// article, genders, adjective, "surname", then the qualifying
    /// pieces in their documented order.
    fn surname_text(&mut self, template: &Template) -> String {
        let template = normalize_numbered(template);
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let typetext = |renderer: &mut Self, template: &Template, key: &str| -> String {
            match template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
            {
                Some(value) if !value.is_empty() => {
                    format!("{} ", renderer.argument_text(value))
                }
                _ => String::new(),
            }
        };
        let adj = template
            .positional
            .get(1)
            .filter(|value| !value.is_empty())
            .cloned();
        let mut genders: Vec<&str> = Vec::new();
        for gender in named("g").unwrap_or_default().split(',') {
            match gender.trim() {
                "" => {}
                "m" | "male" => genders.push("male"),
                "f" | "female" => genders.push("female"),
                "c" | "common gender" | "common-gender" | "unisex" => {
                    genders.push("common-gender")
                }
                _ => genders.push("unknown-gender"),
            }
        }
        let article = match named("A") {
            Some(article) => article.to_string(),
            None => {
                let bare = if genders.first() == Some(&"unknown-gender") {
                    "an"
                } else if genders.is_empty() {
                    match &adj {
                        Some(adj) => indefinite_article(adj),
                        None => "a",
                    }
                } else {
                    "a"
                };
                if named("nocap").is_some() {
                    bare.to_string()
                } else {
                    ucfirst(bare)
                }
            }
        };
        let mut out = format!("{article} ");
        if !genders.is_empty() {
            out.push_str(&genders.join(" or "));
            out.push(' ');
        }
        if let Some(adj) = &adj {
            let adj = adj.clone();
            out.push_str(&self.argument_text(&adj));
            out.push(' ');
        }
        out.push_str("surname");
        let mut need_comma = false;
        let xlit = named_family(&template, "xlit");
        if !xlit.is_empty() {
            let (text, _) = self.name_list(&xlit.join(","), false, "and");
            out.push_str(&format!(", {text}"));
            need_comma = true;
        }
        let from = named_family(&template, "from");
        if !from.is_empty() {
            if need_comma {
                out.push(',');
            }
            need_comma = true;
            out.push(' ');
            out.push_str(&typetext(self, &template, "fromtype"));
            let text = self.name_from_text(&from);
            out.push_str(&text);
        }
        let mut meanings: Vec<String> = named_family(&template, "meaning")
            .iter()
            .map(|meaning| format!("\"{}\"", self.argument_text(meaning)))
            .collect();
        let parent = named_family(&template, "parent");
        if !parent.is_empty() {
            let (text, _) = self.name_list(&parent.join(","), false, "and");
            let child = match (genders.contains(&"male"), genders.contains(&"female")) {
                (true, false) => "son",
                (false, true) => "daughter",
                _ => "son/daughter",
            };
            meanings.push(format!("\"{child} of {text}\""));
        }
        if !meanings.is_empty() {
            if need_comma {
                out.push(',');
            }
            need_comma = true;
            out.push(' ');
            out.push_str(&typetext(self, &template, "meaningtype"));
            out.push_str(&format!("meaning {}", serial_join(&meanings, "or")));
        }
        if let Some(origin) = named("origin") {
            if need_comma {
                out.push(',');
            }
            need_comma = true;
            let origin = origin.to_string();
            out.push_str(&format!(" of {} origin", self.argument_text(&origin)));
        }
        if let Some(usage) = named("usage") {
            if need_comma {
                out.push(',');
            }
            let usage = usage.to_string();
            out.push_str(&format!(" of {} usage", self.argument_text(&usage)));
        }
        for (family, label, conjunction) in [
            ("varof", "variant of", "and"),
            ("var", "variant of", "and"),
            ("clipof", "clipping of", "and"),
            ("blend", "blend of", "and"),
            ("m", "masculine equivalent", "and"),
            ("f", "feminine equivalent", "and"),
        ] {
            let values = named_family(&template, family);
            if values.is_empty() {
                continue;
            }
            // A variant or clipping IS a form of its target; the
            // gender equivalents are semantic and stay unrecorded.
            if matches!(family, "varof" | "var" | "clipof") {
                for value in &values {
                    self.record_name_form_of(value);
                }
            }
            let (text, _) = self.name_list(&values.join(","), false, conjunction);
            out.push_str(&format!(", {label} {text}"));
        }
        let eq = named_family(&template, "eq");
        if !eq.is_empty() {
            let (text, _) = self.name_list(&eq.join(","), true, "or");
            out.push_str(&format!(
                ", {}equivalent to {text}",
                typetext(self, &template, "eqtype")
            ));
        }
        if let Some(addl) = named("addl") {
            let addl = addl.to_string();
            let rendered = self.argument_text(&addl);
            if let Some(rest) = rendered.strip_prefix(';') {
                out.push_str(&format!("; {}", rest.trim_start()));
            } else if let Some(rest) = rendered.strip_prefix('_') {
                out.push_str(&format!(" {rest}"));
            } else if !rendered.is_empty() {
                out.push_str(&format!(", {rendered}"));
            }
        }
        let varform = named_family(&template, "varform");
        if !varform.is_empty() {
            for value in &varform {
                self.record_name_forms(value);
            }
            let (text, count) = self.name_list(&varform.join(","), false, "and");
            let plural = if count > 1 { "s" } else { "" };
            out.push_str(&format!(
                "; {}variant form{plural} {text}",
                typetext(self, &template, "varformtype")
            ));
        }
        out
    }

    /// The given-name definition line, the Module:names assembly:
    /// diminutive-of leads when present, then genders, "given
    /// name(s)", and the qualifying pieces in order.
    fn given_name_text(&mut self, template: &Template) -> String {
        let template = normalize_numbered(template);
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let typetext = |renderer: &mut Self, template: &Template, key: &str| -> String {
            match template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
            {
                Some(value) if !value.is_empty() => {
                    format!("{} ", renderer.argument_text(value))
                }
                _ => String::new(),
            }
        };
        const ANIMALS: [&str; 5] = ["dog", "cat", "cow", "horse", "animal"];
        let gender_raw = template
            .positional
            .get(1)
            .cloned()
            .filter(|value| !value.is_empty())
            .or_else(|| named("gender").map(str::to_string))
            .unwrap_or_default();
        let mut genders: Vec<String> = Vec::new();
        let mut is_animal = false;
        for piece in split_modifier_commas(&gender_raw) {
            let (base, modifiers) = term_modifiers(piece.trim());
            if base.is_empty() {
                continue;
            }
            let display = modifiers
                .iter()
                .find(|(name, _)| name == "text")
                .map(|(_, value)| value.replace('+', &base))
                .unwrap_or_else(|| base.clone());
            if ANIMALS.contains(&base.as_str()) {
                is_animal = true;
            }
            genders.push(display);
        }
        let dimof: Vec<String> = [named_family(&template, "dimof"), named_family(&template, "dim")]
            .concat();
        // A diminutive is a form of the names it shortens.
        for value in &dimof {
            if value != "-" {
                self.record_name_form_of(value);
            }
        }
        let mut out = String::new();
        let mut force_plural = false;
        if !dimof.is_empty() {
            out.push_str(&typetext(self, &template, "dimoftype"));
            out.push_str(&typetext(self, &template, "dimtype"));
            out.push_str("diminutive");
            let xlit = named_family(&template, "xlit");
            if !xlit.is_empty() {
                let (text, _) = self.name_list(&xlit.join(","), false, "and");
                out.push_str(&format!(", {text},"));
            }
            out.push_str(" of ");
            if dimof.len() == 1 && dimof[0] == "-" {
                force_plural = true;
            } else {
                out.push_str("the ");
            }
        }
        if !is_animal && !genders.is_empty() {
            out.push_str(&genders.join(" or "));
            out.push(' ');
        }
        let plural = force_plural || {
            let (_, count) = {
                let joined = dimof.join(",");
                if joined.is_empty() || joined == "-" {
                    (String::new(), 0)
                } else {
                    (String::new(), split_modifier_commas(&joined).len())
                }
            };
            count > 1
        };
        out.push_str(if plural { "given names" } else { "given name" });
        let mut need_comma = false;
        let bare_dash = dimof.len() == 1 && dimof[0] == "-";
        if !dimof.is_empty() && !bare_dash {
            let (text, _) = self.name_list(&dimof.join(","), false, "and");
            out.push_str(&format!(" {text}"));
            need_comma = !is_animal;
        } else if dimof.is_empty() {
            // With a diminutive-of, the xlit already rode inside the
            // leading "diminutive, X, of" text.
            let xlit = named_family(&template, "xlit");
            if !xlit.is_empty() {
                let (text, _) = self.name_list(&xlit.join(","), false, "and");
                out.push_str(&format!(", {text}"));
                need_comma = true;
            }
        }
        if is_animal {
            if need_comma {
                out.push(',');
            }
            need_comma = true;
            let gender_text = genders.join(" or ");
            out.push_str(&format!(
                " for {} {gender_text}",
                indefinite_article(&gender_text)
            ));
        }
        let from = named_family(&template, "from");
        if !from.is_empty() {
            if need_comma {
                out.push(',');
            }
            need_comma = true;
            out.push(' ');
            out.push_str(&typetext(self, &template, "fromtype"));
            let text = self.name_from_text(&from);
            out.push_str(&text);
        }
        let meanings: Vec<String> = named_family(&template, "meaning")
            .iter()
            .map(|meaning| format!("\"{}\"", self.argument_text(meaning)))
            .collect();
        if !meanings.is_empty() {
            if need_comma {
                out.push(',');
            }
            need_comma = true;
            out.push(' ');
            out.push_str(&typetext(self, &template, "meaningtype"));
            out.push_str(&format!("meaning {}", serial_join(&meanings, "or")));
        }
        if let Some(origin) = named("origin") {
            if need_comma {
                out.push(',');
            }
            need_comma = true;
            let origin = origin.to_string();
            out.push_str(&format!(" of {} origin", self.argument_text(&origin)));
        }
        if let Some(usage) = named("usage") {
            if need_comma {
                out.push(',');
            }
            let usage = usage.to_string();
            out.push_str(&format!(" of {} usage", self.argument_text(&usage)));
        }
        for (family, label) in [
            ("varof", "variant of"),
            ("var", "variant of"),
            ("clipof", "clipping of"),
        ] {
            let values = named_family(&template, family);
            if values.is_empty() {
                continue;
            }
            for value in &values {
                self.record_name_form_of(value);
            }
            let (text, _) = self.name_list(&values.join(","), false, "and");
            let label_type = format!("{family}type");
            out.push_str(&format!(
                ", {}{label} {text}",
                typetext(self, &template, &label_type)
            ));
        }
        let blend = named_family(&template, "blend");
        if !blend.is_empty() {
            let (text, _) = self.name_list(&blend.join(","), false, "and");
            out.push_str(&format!(
                ", {}blend of {text}",
                typetext(self, &template, "blendtype")
            ));
        }
        if let Some(popular) = named("popular") {
            let popular = popular.to_string();
            out.push_str(&format!(
                ", {}popular {}",
                typetext(self, &template, "populartype"),
                self.argument_text(&popular)
            ));
        }
        for (family, label) in [
            ("m", "masculine equivalent"),
            ("f", "feminine equivalent"),
        ] {
            let values = named_family(&template, family);
            if values.is_empty() {
                continue;
            }
            let (text, _) = self.name_list(&values.join(","), false, "and");
            out.push_str(&format!(", {label} {text}"));
        }
        let eq = named_family(&template, "eq");
        if !eq.is_empty() {
            let (text, _) = self.name_list(&eq.join(","), true, "or");
            out.push_str(&format!(
                ", {}equivalent to {text}",
                typetext(self, &template, "eqtype")
            ));
        }
        if let Some(addl) = named("addl") {
            let addl = addl.to_string();
            let rendered = self.argument_text(&addl);
            if let Some(rest) = rendered.strip_prefix(';') {
                out.push_str(&format!("; {}", rest.trim_start()));
            } else if let Some(rest) = rendered.strip_prefix('_') {
                out.push_str(&format!(" {rest}"));
            } else if !rendered.is_empty() {
                out.push_str(&format!(", {rendered}"));
            }
        }
        for (family, label) in [
            ("varform", "variant form"),
            ("dimform", "diminutive form"),
        ] {
            let values = named_family(&template, family);
            if values.is_empty() {
                continue;
            }
            for value in &values {
                self.record_name_forms(value);
            }
            let (text, count) = self.name_list(&values.join(","), false, "and");
            let plural = if count > 1 { "s" } else { "" };
            let label_type = format!("{family}type");
            out.push_str(&format!(
                "; {}{label}{plural} {text}",
                typetext(self, &template, &label_type)
            ));
        }
        let article = match named("A") {
            Some(article) => article.to_string(),
            None => {
                let bare = indefinite_article(&out);
                if named("nocap").is_some() {
                    bare.to_string()
                } else {
                    ucfirst(bare)
                }
            }
        };
        format!("{article} {out}")
    }

    /// A placetype spec's display: slash-parted, aliases expanded,
    /// recognized qualifiers canonicalized; returns the display and
    /// the article override a qualifier carries.
    fn placetype_display(&mut self, spec: &str) -> (String, String) {
        let table = place_table();
        let mut parts: Vec<String> = Vec::new();
        let mut article = String::new();
        for (index, part) in spec.split('/').enumerate() {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if part == "and" || part == "or" {
                parts.push(part.to_string());
                continue;
            }
            let expanded = table.expand_alias(part).to_string();
            // Leading recognized qualifiers display canonically; the
            // remainder expands as its own alias.
            let mut words: Vec<&str> = expanded.split(' ').collect();
            let mut display_words: Vec<String> = Vec::new();
            while words.len() > 1 {
                let Some((display, qualifier_article)) = table.qualifier(words[0]) else {
                    break;
                };
                if index == 0 && display_words.is_empty() && article.is_empty() {
                    match qualifier_article {
                        "the" => article = String::from("the"),
                        "none" => article = String::from("none"),
                        _ => {}
                    }
                }
                display_words.push(display.to_string());
                words.remove(0);
            }
            let reduced = table.expand_alias(&words.join(" ")).to_string();
            display_words.push(reduced);
            parts.push(display_words.join(" "));
        }
        let mut out = String::new();
        let mut previous_connector = true;
        for part in &parts {
            if part == "and" || part == "or" {
                out.push_str(&format!(" {part} "));
                previous_connector = true;
                continue;
            }
            if !previous_connector {
                out.push_str(", ");
            }
            out.push_str(part);
            previous_connector = false;
        }
        (out, article)
    }

    /// One holonym's display: alias-resolved location name, "the"
    /// where taken, placetype prefix/suffix modifiers applied, and
    /// the name anchored (a `:`-led name stays plain).
    fn holonym_display(
        &mut self,
        type_spec: &str,
        names: &str,
        allow_the: bool,
    ) -> String {
        let table = place_table();
        let mut type_parts = type_spec.split(':');
        let raw_type = type_parts.next().unwrap_or_default().trim();
        let modifiers: Vec<&str> = type_parts.map(str::trim).collect();
        let full_type = table.expand_alias(raw_type).to_string();
        let resolved = table.placetype_resolved(&full_type);
        let mut rendered: Vec<String> = Vec::new();
        let name_list = split_modifier_commas(names);
        for (index, raw_name) in name_list.iter().enumerate() {
            let raw_name = raw_name.trim();
            if raw_name.is_empty() {
                continue;
            }
            // A langcode: prefix names a foreign form; a bare colon
            // suppresses the anchor.
            let (no_anchor, name) = match raw_name.split_once(':') {
                Some(("", rest)) => (true, rest.to_string()),
                Some((code, rest)) if language_table().name(code).is_some() => {
                    (false, rest.to_string())
                }
                _ => (false, raw_name.to_string()),
            };
            let location = table.location(&name);
            let display_name = match location {
                Some(row) if !row.display_as.is_empty() => row.display_as.clone(),
                Some(row) if row.display_expand && !row.alias_of.is_empty() => {
                    row.alias_of.clone()
                }
                _ => name.clone(),
            };
            // Position gates whether an article may appear at all;
            // the name's own the-flag decides the plain and suffix
            // forms, while a prefix affix carries its position-gated
            // "the" regardless of the name (the state of New York).
            let position_the = allow_the || index > 0 || modifiers.contains(&"the");
            let the = position_the && table.holonym_takes_the(&full_type, &name);
            let mut text = String::new();
            let anchored = if no_anchor || display_name.contains("[[") {
                self.argument_text(&display_name)
            } else {
                anchor(&display_name, &display_name)
            };
            let affix_type = if modifiers.contains(&"noaff") {
                ""
            } else if let Some(explicit) = modifiers
                .iter()
                .find(|m| matches!(**m, "suf" | "Suf" | "pref" | "Pref"))
            {
                explicit
            } else {
                resolved.affix_type.as_str()
            };
            let affix_word = if resolved.affix.is_empty() {
                full_type.clone()
            } else {
                resolved.affix.clone()
            };
            let already_affixed = display_name
                .to_lowercase()
                .contains(&affix_word.to_lowercase());
            match affix_type {
                "suf" | "Suf" if !already_affixed => {
                    if the {
                        text.push_str("the ");
                    }
                    let word = if affix_type == "Suf" {
                        ucfirst(&affix_word)
                    } else {
                        affix_word.clone()
                    };
                    text.push_str(&format!("{anchored} {word}"));
                }
                "pref" | "Pref" if !already_affixed => {
                    let word = if affix_type == "Pref" {
                        ucfirst(&affix_word)
                    } else {
                        affix_word.clone()
                    };
                    if position_the {
                        text.push_str("the ");
                    }
                    text.push_str(&format!("{word} of "));
                    if table.holonym_takes_the(&full_type, &name) {
                        text.push_str("the ");
                    }
                    text.push_str(&anchored);
                }
                _ => {
                    if the {
                        text.push_str("the ");
                    }
                    text.push_str(&anchored);
                }
            }
            rendered.push(text);
        }
        serial_join(&rendered, "and")
    }

    /// A raw prose fragment appended with its boundary spaces kept
    /// (inline rendering trims, which would fuse fragments onto
    /// their neighboring markers).
    fn spaced_fragment(&mut self, raw: &str, out: &mut String) {
        let rendered = self.argument_text(raw);
        if rendered.is_empty() {
            if raw.chars().any(char::is_whitespace)
                && !out.is_empty()
                && !out.ends_with(' ')
            {
                out.push(' ');
            }
            return;
        }
        if raw.starts_with(char::is_whitespace)
            && !out.is_empty()
            && !out.ends_with(' ')
        {
            out.push(' ');
        }
        out.push_str(&rendered);
        if raw.ends_with(char::is_whitespace) {
            out.push(' ');
        }
    }

    /// A single-spec place description: `<<placetype>>` and
    /// `<<type/name>>` markers replaced inside the raw text.
    fn place_single_spec(&mut self, spec: &str) -> String {
        let mut out = String::new();
        let mut rest = spec;
        while let Some(open) = rest.find("<<") {
            let before = rest[..open].to_string();
            if !before.is_empty() {
                self.spaced_fragment(&before, &mut out);
            }
            let Some(close) = rest[open..].find(">>").map(|c| open + c) else {
                let tail = rest[open..].to_string();
                self.spaced_fragment(&tail, &mut out);
                return out;
            };
            let inner = rest[open + 2..close].to_string();
            match inner.split_once('/') {
                Some((type_spec, names)) => {
                    let text = self.holonym_display(type_spec, names, false);
                    out.push_str(&text);
                }
                None => {
                    let (display, _) = self.placetype_display(&inner);
                    out.push_str(&display);
                }
            }
            rest = &rest[close + 2..];
        }
        if !rest.is_empty() {
            let rest = rest.to_string();
            self.spaced_fragment(&rest, &mut out);
        }
        out
    }

    /// The place definition line: article and placetype, holonyms
    /// under the comma algorithm, restarts, and the extra-information
    /// tails. def= replaces the whole definition.
    fn place_text(&mut self, template: &Template) -> String {
        let template = normalize_numbered(template);
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        if let Some(definition) = named("def") {
            let definition = definition.to_string();
            return self.argument_text(&definition);
        }
        let pieces: Vec<&str> = template
            .positional
            .iter()
            .skip(1)
            .map(String::as_str)
            .collect();
        let mut out = String::new();
        let mut expect_placetype = true;
        let mut preposition = String::new();
        let mut seen_holonym = false;
        let mut previous_raw = false;
        let mut pending_join: Option<String> = None;
        let mut first_segment = true;
        for piece in pieces {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            if piece.starts_with('@') {
                self.audit("place_arguments_unhandled", template_signature(&template));
                continue;
            }
            // A `;` (or `;text`) restarts the description.
            if piece == ";" || piece == ";;" || (piece.starts_with(';') && !piece.starts_with("; ")) {
                let join = if piece == ";" {
                    String::from("; ")
                } else if piece == ";;" {
                    String::from(" ")
                } else {
                    let joiner = &piece[1..];
                    if joiner.chars().next().is_some_and(char::is_alphabetic) {
                        format!(" {joiner} ")
                    } else {
                        format!("{joiner} ")
                    }
                };
                pending_join = Some(join);
                expect_placetype = true;
                seen_holonym = false;
                previous_raw = false;
                first_segment = false;
                preposition.clear();
                continue;
            }
            if expect_placetype {
                if let Some(join) = pending_join.take() {
                    out.push_str(&join);
                }
                if piece.contains("<<") {
                    let text = self.place_single_spec(piece);
                    out.push_str(&text);
                    // Mixed format: later holonyms join with no
                    // inserted preposition.
                    seen_holonym = true;
                    previous_raw = false;
                } else {
                    let (mut display, article_override) = self.placetype_display(piece);
                    // A restarted description continues mid-sentence:
                    // its article stays lowercase.
                    let article = match named("a") {
                        Some(article) => format!("{article} "),
                        None => match article_override.as_str() {
                            "the" if first_segment => String::from("The "),
                            "the" => String::from("the "),
                            "none" => {
                                if first_segment {
                                    display = ucfirst(&display);
                                }
                                String::new()
                            }
                            _ if first_segment => {
                                format!("{} ", ucfirst(indefinite_article(&display)))
                            }
                            _ => format!("{} ", indefinite_article(&display)),
                        },
                    };
                    let first_type = piece.split('/').next().unwrap_or_default();
                    let resolved = place_table()
                        .placetype_resolved(place_table().expand_alias(first_type));
                    preposition = if resolved.preposition.is_empty() {
                        String::from("in")
                    } else {
                        resolved.preposition.clone()
                    };
                    out.push_str(&article);
                    out.push_str(&display);
                }
                expect_placetype = false;
                continue;
            }
            match piece.split_once('/') {
                Some((type_spec, names)) if !type_spec.trim().is_empty() => {
                    // Raw text before the first holonym carries its
                    // own preposition; the placetype's inserts only
                    // when holonyms follow the placetype directly.
                    let joiner = if previous_raw {
                        String::from(" ")
                    } else if !seen_holonym {
                        if preposition.is_empty() {
                            String::from(" ")
                        } else {
                            format!(" {preposition} ")
                        }
                    } else {
                        String::from(", ")
                    };
                    let first_position = !seen_holonym && !previous_raw;
                    let text = self.holonym_display(type_spec, names, first_position);
                    out.push_str(&joiner);
                    out.push_str(&text);
                    seen_holonym = true;
                    previous_raw = false;
                }
                _ => {
                    // Raw connector text; `and` and `in` and a
                    // *-led piece suppress the comma.
                    let stripped = piece.strip_prefix('*').unwrap_or(piece);
                    let no_comma = previous_raw
                        || !seen_holonym
                        || piece.starts_with('*')
                        || stripped == "and"
                        || stripped == "in"
                        || stripped.starts_with("and ")
                        || stripped.starts_with("in ");
                    let joiner = if no_comma { " " } else { ", " };
                    let stripped = stripped.to_string();
                    out.push_str(joiner);
                    out.push_str(&self.argument_text(&stripped));
                    previous_raw = true;
                }
            }
        }
        for (key, label) in [
            ("caplc", "capital and largest city"),
            ("capital", "capital"),
            ("largest city", "largest city"),
            ("seat", "seat"),
            ("shire town", "shire town"),
            ("official", "official name"),
            ("modern", "modern name"),
        ] {
            let values = named_family(&template, key);
            if values.is_empty() {
                continue;
            }
            let rendered: Vec<String> = values
                .iter()
                .map(|value| {
                    let trimmed = value.trim();
                    let name = trimmed.strip_prefix(':').unwrap_or(trimmed);
                    if name.contains("[[") {
                        let name = name.to_string();
                        self.argument_text(&name)
                    } else {
                        anchor(name, name)
                    }
                })
                .collect();
            out.push_str(&format!("; {label}: {}", rendered.join(", ")));
        }
        if let Some(addl) = named("addl") {
            let addl = addl.to_string();
            let rendered = self.argument_text(&addl);
            if let Some(rest) = rendered.strip_prefix(';') {
                out.push_str(&format!("; {}", rest.trim_start()));
            } else if !rendered.is_empty() {
                out.push_str(&format!(", {rendered}"));
            }
        }
        out
    }

    /// A citation line off a quote-family template: italicized title
    /// (plus work), a colon, the double-quoted passage with its bold
    /// target kept. Authors, dates, and identifiers drop.
    fn citation_text(&mut self, template: &Template) -> Option<String> {
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let passage = named("passage").or(named("text"))?;
        let mut out = String::new();
        if let Some(title) = named("title") {
            out.push('*');
            out.push_str(&self.argument_text(title));
            out.push('*');
        }
        if let Some(work) = named("work").or(named("journal")).or(named("magazine")) {
            if !out.is_empty() {
                out.push_str(", ");
            }
            out.push('*');
            out.push_str(&self.argument_text(work));
            out.push('*');
        }
        if out.is_empty() {
            self.audit("citation_untitled", template.name.clone());
            return None;
        }
        out.push_str(": \"");
        out.push_str(&self.argument_text(passage));
        out.push('"');
        Some(out)
    }

    /// The IPA pronunciation line: mapped accent names, the
    /// transcriptions joined; unknown codes stay verbatim, audited.
    fn ipa_text(&mut self, template: &Template) -> String {
        // `;` and `~` positionals are the IPA template's visual
        // separators, not transcriptions.
        let transcriptions: Vec<String> = template
            .positional
            .iter()
            .skip(1)
            .filter(|part| !part.is_empty() && *part != ";" && *part != "~")
            .map(|part| normalize_typography(part))
            .collect();
        let accents = template
            .named
            .iter()
            .find(|(name, _)| name == "a")
            .map(|(_, value)| value.as_str())
            .unwrap_or_default();
        let table = accent_table();
        let accent_display = |code: &str| -> Option<String> {
            if let Some((_, name)) =
                table.names.iter().find(|(known, _)| known == code)
            {
                return Some(name.clone());
            }
            if table.verbatim.iter().any(|known| known == code) {
                return Some(code.to_string());
            }
            None
        };
        // A piece may carry `<<X>>` label markers (each its own code)
        // and the `alias!display` bang form (the display side wins).
        let mut codes: Vec<String> = Vec::new();
        for piece in accents.split(',').filter(|piece| !piece.is_empty()) {
            if piece.contains("<<") {
                let mut rest = piece;
                while let Some(open) = rest.find("<<") {
                    let Some(close) = rest[open..].find(">>").map(|c| open + c)
                    else {
                        break;
                    };
                    codes.push(rest[open + 2..close].to_string());
                    rest = &rest[close + 2..];
                }
                continue;
            }
            codes.push(piece.to_string());
        }
        let mut names: Vec<String> = Vec::new();
        for code in &codes {
            let code = match code.split_once('!') {
                Some((_, right)) if !right.is_empty() => right,
                Some((left, _)) => left,
                None => code.as_str(),
            };
            if let Some(name) = accent_display(code) {
                names.push(name);
                continue;
            }
            // The non- prefix negates a resolvable base: "without
            // the X merger" / "without X".
            if let Some(base) = code.strip_prefix("non-")
                && let Some(resolved) = accent_display(base)
            {
                if resolved.ends_with("merger") {
                    names.push(format!("without the {resolved}"));
                } else {
                    names.push(format!("without {resolved}"));
                }
                continue;
            }
            match code.strip_prefix("dialects of ") {
                Some(place) => names.push(format!("also of {place}")),
                None => {
                    self.audit("accent_code_unknown", code.to_string());
                    names.push(code.to_string());
                }
            }
        }
        let joined = transcriptions.join(", ");
        if names.is_empty() {
            format!("IPA: {joined}")
        } else {
            format!("IPA ({}): {joined}", names.join(", "))
        }
    }

    /// The en-noun engine: positionals are plural specs (the special
    /// values plus literals), countability markers ride the same
    /// slots, and plurale tantum adds sg=/attr=.
    fn noun_parenthetical(&mut self, template: &Template, proper: bool) -> Option<String> {
        let title = self.page_title.to_string();
        let mut uncountable = false;
        let mut both = false;
        let mut plural_only = false;
        let mut unattested = false;
        let mut plurals: Vec<SpecForm> = Vec::new();
        for raw in &template.positional {
            for form in spec_forms(raw, &title) {
                let labels = form.labels.clone();
                match form.text.as_str() {
                    "-" => uncountable = true,
                    "~" => both = true,
                    "p" => plural_only = true,
                    "!" => unattested = true,
                    "?" => return None,
                    "+" | "^" => plurals.push(SpecForm {
                        text: s_form(&title, proper, false),
                        labels,
                    }),
                    "++" => {
                        // The module's ++ plural: a final s/z/x
                        // doubles before -es.
                        let text = if title.ends_with(['s', 'z', 'x']) {
                            doubled(&title, "es")
                        } else {
                            s_form(&title, proper, false)
                        };
                        plurals.push(SpecForm { text, labels });
                    }
                    "*" => plurals.push(SpecForm { text: title.clone(), labels }),
                    "s" => plurals.push(SpecForm { text: format!("{title}s"), labels }),
                    "es" => plurals.push(SpecForm { text: format!("{title}es"), labels }),
                    "ies" => {
                        // -ey words take -ies whole (whiskey ->
                        // whiskies); otherwise the -y drops.
                        let stem = title
                            .strip_suffix("ey")
                            .or_else(|| title.strip_suffix('y'))
                            .unwrap_or(&title);
                        plurals.push(SpecForm { text: format!("{stem}ies"), labels });
                    }
                    _ => plurals.push(form),
                }
            }
        }
        let plural_piece = |plurals: &[SpecForm]| -> String {
            let rendered: Vec<String> =
                plurals.iter().map(SpecForm::rendered).collect();
            format!("plural {}", rendered.join(" or "))
        };
        // sg=/attr= plus their numbered variants (sg2=, ...) collect
        // as additional forms.
        let collect_family = |base: &str| -> Vec<SpecForm> {
            let mut forms = Vec::new();
            for (key, value) in &template.named {
                if key_base(key) == base {
                    forms.extend(spec_forms(value, &title));
                }
            }
            forms
        };
        let mut pieces: Vec<String> = Vec::new();
        if plural_only {
            pieces.push(String::from("plural only"));
            let singulars = collect_family("sg");
            if !singulars.is_empty() {
                for form in &singulars {
                    self.record_form(&form.text);
                }
                let rendered: Vec<String> =
                    singulars.iter().map(SpecForm::rendered).collect();
                pieces.push(format!("singular {}", rendered.join(" or ")));
            }
            let attributives = collect_family("attr");
            if !attributives.is_empty() {
                for form in &attributives {
                    self.record_form(&form.text);
                }
                let rendered: Vec<String> =
                    attributives.iter().map(SpecForm::rendered).collect();
                pieces.push(format!("attributive {}", rendered.join(" or ")));
            }
        } else if uncountable && plurals.is_empty() {
            pieces.push(String::from("uncountable"));
        } else if uncountable {
            pieces.push(String::from("usually uncountable"));
            pieces.push(plural_piece(&plurals));
        } else if both {
            if plurals.is_empty() {
                plurals.push(SpecForm::plain(regular_plural(&title)));
            }
            pieces.push(String::from("countable and uncountable"));
            pieces.push(plural_piece(&plurals));
        } else if unattested {
            pieces.push(String::from("plural not attested"));
        } else if !plurals.is_empty() {
            pieces.push(plural_piece(&plurals));
        } else if !proper {
            let plural = regular_plural(&title);
            self.record_form(&plural);
            pieces.push(format!("plural [{plural}]"));
        }
        // The presented plural forms, as data (the branches above
        // present `plurals` whenever it is non-empty and neither
        // plural-only nor unattested).
        if !plural_only && !unattested {
            let recorded: Vec<String> =
                plurals.iter().map(|form| form.text.clone()).collect();
            for text in recorded {
                self.record_form(&text);
            }
        }
        if pieces.is_empty() {
            return None;
        }
        Some(format!("({})", pieces.join(", ")))
    }

    /// Derive one en-verb slot's forms for a special indicator; the
    /// multiword `*` variants conjugate the first word only.
    fn verb_indicator_forms(
        &self,
        indicator: &str,
        slot: usize,
        labels: &[String],
    ) -> Vec<SpecForm> {
        let title = self.page_title.to_string();
        let (word, suffix_rest) = if indicator.starts_with('*') {
            match title.split_once(' ') {
                Some((first, rest)) => (first.to_string(), format!(" {rest}")),
                None => (title.clone(), String::new()),
            }
        } else {
            (title.clone(), String::new())
        };
        let base = match indicator {
            "*" => "+",
            "**" => "++",
            "*l" => "+l",
            "*!" => "+!",
            "*'" => "+'",
            other => other,
        };
        let with_labels = |text: String, extra: Option<&str>| -> SpecForm {
            let mut labels = labels.to_vec();
            if let Some(extra) = extra {
                labels.push(String::from(extra));
            }
            SpecForm { text: format!("{text}{suffix_rest}"), labels }
        };
        match (base, slot) {
            ("+" | "^", 0) => vec![with_labels(verb_s_form(&word), None)],
            ("+" | "^", 1) => vec![with_labels(regular_participle(&word), None)],
            ("+" | "^", _) => vec![with_labels(regular_past(&word), None)],
            ("++", 0) => {
                // The module's ++ s-form: a final s/z/x doubles
                // before -es; anything else takes the default.
                let text = if word.ends_with(['s', 'z', 'x']) {
                    doubled(&word, "es")
                } else {
                    verb_s_form(&word)
                };
                vec![with_labels(text, None)]
            }
            ("++", 1) => vec![with_labels(doubled(&word, "ing"), None)],
            ("++", _) => vec![with_labels(doubled(&word, "ed"), None)],
            ("+!", 0) => vec![with_labels(format!("{word}s"), None)],
            ("+!", 1) => vec![with_labels(format!("{word}ing"), None)],
            ("+!", _) => vec![with_labels(format!("{word}ed"), None)],
            ("+'", 0) => vec![with_labels(format!("{word}'s"), None)],
            ("+'", 1) => vec![with_labels(format!("{word}'ing"), None)],
            ("+'", _) => vec![
                with_labels(format!("{word}'d"), None),
                with_labels(format!("{word}'ed"), None),
            ],
            ("+l", 0) => vec![with_labels(verb_s_form(&word), None)],
            ("+l", 1) => vec![
                with_labels(format!("{word}ing"), Some("US")),
                with_labels(doubled(&word, "ing"), Some("UK")),
            ],
            ("+l", _) => vec![
                with_labels(format!("{word}ed"), Some("US")),
                with_labels(doubled(&word, "ed"), Some("UK")),
            ],
            _ => Vec::new(),
        }
    }

    /// The en-verb engine: four slots (pres3sg, presp, past, pastp),
    /// slot-one special indicators becoming later defaults, and the
    /// simple angle-bracket forms; the complex residue audits.
    fn verb_parenthetical(&mut self, template: &Template) -> Option<String> {
        const INDICATORS: [&str; 11] =
            ["+", "^", "++", "+l", "+!", "+'", "*", "**", "*l", "*!", "*'"];
        let title = self.page_title.to_string();
        if template.positional.len() > 4 {
            self.audit("head_arguments_unhandled", template_signature(template));
            return None;
        }
        let first = template.positional.first().cloned().unwrap_or_default();
        let angle = first.find('<').and_then(|open| {
            let close = first[open + 1..].find('>').map(|c| open + 1 + c)?;
            let body = &first[open + 1..close];
            let is_modifier = ["l:", "ll:", "q:", "qq:", "ref:"]
                .iter()
                .any(|prefix| body.starts_with(prefix));
            if is_modifier {
                None
            } else {
                Some((open, close))
            }
        });
        if let Some((open, close)) = angle {
            let after = &first[close + 1..];
            if first.contains("((") || after.contains('<') || template.positional.len() > 1 {
                self.audit("head_arguments_unhandled", template_signature(template));
                return None;
            }
            let before = &first[..open];
            let (word, prefix, rest) = if before.trim().is_empty() {
                match title.split_once(' ') {
                    Some((w, r)) => (w.to_string(), String::new(), format!(" {r}")),
                    None => (title.clone(), String::new(), String::new()),
                }
            } else {
                let word_start = before.rfind(' ').map(|i| i + 1).unwrap_or(0);
                (
                    before[word_start..].to_string(),
                    before[..word_start].to_string(),
                    after.to_string(),
                )
            };
            let body = &first[open + 1..close];
            let slot_specs: Vec<&str> = body.split(',').collect();
            if slot_specs.len() > 4 {
                self.audit("head_arguments_unhandled", template_signature(template));
                return None;
            }
            let mut participle_defective = false;
            let mut slots: Vec<Vec<SpecForm>> = Vec::new();
            for slot in 0..4 {
                let spec = slot_specs.get(slot).copied().unwrap_or("");
                if spec.trim() == "-" {
                    if slot == 3 {
                        participle_defective = true;
                    }
                    slots.push(Vec::new());
                    continue;
                }
                let mut forms: Vec<SpecForm> = Vec::new();
                for alternative in spec.split(':') {
                    let mut text = alternative.trim().to_string();
                    let mut labels: Vec<String> = Vec::new();
                    while text.ends_with(']')
                        && let Some(bracket) = text.rfind('[')
                    {
                        let body = text[bracket + 1..text.len() - 1].to_string();
                        text.truncate(bracket);
                        if let Some(label) = label_text(&body) {
                            labels.push(label);
                        }
                    }
                    let derived = if text.is_empty() || text == "+" || text == "^" {
                        match slot {
                            0 => verb_s_form(&word),
                            1 => regular_participle(&word),
                            _ => regular_past(&word),
                        }
                    } else if text == "~" {
                        word.clone()
                    } else {
                        text
                    };
                    forms.push(SpecForm {
                        text: format!("{prefix}{derived}{rest}"),
                        labels,
                    });
                }
                slots.push(forms);
            }
            if slot_specs.len() < 4 {
                slots[3] = Vec::new();
            }
            return Some(self.verb_pieces(slots, participle_defective));
        }
        let mut participle_defective = false;
        let mut slots: Vec<Vec<SpecForm>> = Vec::new();
        let mut slot_one_indicators: Vec<(String, Vec<String>)> = Vec::new();
        for slot in 0..4 {
            let raw = template.positional.get(slot).cloned().unwrap_or_default();
            let mut forms: Vec<SpecForm> = Vec::new();
            let mut defective = false;
            let parsed = spec_forms(&raw, &title);
            let effective = if parsed.is_empty() && slot < 3 {
                vec![SpecForm::plain(String::from("+"))]
            } else {
                parsed
            };
            for form in effective {
                let text = form.text.as_str();
                if text == "-" {
                    defective = true;
                    continue;
                }
                if slot == 0 && INDICATORS.contains(&text) {
                    slot_one_indicators.push((text.to_string(), form.labels.clone()));
                    forms.extend(self.verb_indicator_forms(text, 0, &form.labels));
                    continue;
                }
                if text == "+" {
                    if slot_one_indicators.is_empty() {
                        forms.extend(self.verb_indicator_forms("+", slot, &form.labels));
                    } else {
                        for (indicator, labels) in slot_one_indicators.clone() {
                            let mut labels = labels;
                            labels.extend(form.labels.clone());
                            forms.extend(
                                self.verb_indicator_forms(&indicator, slot, &labels),
                            );
                        }
                    }
                    continue;
                }
                if text == "^" || (slot > 0 && INDICATORS.contains(&text)) {
                    forms.extend(self.verb_indicator_forms(text, slot, &form.labels));
                    continue;
                }
                if slot == 3 && text == "n" {
                    forms.push(SpecForm {
                        text: format!("{title}n"),
                        labels: form.labels.clone(),
                    });
                    continue;
                }
                if text == "~" {
                    forms.push(SpecForm {
                        text: title.clone(),
                        labels: form.labels.clone(),
                    });
                    continue;
                }
                forms.push(form);
            }
            if defective && forms.is_empty() {
                if slot == 3 {
                    participle_defective = true;
                }
                slots.push(Vec::new());
            } else {
                slots.push(forms);
            }
        }
        // The legacy numbered named args append additional forms to
        // their slots (pres_3sg2=, pres_ptc2=, past2=, past_ptc2=).
        for (key, value) in &template.named {
            let slot = match key_base(key) {
                "pres_3sg" => 0usize,
                "pres_ptc" => 1,
                "past" => 2,
                "past_ptc" => 3,
                _ => continue,
            };
            for form in spec_forms(value, &title) {
                let text = form.text.as_str();
                if text == "-" {
                    continue;
                }
                if INDICATORS.contains(&text) {
                    if text == "+" && !slot_one_indicators.is_empty() {
                        for (indicator, labels) in slot_one_indicators.clone() {
                            let mut labels = labels;
                            labels.extend(form.labels.clone());
                            slots[slot].extend(
                                self.verb_indicator_forms(&indicator, slot, &labels),
                            );
                        }
                    } else {
                        slots[slot]
                            .extend(self.verb_indicator_forms(text, slot, &form.labels));
                    }
                    continue;
                }
                if text == "~" {
                    slots[slot].push(SpecForm {
                        text: title.clone(),
                        labels: form.labels.clone(),
                    });
                    continue;
                }
                slots[slot].push(form);
            }
        }
        Some(self.verb_pieces(slots, participle_defective))
    }

    /// Render the four verb slots to the parenthetical; an absent or
    /// past-equal participle folds into the combined piece, and an
    /// explicitly defective one leaves the past standing alone.
    fn verb_pieces(
        &mut self,
        slots: Vec<Vec<SpecForm>>,
        participle_defective: bool,
    ) -> String {
        for slot in &slots {
            for form in slot {
                let text = form.text.clone();
                self.record_form(&text);
            }
        }
        let join = |forms: &[SpecForm]| -> String {
            forms
                .iter()
                .map(SpecForm::rendered)
                .collect::<Vec<String>>()
                .join(" or ")
        };
        let mut pieces: Vec<String> = Vec::new();
        if !slots[0].is_empty() {
            pieces.push(format!(
                "third-person singular simple present {}",
                join(&slots[0])
            ));
        }
        if !slots[1].is_empty() {
            pieces.push(format!("present participle {}", join(&slots[1])));
        }
        let past_texts: Vec<&str> =
            slots[2].iter().map(|form| form.text.as_str()).collect();
        let participle_texts: Vec<&str> =
            slots[3].iter().map(|form| form.text.as_str()).collect();
        if !slots[2].is_empty() {
            if participle_defective {
                pieces.push(format!("simple past {}", join(&slots[2])));
            } else if participle_texts.is_empty() || participle_texts == past_texts {
                pieces.push(format!(
                    "simple past and past participle {}",
                    join(&slots[2])
                ));
            } else {
                pieces.push(format!("simple past {}", join(&slots[2])));
                pieces.push(format!("past participle {}", join(&slots[3])));
            }
        } else if !slots[3].is_empty() {
            pieces.push(format!("past participle {}", join(&slots[3])));
        }
        format!("({})", pieces.join(", "))
    }

    /// Graded (-er/-est) words of a multiword term per the +first
    /// family of selectors; words split on spaces and hyphens.
    fn graded_words(&self, selector: &str, suffix: &str) -> String {
        let title = self.page_title;
        let mut words: Vec<String> = Vec::new();
        let mut separators: Vec<char> = Vec::new();
        let mut current = String::new();
        for c in title.chars() {
            if c == ' ' || c == '-' {
                words.push(std::mem::take(&mut current));
                separators.push(c);
            } else {
                current.push(c);
            }
        }
        words.push(current);
        let last = words.len() - 1;
        for (index, word) in words.iter_mut().enumerate() {
            let selected = match selector {
                "+first" => index == 0,
                "+second" => index == 1,
                "+first-second" => index <= 1,
                "+first-last" => index == 0 || index == last,
                "+each" => true,
                _ => false,
            };
            if selected && !word.is_empty() {
                *word = graded_form(word, suffix);
            }
        }
        let mut out = String::new();
        for (index, word) in words.iter().enumerate() {
            out.push_str(word);
            if let Some(&separator) = separators.get(index) {
                out.push(separator);
            }
        }
        out
    }

    /// The en-adj/en-adv engine: comparative specs in the
    /// positionals, superlatives derived or given via sup=.
    fn graded_parenthetical(&mut self, template: &Template) -> Option<String> {
        let title = self.page_title.to_string();
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let mut not_comparable = false;
        let mut comparatives: Vec<SpecForm> = Vec::new();
        let mut default_sups: Vec<SpecForm> = Vec::new();
        let selectors = ["+first", "+second", "+first-second", "+first-last", "+each"];
        for raw in &template.positional {
            for form in spec_forms(raw, &title) {
                let labels = form.labels.clone();
                match form.text.as_str() {
                    "-" => not_comparable = true,
                    "?" => return None,
                    "more" | "+" => {
                        comparatives.push(SpecForm {
                            text: format!("more {title}"),
                            labels: labels.clone(),
                        });
                        default_sups.push(SpecForm {
                            text: format!("most {title}"),
                            labels,
                        });
                    }
                    "further" => {
                        for (comparative, superlative) in
                            [("further", "furthest"), ("farther", "farthest")]
                        {
                            comparatives.push(SpecForm {
                                text: format!("{comparative} {title}"),
                                labels: labels.clone(),
                            });
                            default_sups.push(SpecForm {
                                text: format!("{superlative} {title}"),
                                labels: labels.clone(),
                            });
                        }
                    }
                    "better" => {
                        comparatives.push(SpecForm {
                            text: format!("better {title}"),
                            labels: labels.clone(),
                        });
                        default_sups.push(SpecForm {
                            text: format!("best {title}"),
                            labels,
                        });
                    }
                    "er" => {
                        comparatives.push(SpecForm {
                            text: graded_form(&title, "er"),
                            labels: labels.clone(),
                        });
                        default_sups.push(SpecForm {
                            text: graded_form(&title, "est"),
                            labels,
                        });
                    }
                    selector if selectors.contains(&selector) => {
                        comparatives.push(SpecForm {
                            text: self.graded_words(selector, "er"),
                            labels: labels.clone(),
                        });
                        default_sups.push(SpecForm {
                            text: self.graded_words(selector, "est"),
                            labels,
                        });
                    }
                    "~" => {
                        comparatives.push(SpecForm { text: title.clone(), labels });
                    }
                    _ => {
                        if let Some(stem) = form.text.strip_suffix("er") {
                            default_sups.push(SpecForm {
                                text: format!("{stem}est"),
                                labels: labels.clone(),
                            });
                        }
                        comparatives.push(form);
                    }
                }
            }
        }
        let sup_specs: Vec<String> = template
            .named
            .iter()
            .filter(|(key, _)| key_base(key) == "sup")
            .map(|(_, value)| value.clone())
            .collect();
        if comparatives.is_empty() && !not_comparable && sup_specs.is_empty() {
            comparatives.push(SpecForm::plain(format!("more {title}")));
            default_sups.push(SpecForm::plain(format!("most {title}")));
        }
        let mut superlatives: Vec<SpecForm> = Vec::new();
        if sup_specs.is_empty() {
            superlatives = default_sups;
        } else {
            for sup in &sup_specs {
                for form in spec_forms(sup, &title) {
                    let labels = form.labels.clone();
                    match form.text.as_str() {
                        "+" => superlatives.extend(default_sups.clone()),
                        "most" => superlatives.push(SpecForm {
                            text: format!("most {title}"),
                            labels,
                        }),
                        "furthest" => {
                            for superlative in ["furthest", "farthest"] {
                                superlatives.push(SpecForm {
                                    text: format!("{superlative} {title}"),
                                    labels: labels.clone(),
                                });
                            }
                        }
                        "best" => superlatives.push(SpecForm {
                            text: format!("best {title}"),
                            labels,
                        }),
                        "er" | "est" => superlatives.push(SpecForm {
                            text: graded_form(&title, "est"),
                            labels,
                        }),
                        selector if selectors.contains(&selector) => {
                            superlatives.push(SpecForm {
                                text: self.graded_words(selector, "est"),
                                labels,
                            });
                        }
                        _ => superlatives.push(form),
                    }
                }
            }
        }
        let join = |forms: &[SpecForm]| -> String {
            forms
                .iter()
                .map(SpecForm::rendered)
                .collect::<Vec<String>>()
                .join(" or ")
        };
        let mut pieces: Vec<String> = Vec::new();
        if named("componly").is_some() {
            pieces.push(String::from("comparative form only"));
        }
        if named("suponly").is_some() {
            pieces.push(String::from("superlative form only"));
        }
        if not_comparable && comparatives.is_empty() {
            pieces.push(String::from("not comparable"));
        } else if not_comparable {
            pieces.push(String::from("not generally comparable"));
        }
        if !comparatives.is_empty() {
            pieces.push(format!("comparative {}", join(&comparatives)));
            let recorded: Vec<String> =
                comparatives.iter().map(|form| form.text.clone()).collect();
            for text in recorded {
                self.record_form(&text);
            }
        }
        if !superlatives.is_empty()
            && (!not_comparable || !comparatives.is_empty() || !sup_specs.is_empty())
        {
            pieces.push(format!("superlative {}", join(&superlatives)));
            let recorded: Vec<String> =
                superlatives.iter().map(|form| form.text.clone()).collect();
            for text in recorded {
                self.record_form(&text);
            }
        }
        if pieces.is_empty() {
            return None;
        }
        Some(format!("({})", pieces.join(", ")))
    }

    /// Pairwise (name, form) inflections: `head` skips its language
    /// and part-of-speech positionals, en-pron/en-pronoun pair from
    /// the start and append desc= as a trailing note; `or` continues
    /// the previous name and a formless name rides as a note.
    fn pairs_parenthetical(
        &mut self,
        template: &Template,
        skip: usize,
        note: Option<String>,
    ) -> Option<String> {
        let rest = template.positional.get(skip..).unwrap_or(&[]);
        let mut pieces: Vec<(String, Vec<String>)> = Vec::new();
        let mut index = 0usize;
        while index < rest.len() {
            let name = rest[index].trim().to_string();
            let form = rest.get(index + 1).map(|value| value.trim());
            if name == "or" {
                if let Some(form) = form
                    && !form.is_empty()
                    && let Some(last) = pieces.last_mut()
                {
                    self.record_form(form);
                    let anchored = self.anchored_argument(form);
                    last.1.push(anchored);
                }
            } else if !name.is_empty() {
                let forms = match form {
                    Some(form) if !form.is_empty() => {
                        self.record_form(form);
                        vec![self.anchored_argument(form)]
                    }
                    _ => Vec::new(),
                };
                pieces.push((name, forms));
            }
            index += 2;
        }
        let mut rendered: Vec<String> = pieces
            .into_iter()
            .map(|(name, forms)| {
                if forms.is_empty() {
                    name
                } else {
                    format!("{name} {}", forms.join(" or "))
                }
            })
            .collect();
        if let Some(note) = note {
            let text = self.argument_text(&note);
            if !text.is_empty() {
                rendered.push(text);
            }
        }
        if rendered.is_empty() {
            return None;
        }
        Some(format!("({})", rendered.join(", ")))
    }

    /// A POS section's head template, dispatched by family. The
    /// families grow audit-driven: arguments outside a family's
    /// grammar audit with the full signature and contribute nothing.
    fn head_line(&mut self, template: &Template) -> HeadLine {
        const IGNORED_BASES: [&str; 22] = [
            "id", "sort", "pagename", "sc", "sccat", "g", "tr", "ts",
            "autotrinfl", "nolink", "nolinkhead", "splithyph", "nosplithyph",
            "hyphspace", "nosuffix", "nomultiwordcat", "nopalindromecat",
            "noposcat", "nogendercat", "cat", "angle_bracket", "plqual",
        ];
        const FAMILY_BASES: [&str; 14] = [
            "head", "def", "the", "abbr", "sg", "attr", "sup", "componly",
            "suponly", "pres_3sg", "pres_ptc", "past", "past_ptc", "desc",
        ];
        let title = self.page_title.to_string();
        let normalized = normalize_numbered(template);
        let template = &normalized;
        for (key, _) in &template.named {
            let key = key.as_str();
            let base = key_base(key);
            let form_meta = key.starts_with('f')
                && key.chars().nth(1).is_some_and(|c| c.is_ascii_digit());
            if !IGNORED_BASES.contains(&base)
                && !FAMILY_BASES.contains(&base)
                && !form_meta
            {
                self.audit("head_arguments_unhandled", template_signature(template));
                return HeadLine::default();
            }
        }
        let named = |key: &str| -> Option<&str> {
            template
                .named
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let mut headword = named("head")
            .filter(|raw| *raw != "?")
            .map(|raw| {
                // Headword lines carry no anchors: flatten to plain
                // display text.
                plain_anchor_text(&self.argument_text(&raw.replace("\\,", ",")))
            })
            .filter(|text| !text.is_empty());
        match named("def").or(named("the")) {
            Some("1") => {
                let base = headword.unwrap_or_else(|| title.clone());
                headword = Some(format!("the {base}"));
            }
            Some("~") => {
                let base = headword.unwrap_or_else(|| title.clone());
                headword = Some(format!("(the) {base}"));
            }
            _ => {}
        }
        let parenthetical = match template.name.as_str() {
            "en-noun" => self.noun_parenthetical(template, false),
            "en-prop" | "en-proper noun" | "en-proper-noun" | "en-propn" => {
                self.noun_parenthetical(template, true)
            }
            "en-verb" => self.verb_parenthetical(template),
            "en-adj" | "en-adjective" | "en-adv" | "en-adverb" => {
                self.graded_parenthetical(template)
            }
            "head" => self.pairs_parenthetical(template, 2, None),
            "en-pron" | "en-pronoun" => {
                let desc = named("desc").map(str::to_string);
                self.pairs_parenthetical(template, 0, desc)
            }
            "en-head" => {
                // Positional 1 is the part of speech; nothing derives.
                if template.positional.len() <= 1 {
                    None
                } else {
                    self.audit("head_template_unhandled", template_signature(template));
                    None
                }
            }
            _ => {
                // Any other en-* head: the bare and head=-only forms
                // carry no inflections; positionals are the residue.
                if template.positional.is_empty() {
                    None
                } else {
                    self.audit("head_template_unhandled", template_signature(template));
                    None
                }
            }
        };
        let parenthetical = match (parenthetical, named("abbr")) {
            (parenthetical, None) => parenthetical,
            (parenthetical, Some(abbr)) => {
                let forms = spec_forms(abbr, &title);
                for form in &forms {
                    let text = form.text.clone();
                    self.record_form(&text);
                }
                let rendered: Vec<String> =
                    forms.iter().map(SpecForm::rendered).collect();
                let piece = format!("abbreviation {}", rendered.join(" or "));
                Some(match parenthetical {
                    Some(inner) => {
                        format!("({}, {piece})", &inner[1..inner.len() - 1])
                    }
                    None => format!("({piece})"),
                })
            }
        };
        HeadLine { headword, parenthetical }
    }
}

/// One section of the English subtree: its source heading level, its
/// name, and its blocks.
struct Section<'a> {
    level: usize,
    name: String,
    blocks: Vec<&'a Block>,
}

/// Split the English h2 subtree into its sections; the English
/// heading itself drops (the document IS the English entry).
fn english_sections(blocks: &[Block]) -> Vec<Section<'_>> {
    let mut sections: Vec<Section<'_>> = Vec::new();
    let mut inside = false;
    for block in blocks {
        match block {
            Block::Heading { level, content } if *level == 2 => {
                let name = heading_name(content);
                inside = name == "English";
                continue;
            }
            Block::Heading { level, content } if inside => {
                sections.push(Section {
                    level: *level as usize,
                    name: heading_name(content),
                    blocks: Vec::new(),
                });
            }
            _ if inside => {
                if let Some(section) = sections.last_mut() {
                    section.blocks.push(block);
                }
            }
            _ => {}
        }
    }
    sections
}

/// A heading's plain-text name.
fn heading_name(content: &[Inline]) -> String {
    content
        .iter()
        .filter_map(|inline| match inline {
            Inline::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>()
        .trim()
        .to_string()
}

/// Whether a section name is a POS (Etymology N handled separately).
fn is_pos(name: &str) -> bool {
    POS_NAMES.contains(&name)
}

/// Render one page's English subtree into the renderer. Output
/// levels are fixed by section KIND, not source depth (TheUser's
/// cleanup is the ruling): every top section is h2 - Etymology N
/// included - POS headwords h3, and POS subsections h3, so single-
/// and multi-etymology pages read uniformly.
fn render_page(renderer: &mut Renderer<'_>, blocks: &[Block]) -> BiquestResult<()> {
    let sections = english_sections(blocks);
    let mut index = 0usize;
    while index < sections.len() {
        let section = &sections[index];
        let out_level = 2usize;
        let name = section.name.as_str();
        if DROP_SECTIONS.contains(&name) {
            renderer.audit("section_dropped", name.to_string());
            index += 1;
            continue;
        }
        if name == "Pronunciation" {
            render_pronunciation(renderer, section, out_level);
            index += 1;
            continue;
        }
        if name == "Anagrams" {
            render_anagrams(renderer, section, out_level);
            index += 1;
            continue;
        }
        if name == "Alternative forms" || LIST_SECTIONS.contains(&name) {
            render_list_section(renderer, section, out_level);
            index += 1;
            continue;
        }
        if is_pos(name) {
            render_pos(renderer, section, out_level);
            index += 1;
            // The POS's own deeper subsections follow it in the flat
            // walk; list sections render, Translations drops.
            while index < sections.len() && sections[index].level > section.level {
                let sub = &sections[index];
                let sub_level = 3usize;
                let sub_name = sub.name.as_str();
                if DROP_SECTIONS.contains(&sub_name) {
                    renderer.audit("section_dropped", sub_name.to_string());
                } else if LIST_SECTIONS.contains(&sub_name)
                    || sub_name == "Alternative forms"
                {
                    render_list_section(renderer, sub, sub_level);
                } else if sub_name == "Usage notes" {
                    render_prose(renderer, sub, sub_level);
                } else {
                    renderer.audit("section_unhandled", sub_name.to_string());
                }
                index += 1;
            }
            continue;
        }
        // Etymology (N), Usage notes, and anything prose-shaped.
        render_prose(renderer, section, out_level);
        index += 1;
    }
    Ok(())
}

/// A prose section: paragraphs as prose, list items as dash lines.
fn render_prose(renderer: &mut Renderer<'_>, section: &Section<'_>, level: usize) {
    renderer.heading(level, &section.name);
    for block in &section.blocks {
        match block {
            Block::Paragraph { content } => {
                let text = renderer.inline_text(content);
                if !text.is_empty() {
                    renderer.push_prose(text);
                }
            }
            Block::ListItem { markers, content } => {
                let text = renderer.inline_text(content);
                if !text.is_empty() {
                    let depth = markers.len().saturating_sub(1);
                    renderer.push_item(depth, text);
                }
            }
            Block::Table { .. } => renderer.audit("table_dropped", section.name.clone()),
            _ => {}
        }
    }
}

/// The pronunciation section: IPA lines kept, plumbing dropped.
fn render_pronunciation(
    renderer: &mut Renderer<'_>,
    section: &Section<'_>,
    level: usize,
) {
    let mut items: Vec<String> = Vec::new();
    for block in &section.blocks {
        let Block::ListItem { content, .. } = block else { continue };
        for inline in content {
            let Inline::Template(template) = inline else { continue };
            if template.name == "IPA" {
                items.push(renderer.ipa_text(template));
            } else {
                renderer.audit("pronunciation_dropped", template.name.clone());
            }
        }
    }
    if items.is_empty() {
        renderer.audit("section_empty", section.name.clone());
        return;
    }
    renderer.heading(level, &section.name);
    for text in items {
        renderer.push_item(0, text);
    }
}

/// The anagrams section: the anagrams template's word items anchored
/// (the alphagram argument drops; the set is derivable in-house).
fn render_anagrams(renderer: &mut Renderer<'_>, section: &Section<'_>, level: usize) {
    let mut items: Vec<String> = Vec::new();
    for block in &section.blocks {
        let Block::ListItem { content, .. } = block else { continue };
        for inline in content {
            let Inline::Template(template) = inline else { continue };
            if template.name == "anagrams" {
                for word in template.positional.iter().skip(1) {
                    if !word.is_empty() {
                        items.push(format!("[{word}]"));
                    }
                }
            }
        }
    }
    if items.is_empty() {
        renderer.audit("section_empty", section.name.clone());
        return;
    }
    renderer.heading(level, &section.name);
    for text in items {
        renderer.push_item(0, text);
    }
}

/// A relational list section: column-template items and plain list
/// items, every entry an anchor.
fn render_list_section(
    renderer: &mut Renderer<'_>,
    section: &Section<'_>,
    level: usize,
) {
    let mut items: Vec<String> = Vec::new();
    for block in &section.blocks {
        match block {
            Block::ListItem { content, .. } => {
                let text = renderer.inline_text(content);
                if !text.is_empty() {
                    items.push(text);
                }
            }
            Block::Paragraph { content } => {
                for inline in content {
                    let Inline::Template(template) = inline else { continue };
                    if template.name.starts_with("col")
                        || template.name == "coi"
                        || template.name == "co"
                    {
                        for item in template.positional.iter().skip(1) {
                            let anchored = renderer.anchored_argument(item);
                            if !anchored.is_empty() {
                                items.push(anchored);
                            }
                        }
                    } else if is_silent_template(template.name.as_str()) {
                        // Metadata in list-section paragraph position.
                    } else {
                        renderer.audit("template_unhandled", template.name.clone());
                    }
                }
            }
            _ => {}
        }
    }
    if items.is_empty() {
        renderer.audit("section_empty", section.name.clone());
        return;
    }
    renderer.heading(level, &section.name);
    for text in items {
        renderer.push_item(0, text);
    }
}

/// A POS section: the heading, the headword with its derived
/// inflection parenthetical, the numbered glosses with their
/// sense-scoped synonyms and citations.
fn render_pos(renderer: &mut Renderer<'_>, section: &Section<'_>, level: usize) {
    renderer.heading(level, &section.name);
    renderer.blank();
    let mut head_line = HeadLine::default();
    let mut head_seen = false;
    for block in &section.blocks {
        if let Block::Paragraph { content } = block {
            for inline in content {
                let Inline::Template(template) = inline else { continue };
                let name = template.name.as_str();
                if !head_seen && is_head_family(name) {
                    head_seen = true;
                    head_line = renderer.head_line(template);
                } else if matches!(name, "wp" | "swp" | "wikipedia" | "slim-wp") {
                    // The word-to-article signal, kept in the audit
                    // for the spidering service.
                    renderer.audit("wikipedia_pointer", template_signature(template));
                } else if is_silent_template(name) {
                    // Metadata carries no document meaning in
                    // paragraph position either.
                } else {
                    renderer.audit("template_unhandled", template.name.clone());
                }
            }
        }
    }
    let headword = head_line
        .headword
        .unwrap_or_else(|| renderer.page_title.to_string());
    renderer.push_head_word(level + 1, headword);
    if let Some(parenthetical) = head_line.parenthetical {
        renderer.push_head_line(parenthetical);
    }
    // Gloss lines nest: `##`/`###` are subsenses of the sense above
    // (glacis holds its military senses two deep), so a per-depth
    // counter stack numbers them and the indentation carries the
    // nesting. Sense lines (`:`) and quote lines (`*`) attach at any
    // depth.
    let mut counters: Vec<usize> = Vec::new();
    let mut synonyms: Vec<String> = Vec::new();
    let mut antonyms: Vec<String> = Vec::new();
    for block in &section.blocks {
        let Block::ListItem { markers, content } = block else { continue };
        let depth = markers.chars().take_while(|c| *c == '#').count();
        let suffix = &markers[depth..];
        if depth == 0 {
            let text = renderer.inline_text(content);
            renderer.audit("list_line_dropped", text);
            continue;
        }
        if suffix.is_empty() {
            counters.truncate(depth);
            while counters.len() < depth {
                counters.push(0);
            }
            counters[depth - 1] += 1;
            let number = counters[depth - 1];
            let text = renderer.inline_text(content);
            if !text.is_empty() {
                renderer.senses.push((section.name.clone(), text.clone()));
            }
            renderer.push_sense(depth, number, text);
            continue;
        }
        if suffix.chars().all(|c| c == ':') {
            for inline in content {
                let Inline::Template(template) = inline else { continue };
                match template.name.as_str() {
                    "syn" | "synonyms" => {
                        collect_sense_items(renderer, template, &mut synonyms)
                    }
                    "ant" | "antonyms" => {
                        collect_sense_items(renderer, template, &mut antonyms)
                    }
                    "ux" | "uxi" | "usex" => {
                        let text = template
                            .positional
                            .get(1)
                            .cloned()
                            .unwrap_or_default();
                        let rendered = renderer.argument_text(&text);
                        if !rendered.is_empty() {
                            renderer.push_quote(depth, format!("\"{rendered}\""));
                        }
                    }
                    name if is_silent_template(name) => {}
                    other => renderer.audit("sense_line_dropped", other.to_string()),
                }
            }
            continue;
        }
        if suffix.contains('*') {
            for inline in content {
                let Inline::Template(template) = inline else { continue };
                if template.name.starts_with("quote-") {
                    if let Some(citation) = renderer.citation_text(template) {
                        renderer.push_quote(depth, citation);
                    }
                } else if matches!(template.name.as_str(), "ux" | "uxi" | "usex") {
                    let text = template
                        .positional
                        .get(1)
                        .cloned()
                        .unwrap_or_default();
                    let rendered = renderer.argument_text(&text);
                    if !rendered.is_empty() {
                        renderer.push_quote(depth, format!("\"{rendered}\""));
                    }
                } else {
                    renderer.audit("quote_line_dropped", template.name.clone());
                }
            }
            continue;
        }
        let text = renderer.inline_text(content);
        renderer.audit("list_line_dropped", text);
    }
    if !synonyms.is_empty() {
        renderer.heading(level + 1, "Synonyms");
        for text in synonyms {
            renderer.push_item(0, text);
        }
    }
    if !antonyms.is_empty() {
        renderer.heading(level + 1, "Antonyms");
        for text in antonyms {
            renderer.push_item(0, text);
        }
    }
}

/// The sense-relational items: word positionals past the language,
/// their angle-bracket inline qualifiers stripped; a markup-bearing
/// item re-renders rather than leaking.
fn collect_sense_items(
    renderer: &mut Renderer<'_>,
    template: &Template,
    items: &mut Vec<String>,
) {
    for raw in template.positional.iter().skip(1) {
        let word = match raw.find('<') {
            Some(cut) => &raw[..cut],
            None => raw.as_str(),
        };
        let word = word.trim();
        if word.is_empty() {
            continue;
        }
        let anchored = renderer.anchored_argument(word);
        if !anchored.is_empty() {
            items.push(anchored);
        }
    }
}

/// Keep a word's renderable pages (ns0, not redirects) and order
/// them: the word's own casing first, the capitalized form second,
/// the rest title-sorted. The first page's title becomes the h1.
pub(crate) fn order_word_pages(word: &str, pages: &mut Vec<WikiPage>) {
    pages.retain(|page| page.ns == 0 && page.redirect.is_none());
    let capitalized: String = {
        let mut chars = word.chars();
        match chars.next() {
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            None => String::new(),
        }
    };
    pages.sort_by_key(|page| {
        let rank = if page.title == word {
            0
        } else if page.title == capitalized {
            1
        } else {
            2
        };
        (rank, page.title.clone())
    });
}

/// One page's derived data: what the render presents, as data - the
/// provenance derivation's per-page unit.
pub(crate) struct PageDerivation {
    /// Whether the page carries an English h2 section.
    pub(crate) has_english: bool,
    /// Whether the English subtree carries the `no entry` soft
    /// redirect - the wiki's own declaration that no English entry
    /// exists, so the title is not a vocabulary word.
    pub(crate) no_entry: bool,
    /// The head engines' inflected forms, render order.
    pub(crate) forms: Vec<String>,
    /// (pos section name, rendered gloss) per numbered sense line.
    pub(crate) senses: Vec<(String, String)>,
    /// Definitional form-of lemma targets, render order.
    pub(crate) form_of_lemmas: Vec<String>,
}

/// Whether a block list carries the English h2.
fn has_english_section(blocks: &[Block]) -> bool {
    blocks.iter().any(|block| {
        matches!(block, Block::Heading { level: 2, content }
            if heading_name(content) == "English")
    })
}

/// Whether the English subtree carries the `no entry` template. It
/// sits directly under the English heading, before any section, so
/// the section walk never sees it - this scan does.
fn english_no_entry(blocks: &[Block]) -> bool {
    let mut inside = false;
    for block in blocks {
        match block {
            Block::Heading { level: 2, content } => {
                inside = heading_name(content) == "English";
            }
            Block::Paragraph { content } | Block::ListItem { content, .. }
                if inside =>
            {
                let hit = content.iter().any(|inline| {
                    matches!(inline, Inline::Template(template)
                        if template.name == "no entry"
                            || template.name == "noentry")
                });
                if hit {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Derive one page's data through the document renderer, so the data
/// equals what the rendered document presents.
pub(crate) fn derive_page(page: &WikiPage) -> BiquestResult<PageDerivation> {
    let blocks = parse_blocks(&page.text)?;
    let has_english = has_english_section(&blocks);
    let no_entry = has_english && english_no_entry(&blocks);
    let mut renderer = Renderer::new(&page.title);
    if has_english && !no_entry {
        render_page(&mut renderer, &blocks)?;
    }
    Ok(PageDerivation {
        has_english,
        no_entry,
        forms: renderer.forms,
        senses: renderer.senses,
        form_of_lemmas: renderer.form_of_lemmas,
    })
}

/// Render a word's page set into its one markdown document: the
/// pages in given order, the first page's title as the h1.
pub(crate) fn render_word_document(pages: &[WikiPage]) -> BiquestResult<Document> {
    snafu::ensure_whatever!(!pages.is_empty(), "no pages to render");
    let mut lines: Vec<String> = vec![format!("# {}", pages[0].title)];
    let mut audit: Vec<AuditRow> = Vec::new();
    for page in pages {
        let blocks = parse_blocks(&page.text)?;
        let mut page_renderer = Renderer::new(&page.title);
        render_page(&mut page_renderer, &blocks)?;
        if !page_renderer.lines.is_empty() {
            if matches!(lines.last(), Some(last) if !last.is_empty()) {
                lines.push(String::new());
            }
            lines.extend(page_renderer.lines);
        }
        audit.append(&mut page_renderer.audit);
    }
    let mut markdown = lines.join("\n");
    markdown.push('\n');
    Ok(Document { markdown, audit })
}
