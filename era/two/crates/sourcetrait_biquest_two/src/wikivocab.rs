//! The vendored third-party vocabularies the renderer resolves
//! against: language codes, inflection tags, and place data. Each is
//! carried data (never hardcoded), parsed once, dump-versioned.
use crate::*;

/// Third-party data carried in (TheUser's ruling): the language-code
/// vocabulary, vendored under a dump-date version directory.
const LANGUAGES_DATA: &str = include_str!("../data/languages/20260801/languages.nuon");
/// The inflection-tag vocabulary (Module:form of data).
const TAGS_DATA: &str = include_str!("../data/form_of/20260801/tags.nuon");
/// The place vocabulary (Module:place placetypes and locations).
const PLACE_DATA: &str = include_str!("../data/place/20260801/place.nuon");

fn parse_rows(payload: &str, key: &str) -> Vec<harness::nu::Value> {
    let value = harness::nu::from_nuon_text(payload).expect("vendored data parses");
    let record = value.as_record().expect("vendored data is a record");
    record
        .get(key)
        .unwrap_or_else(|| panic!("vendored data carries {key}"))
        .as_list()
        .expect("vendored key is a table")
        .to_vec()
}

fn row_str(row: &harness::nu::Value, key: &str) -> String {
    let record = row.as_record().expect("vendored row is a record");
    field_str(record, key).unwrap_or_else(|_| panic!("vendored row carries {key}"))
}

fn row_bool(row: &harness::nu::Value, key: &str) -> bool {
    row.as_record()
        .ok()
        .and_then(|record| record.get(key))
        .and_then(|value| value.as_bool().ok())
        .unwrap_or(false)
}

// ------------------------------------------------------------- languages

/// The code-to-canonical-name map: languages, etymology-only
/// languages, and families in one table.
pub(crate) struct LanguageTable {
    names: HashMap<String, String>,
}

impl LanguageTable {
    /// A code's canonical display name; families answer too ("gem"
    /// names "Germanic", rendered with " languages" by the caller
    /// where the family sense needs saying - the site renders bare).
    pub(crate) fn name(&self, code: &str) -> Option<&str> {
        self.names.get(code).map(String::as_str)
    }
}

pub(crate) fn language_table() -> &'static LanguageTable {
    static TABLE: std::sync::OnceLock<LanguageTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut names = HashMap::new();
        for row in parse_rows(LANGUAGES_DATA, "codes") {
            names.insert(row_str(&row, "code"), row_str(&row, "name"));
        }
        LanguageTable { names }
    })
}

// ------------------------------------------------------------------ tags

/// The inflection-tag vocabulary: canonical tags with display forms
/// and shortcut expansions.
pub(crate) struct TagTable {
    displays: HashMap<String, String>,
    shortcuts: HashMap<String, Vec<String>>,
}

impl TagTable {
    /// Resolve one raw tag token to its display tokens: shortcuts
    /// expand (recursively, list-valued ones to several tags), a
    /// `//` multipart joins its parts' displays with a slash, a
    /// known tag renders its display, and anything else is verbatim
    /// (the documented spell-it-out convention).
    pub(crate) fn resolve(&self, token: &str) -> Vec<String> {
        self.resolve_depth(token, 0)
    }

    fn resolve_depth(&self, token: &str, depth: usize) -> Vec<String> {
        if depth > 8 {
            return vec![token.to_string()];
        }
        if token.contains("//") {
            let parts: Vec<String> = token
                .split("//")
                .map(|part| self.resolve_depth(part, depth + 1).join(" "))
                .collect();
            return vec![parts.join("/")];
        }
        if let Some(expansion) = self.shortcuts.get(token) {
            let mut out = Vec::new();
            for element in expansion {
                out.extend(self.resolve_depth(element, depth + 1));
            }
            return out;
        }
        if let Some(display) = self.displays.get(token) {
            return vec![display.clone()];
        }
        vec![token.to_string()]
    }
}

pub(crate) fn tag_table() -> &'static TagTable {
    static TABLE: std::sync::OnceLock<TagTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut displays = HashMap::new();
        for row in parse_rows(TAGS_DATA, "tags") {
            displays.insert(row_str(&row, "name"), row_str(&row, "display"));
        }
        let mut shortcuts = HashMap::new();
        for row in parse_rows(TAGS_DATA, "shortcuts") {
            let expansion: Vec<String> = row
                .as_record()
                .expect("shortcut row is a record")
                .get("expansion")
                .expect("shortcut row carries expansion")
                .as_list()
                .expect("expansion is a list")
                .iter()
                .map(|item| item.as_str().expect("expansion element").to_string())
                .collect();
            shortcuts.insert(row_str(&row, "shortcut"), expansion);
        }
        TagTable { displays, shortcuts }
    })
}

// ----------------------------------------------------------------- place

/// One placetype's render keys.
#[derive(Debug, Clone, Default)]
pub(crate) struct PlacetypeRow {
    pub(crate) preposition: String,
    pub(crate) fallback: String,
    pub(crate) affix_type: String,
    pub(crate) affix: String,
    pub(crate) article: String,
    pub(crate) holonym_use_the: bool,
}

/// One known location's article and alias data. display_expand
/// canonicalizes the display to the alias target; display_as
/// displays a specific string; a bare alias categorizes only and
/// keeps its written display.
#[derive(Debug, Clone, Default)]
pub(crate) struct LocationRow {
    pub(crate) the: bool,
    pub(crate) alias_of: String,
    pub(crate) display_expand: bool,
    pub(crate) display_as: String,
}

/// The place vocabulary: aliases, qualifiers, placetype render
/// keys, and location articles.
pub(crate) struct PlaceTable {
    aliases: HashMap<String, String>,
    /// name -> (display, article: default|the|none)
    qualifiers: HashMap<String, (String, String)>,
    placetypes: HashMap<String, PlacetypeRow>,
    placename_articles: HashSet<(String, String)>,
    /// (placetype-or-*, kind: prefix|suffix|contains, text)
    the_patterns: Vec<(String, String, String)>,
    locations: HashMap<String, LocationRow>,
}

impl PlaceTable {
    /// A placetype alias's full form, or the input.
    pub(crate) fn expand_alias<'a>(&'a self, placetype: &'a str) -> &'a str {
        self.aliases
            .get(placetype)
            .map(String::as_str)
            .unwrap_or(placetype)
    }

    /// A recognized qualifier's display and article override.
    pub(crate) fn qualifier(&self, word: &str) -> Option<(&str, &str)> {
        self.qualifiers
            .get(word)
            .map(|(display, article)| (display.as_str(), article.as_str()))
    }

    /// A placetype property resolved through the fallback chain:
    /// qualifiers split off the left first, then fallbacks walk.
    pub(crate) fn placetype_resolved(&self, name: &str) -> PlacetypeRow {
        let mut current = self.expand_alias(name).to_string();
        // Strip recognized qualifiers to find the reduced placetype.
        loop {
            if self.placetypes.contains_key(&current) {
                break;
            }
            let Some((first, rest)) = current.split_once(' ') else { break };
            if self.qualifier(first).is_none() {
                break;
            }
            current = self.expand_alias(rest).to_string();
        }
        let mut resolved = PlacetypeRow::default();
        let mut cursor = current;
        for _ in 0..8 {
            let Some(row) = self.placetypes.get(&cursor) else { break };
            if resolved.preposition.is_empty() {
                resolved.preposition = row.preposition.clone();
            }
            if resolved.article.is_empty() {
                resolved.article = row.article.clone();
            }
            if resolved.affix_type.is_empty() {
                resolved.affix_type = row.affix_type.clone();
                resolved.affix = row.affix.clone();
            }
            resolved.holonym_use_the = resolved.holonym_use_the || row.holonym_use_the;
            if row.fallback.is_empty() {
                break;
            }
            cursor = row.fallback.clone();
        }
        resolved
    }

    pub(crate) fn location(&self, name: &str) -> Option<&LocationRow> {
        self.locations.get(name)
    }

    /// Whether "the" precedes a holonym: its location row, the
    /// per-placetype article rows, the name patterns, or the
    /// placetype's holonym_use_the.
    pub(crate) fn holonym_takes_the(&self, placetype: &str, name: &str) -> bool {
        if let Some(row) = self.locations.get(name)
            && row.the
        {
            return true;
        }
        if self
            .placename_articles
            .contains(&(placetype.to_string(), name.to_string()))
        {
            return true;
        }
        for (pattern_type, kind, text) in &self.the_patterns {
            if pattern_type != "*" && pattern_type != placetype {
                continue;
            }
            let hit = match kind.as_str() {
                "prefix" => name.starts_with(text.as_str()),
                "suffix" => name.ends_with(text.as_str()),
                _ => name.contains(text.as_str()),
            };
            if hit {
                return true;
            }
        }
        self.placetype_resolved(placetype).holonym_use_the
    }
}

pub(crate) fn place_table() -> &'static PlaceTable {
    static TABLE: std::sync::OnceLock<PlaceTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut aliases = HashMap::new();
        for row in parse_rows(PLACE_DATA, "aliases") {
            aliases.insert(row_str(&row, "alias"), row_str(&row, "full"));
        }
        let mut qualifiers = HashMap::new();
        for row in parse_rows(PLACE_DATA, "qualifiers") {
            qualifiers.insert(
                row_str(&row, "name"),
                (row_str(&row, "display"), row_str(&row, "article")),
            );
        }
        let mut placetypes = HashMap::new();
        for row in parse_rows(PLACE_DATA, "placetypes") {
            placetypes.insert(
                row_str(&row, "name"),
                PlacetypeRow {
                    preposition: row_str(&row, "preposition"),
                    fallback: row_str(&row, "fallback"),
                    affix_type: row_str(&row, "affix_type"),
                    affix: row_str(&row, "affix"),
                    article: row_str(&row, "article"),
                    holonym_use_the: row_bool(&row, "holonym_use_the"),
                },
            );
        }
        let mut placename_articles = HashSet::new();
        for row in parse_rows(PLACE_DATA, "placename_articles") {
            placename_articles.insert((row_str(&row, "placetype"), row_str(&row, "name")));
        }
        let mut the_patterns = Vec::new();
        for row in parse_rows(PLACE_DATA, "the_patterns") {
            the_patterns.push((
                row_str(&row, "placetype"),
                row_str(&row, "kind"),
                row_str(&row, "text"),
            ));
        }
        let mut locations = HashMap::new();
        for row in parse_rows(PLACE_DATA, "locations") {
            locations.insert(
                row_str(&row, "name"),
                LocationRow {
                    the: row_bool(&row, "the"),
                    alias_of: row_str(&row, "alias_of"),
                    display_expand: row_bool(&row, "display_expand"),
                    display_as: row_str(&row, "display_as"),
                },
            );
        }
        PlaceTable {
            aliases,
            qualifiers,
            placetypes,
            placename_articles,
            the_patterns,
            locations,
        }
    })
}

// ------------------------------------------------------------- articles

/// The indefinite article for a phrase: "an" before a vowel sound,
/// with the u-as-"you" words and "one" as "a".
pub(crate) fn indefinite_article(phrase: &str) -> &'static str {
    let first_word = phrase.split_whitespace().next().unwrap_or("");
    let lower = first_word.to_lowercase();
    for a_start in ["uni", "unio", "use", "one", "eu"] {
        if lower.starts_with(a_start) {
            return "a";
        }
    }
    match lower.chars().next() {
        Some('a') | Some('e') | Some('i') | Some('o') | Some('u') => "an",
        _ => "a",
    }
}

/// The first letter capitalized.
pub(crate) fn ucfirst(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// The first letter lowercased (the nocap rendering).
pub(crate) fn lcfirst(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Serial-join with a conjunction: "a", "a and b", "a, b and c".
pub(crate) fn serial_join(items: &[String], conjunction: &str) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 => format!("{} {conjunction} {}", items[0], items[1]),
        _ => format!(
            "{} {conjunction} {}",
            items[..items.len() - 1].join(", "),
            items[items.len() - 1]
        ),
    }
}

/// Split a term off its inline `<mod:value>` modifiers (the
/// col/syn/inflection-of convention); nested `<<...>>` label syntax
/// stays inside its modifier's value.
pub(crate) fn term_modifiers(raw: &str) -> (String, Vec<(String, String)>) {
    let mut base = String::new();
    let mut modifiers = Vec::new();
    let bytes = raw.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'<' {
            // A modifier starts `<name:`; anything else keeps the
            // angle bracket as term text.
            let name_end = bytes[index + 1..]
                .iter()
                .take_while(|b| b.is_ascii_alphanumeric())
                .count();
            let is_modifier = name_end > 0
                && index + 1 + name_end < bytes.len()
                && bytes[index + 1 + name_end] == b':';
            if is_modifier {
                let name = raw[index + 1..index + 1 + name_end].to_string();
                let mut depth = 1usize;
                let mut cursor = index + 1 + name_end + 1;
                while cursor < bytes.len() && depth > 0 {
                    match bytes[cursor] {
                        b'<' => depth += 1,
                        b'>' => depth -= 1,
                        _ => {}
                    }
                    if depth == 0 {
                        break;
                    }
                    cursor += 1;
                }
                let value = raw[index + 1 + name_end + 1..cursor.min(raw.len())].to_string();
                modifiers.push((name, value));
                index = (cursor + 1).min(raw.len());
                continue;
            }
            // A bare `<mod>` flag (like <unc>) with no colon.
            let is_flag = name_end > 0
                && index + 1 + name_end < bytes.len()
                && bytes[index + 1 + name_end] == b'>';
            if is_flag {
                let name = raw[index + 1..index + 1 + name_end].to_string();
                modifiers.push((name, String::new()));
                index = index + 1 + name_end + 1;
                continue;
            }
        }
        let c = raw[index..].chars().next().expect("in-bounds");
        base.push(c);
        index += c.len_utf8();
    }
    (base.trim().to_string(), modifiers)
}
