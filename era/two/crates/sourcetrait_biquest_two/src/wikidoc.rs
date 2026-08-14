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
const POS_NAMES: [&str; 24] = [
    "Adjective", "Adverb", "Article", "Conjunction", "Contraction",
    "Determiner", "Infix", "Interjection", "Letter", "Noun", "Numeral",
    "Particle", "Phrase", "Postposition", "Prefix", "Preposition",
    "Pronoun", "Proper noun", "Proverb", "Punctuation mark", "Suffix",
    "Symbol", "Verb", "Interfix",
];

/// Sections dropped whole, audited.
const DROP_SECTIONS: [&str; 6] = [
    "Translations", "Descendants", "References", "Further reading",
    "See also", "Statistics",
];

/// POS subsections rendered as anchored lists.
const LIST_SECTIONS: [&str; 6] = [
    "Synonyms", "Antonyms", "Derived terms", "Related terms",
    "Collocations", "Coordinate terms",
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

/// The regular English plural (and third-person singular): +es after
/// a sibilant ending, consonant-y to -ies, else +s.
pub(crate) fn regular_plural(word: &str) -> String {
    let lower = word.to_lowercase();
    if ["s", "x", "z", "ch", "sh"].iter().any(|end| lower.ends_with(end)) {
        return format!("{word}es");
    }
    if let Some(stem) = consonant_y_stem(word) {
        return format!("{stem}ies");
    }
    format!("{word}s")
}

/// Whether a lemma has the doubling shape: consonants, one vowel,
/// one final consonant not in w/x/y/h (the en-verb C*VC rule).
fn doubles_final(word: &str) -> bool {
    let lower = word.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let Some((&last, body)) = chars.split_last() else { return false };
    if !last.is_ascii_alphabetic() || "aeiouwxyh".contains(last) {
        return false;
    }
    let Some((&vowel, head)) = body.split_last() else { return false };
    if !"aeiou".contains(vowel) {
        return false;
    }
    head.iter().all(|&c| c.is_ascii_alphabetic() && !"aeiou".contains(c))
}

/// The word with its final consonant doubled before a suffix.
fn doubled(word: &str, suffix: &str) -> String {
    match word.chars().last() {
        Some(last) => format!("{word}{last}{suffix}"),
        None => String::from(suffix),
    }
}

/// The regular present participle (the en-verb exact rules): -ie to
/// -ying, -ue drops the e, consonant-e drops the e, C*VC doubles,
/// else +ing.
pub(crate) fn regular_participle(word: &str) -> String {
    let lower = word.to_lowercase();
    if lower.ends_with("ie") {
        return format!("{}ying", &word[..word.len() - 2]);
    }
    if lower.ends_with("ue") {
        return format!("{}ing", &word[..word.len() - 1]);
    }
    if lower.ends_with('e') {
        let before = lower.chars().rev().nth(1);
        if before.is_some_and(|c| c.is_ascii_alphabetic() && !"aeiouy".contains(c)) {
            return format!("{}ing", &word[..word.len() - 1]);
        }
        return format!("{word}ing");
    }
    if doubles_final(word) {
        return doubled(word, "ing");
    }
    format!("{word}ing")
}

/// The regular past (the en-verb exact rules): e takes +d,
/// consonant-y takes -ied, C*VC doubles, else +ed.
pub(crate) fn regular_past(word: &str) -> String {
    let lower = word.to_lowercase();
    if lower.ends_with('e') {
        return format!("{word}d");
    }
    if let Some(stem) = consonant_y_stem(word) {
        return format!("{stem}ied");
    }
    if doubles_final(word) {
        return doubled(word, "ed");
    }
    format!("{word}ed")
}

/// An -er/-est graded form (the en-adj rules): e drops, consonant-y
/// and consonant-ey become -i-, C*VC doubles, else the bare suffix.
fn graded_form(word: &str, suffix: &str) -> String {
    let lower = word.to_lowercase();
    if lower.ends_with('e') {
        return format!("{}{suffix}", &word[..word.len() - 1]);
    }
    if lower.ends_with("ey") {
        let head_end = word.len() - 2;
        let before = lower.chars().rev().nth(2);
        if before.is_some_and(|c| c.is_ascii_alphabetic() && !"aeiou".contains(c)) {
            return format!("{}i{suffix}", &word[..head_end]);
        }
    }
    if let Some(stem) = consonant_y_stem(word) {
        return format!("{stem}i{suffix}");
    }
    if doubles_final(word) {
        return doubled(word, suffix);
    }
    format!("{word}{suffix}")
}

/// The stem before a consonant-y ending, or None.
fn consonant_y_stem(word: &str) -> Option<&str> {
    let lower = word.to_lowercase();
    let mut chars = lower.chars().rev();
    if chars.next() != Some('y') {
        return None;
    }
    let before = chars.next()?;
    if "aeiou".contains(before) {
        return None;
    }
    Some(&word[..word.len() - 1])
}

/// One inflected form: its text plus rendered label text, from the
/// en-headword inline-modifier grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SpecForm {
    text: String,
    labels: Vec<String>,
}

impl SpecForm {
    fn plain(text: String) -> Self {
        Self { text, labels: Vec::new() }
    }

    /// The document rendering: an anchored form plus its label
    /// parenthetical. Embedded wikilinks flatten to their display
    /// text - the whole form is the anchor.
    fn rendered(&self) -> String {
        let mut text = self.text.clone();
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
    Some(parts.join(", ").replace("<<", "").replace(">>", ""))
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

/// An anchor per the display/resource rule: bare when the display IS
/// the resource name, display form otherwise.
fn anchor(display: &str, target: &str) -> String {
    if display == target {
        format!("[{target}]")
    } else {
        format!("[{display}]({target})")
    }
}

/// The renderer over one page's English subtree.
struct Renderer<'a> {
    page_title: &'a str,
    lines: Vec<String>,
    audit: Vec<AuditRow>,
}

impl<'a> Renderer<'a> {
    fn new(page_title: &'a str) -> Self {
        Self { page_title, lines: Vec::new(), audit: Vec::new() }
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
                        let display = link.display.as_deref().unwrap_or(rest);
                        out.push_str(&anchor(display, rest));
                        continue;
                    }
                    if target.contains(':') {
                        self.audit("link_dropped", target.to_string());
                        continue;
                    }
                    let base = link.display.as_deref().unwrap_or(target);
                    let display = format!("{base}{}", link.trail);
                    out.push_str(&anchor(&display, target));
                }
                Inline::ExternalLink { url, .. } => {
                    self.audit("external_link_dropped", url.clone());
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

    /// An anchored word argument: already-linked text renders as it
    /// is; a bare word bare-anchors.
    fn anchored_argument(&mut self, raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.contains("[[") {
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
                let qualifiers: Vec<String> = positional
                    .iter()
                    .skip(1)
                    .filter(|q| !q.is_empty() && **q != "_")
                    .map(|q| q.to_string())
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
            "l" | "m" | "m+" | "ll" => {
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
            _ => {
                self.audit("template_unhandled", template.name.clone());
                String::new()
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
        let transcriptions: Vec<String> = template
            .positional
            .iter()
            .skip(1)
            .filter(|part| !part.is_empty())
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
        let mut names: Vec<String> = Vec::new();
        for code in accents.split(',').filter(|code| !code.is_empty()) {
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
                        text: regular_plural(&title),
                        labels,
                    }),
                    "++" => {
                        let lower = title.to_lowercase();
                        let text = if lower.ends_with('s') || lower.ends_with('z') {
                            doubled(&title, "es")
                        } else {
                            regular_plural(&title)
                        };
                        plurals.push(SpecForm { text, labels });
                    }
                    "*" => plurals.push(SpecForm { text: title.clone(), labels }),
                    "s" => plurals.push(SpecForm { text: format!("{title}s"), labels }),
                    "es" => plurals.push(SpecForm { text: format!("{title}es"), labels }),
                    "ies" => {
                        let stem = title.strip_suffix('y').unwrap_or(&title);
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
                let rendered: Vec<String> =
                    singulars.iter().map(SpecForm::rendered).collect();
                pieces.push(format!("singular {}", rendered.join(" or ")));
            }
            let attributives = collect_family("attr");
            if !attributives.is_empty() {
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
            pieces.push(format!("plural [{}]", regular_plural(&title)));
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
            ("+" | "^", 0) => vec![with_labels(regular_plural(&word), None)],
            ("+" | "^", 1) => vec![with_labels(regular_participle(&word), None)],
            ("+" | "^", _) => vec![with_labels(regular_past(&word), None)],
            ("++", 0) => vec![with_labels(regular_plural(&word), None)],
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
            ("+l", 0) => vec![with_labels(regular_plural(&word), None)],
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
                            0 => regular_plural(&word),
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
    fn verb_pieces(&self, slots: Vec<Vec<SpecForm>>, participle_defective: bool) -> String {
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
        }
        if !superlatives.is_empty()
            && (!not_comparable || !comparatives.is_empty() || !sup_specs.is_empty())
        {
            pieces.push(format!("superlative {}", join(&superlatives)));
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
                    let anchored = self.anchored_argument(form);
                    last.1.push(anchored);
                }
            } else if !name.is_empty() {
                let forms = match form {
                    Some(form) if !form.is_empty() => {
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
            .map(|raw| self.argument_text(&raw.replace("\\,", ",")))
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
                let rendered: Vec<String> = spec_forms(abbr, &title)
                    .iter()
                    .map(SpecForm::rendered)
                    .collect();
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
                    renderer.blank();
                    renderer.lines.push(text);
                }
            }
            Block::ListItem { markers, content } => {
                let text = renderer.inline_text(content);
                if !text.is_empty() {
                    let depth = markers.len().saturating_sub(1);
                    renderer.lines.push(format!("{}- {text}", "  ".repeat(depth)));
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
    let mut lines: Vec<String> = Vec::new();
    for block in &section.blocks {
        let Block::ListItem { content, .. } = block else { continue };
        for inline in content {
            let Inline::Template(template) = inline else { continue };
            if template.name == "IPA" {
                let text = renderer.ipa_text(template);
                lines.push(format!("- {text}"));
            } else {
                renderer.audit("pronunciation_dropped", template.name.clone());
            }
        }
    }
    if lines.is_empty() {
        renderer.audit("section_empty", section.name.clone());
        return;
    }
    renderer.heading(level, &section.name);
    renderer.lines.extend(lines);
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
                        items.push(format!("- [{word}]"));
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
    renderer.lines.extend(items);
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
                    items.push(format!("- {text}"));
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
                                items.push(format!("- {anchored}"));
                            }
                        }
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
    renderer.lines.extend(items);
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
                } else {
                    renderer.audit("template_unhandled", template.name.clone());
                }
            }
        }
    }
    let headword = head_line
        .headword
        .unwrap_or_else(|| renderer.page_title.to_string());
    renderer
        .lines
        .push(format!("{} {headword}", "#".repeat(level + 1)));
    if let Some(parenthetical) = head_line.parenthetical {
        renderer.lines.push(parenthetical);
        renderer.lines.push(String::new());
    }
    let mut number = 0usize;
    let mut synonyms: Vec<String> = Vec::new();
    let mut antonyms: Vec<String> = Vec::new();
    for block in &section.blocks {
        let Block::ListItem { markers, content } = block else { continue };
        match markers.as_str() {
            "#" => {
                number += 1;
                let text = renderer.inline_text(content);
                renderer.lines.push(format!("{number}. {text}"));
            }
            "#:" => {
                for inline in content {
                    let Inline::Template(template) = inline else { continue };
                    match template.name.as_str() {
                        "syn" | "synonyms" => collect_sense_items(template, &mut synonyms),
                        "ant" | "antonyms" => collect_sense_items(template, &mut antonyms),
                        other => renderer.audit("sense_line_dropped", other.to_string()),
                    }
                }
            }
            "#*" => {
                for inline in content {
                    let Inline::Template(template) = inline else { continue };
                    if template.name.starts_with("quote-") {
                        if let Some(citation) = renderer.citation_text(template) {
                            renderer.lines.push(format!("   - {citation}"));
                        }
                    } else {
                        renderer.audit("quote_line_dropped", template.name.clone());
                    }
                }
            }
            _ => {
                let text = renderer.inline_text(content);
                renderer.audit("list_line_dropped", text);
            }
        }
    }
    if !synonyms.is_empty() {
        renderer.heading(level + 1, "Synonyms");
        renderer.lines.extend(synonyms);
    }
    if !antonyms.is_empty() {
        renderer.heading(level + 1, "Antonyms");
        renderer.lines.extend(antonyms);
    }
}

/// The sense-relational items: word positionals past the language,
/// their angle-bracket inline qualifiers stripped.
fn collect_sense_items(template: &Template, items: &mut Vec<String>) {
    for raw in template.positional.iter().skip(1) {
        let word = match raw.find('<') {
            Some(cut) => &raw[..cut],
            None => raw.as_str(),
        };
        let word = word.trim();
        if !word.is_empty() {
            items.push(format!("- [{word}]"));
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
