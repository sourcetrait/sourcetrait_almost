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

/// The IPA accent-code map, audit-grown from the observed samples.
const ACCENT_NAMES: [(&str, &str); 7] = [
    ("RP", "Received Pronunciation"),
    ("GA", "General American"),
    ("CA", "Canada"),
    ("AU", "Australia"),
    ("NZ", "New Zealand"),
    ("cot-caught", "cot-caught merger"),
    ("non-cot-caught", "without the cot-caught merger"),
];

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

/// The regular present participle: a silent trailing e drops.
pub(crate) fn regular_participle(word: &str) -> String {
    let lower = word.to_lowercase();
    if lower.ends_with('e') && !lower.ends_with("ee") && !lower.ends_with("ye") {
        return format!("{}ing", &word[..word.len() - 1]);
    }
    format!("{word}ing")
}

/// The regular past: trailing e takes +d, consonant-y takes -ied.
pub(crate) fn regular_past(word: &str) -> String {
    let lower = word.to_lowercase();
    if lower.ends_with('e') {
        return format!("{word}d");
    }
    if let Some(stem) = consonant_y_stem(word) {
        return format!("{stem}ied");
    }
    format!("{word}ed")
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
                let parts = self.plus_joined(&positional[1..]);
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
            "affix" | "af" | "compound" | "com" => self.plus_joined(&positional[1..]),
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
        let mut names: Vec<String> = Vec::new();
        for code in accents.split(',').filter(|code| !code.is_empty()) {
            let mapped = ACCENT_NAMES
                .iter()
                .find(|(known, _)| *known == code)
                .map(|(_, name)| name.to_string());
            match mapped {
                Some(name) => names.push(name),
                None => match code.strip_prefix("dialects of ") {
                    Some(place) => names.push(format!("also of {place}")),
                    None => {
                        self.audit("accent_code_unknown", code.to_string());
                        names.push(code.to_string());
                    }
                },
            }
        }
        let joined = transcriptions.join(", ");
        if names.is_empty() {
            format!("IPA: {joined}")
        } else {
            format!("IPA ({}): {joined}", names.join(", "))
        }
    }

    /// The headword parenthetical a bare head template derives; args
    /// beyond the language audit until their grammar is needed.
    fn head_parenthetical(&mut self, template: &Template) -> Option<String> {
        if template.positional.len() > 1 || !template.named.is_empty() {
            self.audit("head_arguments_unhandled", template.name.clone());
            return None;
        }
        let title = self.page_title;
        match template.name.as_str() {
            "en-noun" => {
                Some(format!("(plural [{}])", regular_plural(title)))
            }
            "en-verb" => Some(format!(
                "(third-person singular simple present [{}], present participle [{}], simple past and past participle [{}])",
                regular_plural(title),
                regular_participle(title),
                regular_past(title),
            )),
            "en-adj" => Some(format!(
                "(comparative [more {title}], superlative [most {title}])"
            )),
            "en-prop" | "en-proper noun" | "en-proper-noun" => None,
            other => {
                self.audit("head_template_unhandled", other.to_string());
                None
            }
        }
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
    renderer.lines.push(format!("{} {}", "#".repeat(level + 1), renderer.page_title));
    let mut parenthetical: Option<String> = None;
    let mut head_seen = false;
    for block in &section.blocks {
        if let Block::Paragraph { content } = block {
            for inline in content {
                let Inline::Template(template) = inline else { continue };
                if !head_seen {
                    head_seen = true;
                    parenthetical = renderer.head_parenthetical(template);
                } else {
                    renderer.audit("template_unhandled", template.name.clone());
                }
            }
        }
    }
    if let Some(parenthetical) = parenthetical {
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
