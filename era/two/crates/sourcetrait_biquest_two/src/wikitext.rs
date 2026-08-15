//! The wikitext core: the structural parser (blocks and inlines) and
//! the ruled typography normalization the renderer applies.
use crate::*;

/// Normalize typographic characters to their ASCII equivalents
/// (TheUser rulings). Quote marks collapse to the apostrophe and the
/// double quote by family; dashes to the bare dash, with the em-dash
/// and horizontal bar carrying the spacing the typographic form
/// omitted (' - ' between words; no space doubles where spacing
/// already exists); the ellipsis to three dots, the fraction slash
/// to the slash, the typographic spaces to the plain space, and the
/// invisibles drop. Semantic symbols keep their rows: middle dot,
/// multiplication sign, the modifier apostrophe (Letter-class), and
/// the primes. Applied at document rendering; raw page extraction
/// stays faithful.
pub(crate) fn normalize_typography(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    for (index, &c) in chars.iter().enumerate() {
        match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{2039}' | '\u{203A}'
            | '\u{00B4}' => out.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{00AB}' | '\u{00BB}' => {
                out.push('"')
            }
            '\u{2013}' | '\u{2012}' | '\u{2212}' | '\u{2022}' => out.push('-'),
            '\u{2014}' | '\u{2015}' => {
                let spaced_before =
                    index == 0 || chars[index - 1].is_whitespace();
                let spaced_after = index + 1 == chars.len()
                    || chars[index + 1].is_whitespace();
                if !spaced_before {
                    out.push(' ');
                }
                out.push('-');
                if !spaced_after {
                    out.push(' ');
                }
            }
            '\u{2026}' => out.push_str("..."),
            '\u{2044}' => out.push('/'),
            '\u{00A0}' | '\u{2002}' | '\u{2003}' | '\u{2009}' | '\u{202F}' => {
                out.push(' ')
            }
            '\u{00AD}' | '\u{200B}' | '\u{200E}' | '\u{200F}' => {}
            other => out.push(other),
        }
    }
    out
}

/// One template call: the name plus its raw argument sources. A
/// positional arg keeps its text verbatim; a named arg (a top-level
/// `=` in the segment) trims both sides, the MediaWiki convention.
/// Values stay raw wikitext - family handlers re-parse as needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Template {
    pub(crate) name: String,
    pub(crate) positional: Vec<String>,
    pub(crate) named: Vec<(String, String)>,
}

/// One wikilink: `[[target]]`, `[[target|display]]`, with the
/// linktrail (the ASCII-lowercase run after `]]`) that blends into
/// the display - the display/resource anchor rule's exact source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Link {
    pub(crate) target: String,
    /// Everything after the first top-level pipe, raw.
    pub(crate) display: Option<String>,
    pub(crate) trail: String,
}

/// An emphasis toggle; wikitext and markdown share the toggle model,
/// so no nesting tree is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Emphasis {
    /// `''`
    Italic,
    /// `'''`
    Bold,
    /// `'''''`
    BoldItalic,
}

/// One inline node. Comments strip during the parse; entity
/// references resolve into Text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Inline {
    Text(String),
    Template(Template),
    Link(Link),
    /// `[url]` or `[url label]`.
    ExternalLink { url: String, label: Option<String> },
    Emphasis(Emphasis),
    /// `<ref ...>content</ref>`, or a void `<ref ... />` (None).
    Ref { attrs: String, content: Option<String> },
    /// `<nowiki>` content, literal.
    Nowiki(String),
    /// Any other HTML-shaped tag token; pairing is renderer policy.
    Html { tag: String, attrs: String, closing: bool, void: bool },
}

/// One block of a page's wikitext.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Block {
    Heading { level: u8, content: Vec<Inline> },
    /// The marker run (`#`, `#*`, `#:`, `*`, ...) plus the line.
    ListItem { markers: String, content: Vec<Inline> },
    Paragraph { content: Vec<Inline> },
    /// A `{|`..`|}` table, raw - a strip class for the renderer.
    Table { source: String },
    HorizontalRule,
    Blank,
}

/// The named entities wikitext spells beyond the XML five; numeric
/// references always resolve. An unknown name stays literal text for
/// the audit to see.
const NAMED_ENTITIES: [(&str, char); 30] = [
    ("amp", '&'),
    ("lt", '<'),
    ("gt", '>'),
    ("quot", '"'),
    ("apos", '\''),
    ("nbsp", '\u{00A0}'),
    ("ensp", '\u{2002}'),
    ("emsp", '\u{2003}'),
    ("thinsp", '\u{2009}'),
    ("ndash", '\u{2013}'),
    ("mdash", '\u{2014}'),
    ("minus", '\u{2212}'),
    ("hellip", '\u{2026}'),
    ("middot", '\u{00B7}'),
    ("times", '\u{00D7}'),
    ("deg", '\u{00B0}'),
    ("prime", '\u{2032}'),
    ("Prime", '\u{2033}'),
    ("lsquo", '\u{2018}'),
    ("rsquo", '\u{2019}'),
    ("ldquo", '\u{201C}'),
    ("rdquo", '\u{201D}'),
    ("laquo", '\u{00AB}'),
    ("raquo", '\u{00BB}'),
    ("sect", '\u{00A7}'),
    ("para", '\u{00B6}'),
    ("copy", '\u{00A9}'),
    ("reg", '\u{00AE}'),
    ("trade", '\u{2122}'),
    ("bull", '\u{2022}'),
];

/// The byte cursor. Structural characters are all ASCII, so byte
/// scanning is UTF-8 safe: every acted-on position is an ASCII byte,
/// so every slice boundary is a char boundary.
struct Cursor<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, position: 0 }
    }

    fn done(&self) -> bool {
        self.position >= self.text.len()
    }

    fn byte(&self) -> u8 {
        self.text.as_bytes()[self.position]
    }

    /// Byte-based, so a cursor mid-way through a multibyte character
    /// (the segment scanners advance byte-wise) never panics; an
    /// ASCII pattern cannot start inside one anyway.
    fn starts_with(&self, pattern: &str) -> bool {
        self.text.as_bytes()[self.position..].starts_with(pattern.as_bytes())
    }

    /// The byte run length of `byte` at the cursor.
    fn run_of(&self, byte: u8) -> usize {
        self.text.as_bytes()[self.position..]
            .iter()
            .take_while(|&&b| b == byte)
            .count()
    }

    /// Find a pattern from the cursor; absolute byte position.
    /// Byte-based for the same mid-character safety as starts_with.
    fn find(&self, pattern: &str) -> Option<usize> {
        let needle = pattern.as_bytes();
        self.text.as_bytes()[self.position..]
            .windows(needle.len())
            .position(|window| window == needle)
            .map(|offset| self.position + offset)
    }

    /// The current line's remainder (newline excluded), cursor kept.
    fn line_rest(&self) -> &'a str {
        let rest = &self.text[self.position..];
        match rest.find('\n') {
            Some(end) => &rest[..end],
            None => rest,
        }
    }

    /// Consume through the current line's newline (or the end).
    fn consume_line(&mut self) {
        match self.find("\n") {
            Some(end) => self.position = end + 1,
            None => self.position = self.text.len(),
        }
    }
}

/// Parse a whole page's wikitext into blocks.
pub(crate) fn parse_blocks(text: &str) -> BiquestResult<Vec<Block>> {
    let mut cursor = Cursor::new(text);
    let mut blocks = Vec::new();
    while !cursor.done() {
        let line = cursor.line_rest();
        let trimmed = line.trim_end();
        if trimmed.trim().is_empty() {
            blocks.push(Block::Blank);
            cursor.consume_line();
            continue;
        }
        if let Some(heading) = heading_of(trimmed)? {
            blocks.push(heading);
            cursor.consume_line();
            continue;
        }
        if trimmed.starts_with("{|") {
            blocks.push(table_block(&mut cursor));
            continue;
        }
        if trimmed.starts_with("----") {
            blocks.push(Block::HorizontalRule);
            cursor.consume_line();
            continue;
        }
        let markers = trimmed
            .bytes()
            .take_while(|b| matches!(b, b'#' | b'*' | b':' | b';'))
            .count();
        if markers > 0 {
            let markers_text = line[..markers].to_string();
            cursor.position += markers;
            let content = parse_inlines(&mut cursor, true)?;
            blocks.push(Block::ListItem { markers: markers_text, content });
            continue;
        }
        let content = parse_inlines(&mut cursor, true)?;
        blocks.push(Block::Paragraph { content });
    }
    Ok(blocks)
}

/// A heading when the whole line is `=+ inner =+`; level is the
/// shorter run, capped at six. A line that only looks like one (no
/// inner, a lone run) stays a paragraph.
fn heading_of(line: &str) -> BiquestResult<Option<Block>> {
    if !line.starts_with('=') {
        return Ok(None);
    }
    let leading = line.bytes().take_while(|&b| b == b'=').count();
    let trailing = line.bytes().rev().take_while(|&b| b == b'=').count();
    if leading + trailing >= line.len() {
        return Ok(None);
    }
    let inner = line[leading..line.len() - trailing].trim();
    if inner.is_empty() {
        return Ok(None);
    }
    let level = leading.min(trailing).min(6) as u8;
    Ok(Some(Block::Heading { level, content: parse_inline_text(inner)? }))
}

/// Capture a `{|`..`|}` table raw, nesting-aware, delimiters
/// included. A table that never closes swallows to the end - the
/// MediaWiki behavior - rather than failing the page.
fn table_block(cursor: &mut Cursor<'_>) -> Block {
    let start = cursor.position;
    let mut depth = 0usize;
    loop {
        if cursor.done() {
            return Block::Table { source: cursor.text[start..].to_string() };
        }
        let line = cursor.line_rest().trim_start();
        if line.starts_with("{|") {
            depth += 1;
        } else if line.starts_with("|}") {
            depth -= 1;
            if depth == 0 {
                let line_end = cursor.position + cursor.line_rest().len();
                let source = cursor.text[start..line_end].to_string();
                cursor.consume_line();
                return Block::Table { source };
            }
        }
        cursor.consume_line();
    }
}

/// Parse a standalone wikitext fragment (a heading interior, a
/// template argument) into inlines.
pub(crate) fn parse_inline_text(text: &str) -> BiquestResult<Vec<Inline>> {
    let mut cursor = Cursor::new(text);
    parse_inlines(&mut cursor, false)
}

/// Push pending text as a node.
fn flush(pending: &mut String, inlines: &mut Vec<Inline>) {
    if !pending.is_empty() {
        inlines.push(Inline::Text(std::mem::take(pending)));
    }
}

/// The inline parser: consumes to the stop (a depth-zero newline, or
/// the end). Comments strip; entities resolve into the text.
fn parse_inlines(
    cursor: &mut Cursor<'_>,
    stop_at_newline: bool,
) -> BiquestResult<Vec<Inline>> {
    let mut inlines = Vec::new();
    let mut pending = String::new();
    while !cursor.done() {
        let byte = cursor.byte();
        if byte == b'\n' {
            cursor.position += 1;
            if stop_at_newline {
                break;
            }
            pending.push('\n');
            continue;
        }
        if cursor.starts_with("<!--") {
            skip_comment(cursor);
            continue;
        }
        if cursor.starts_with("{{") {
            // An unterminated construct is literal text on-site, so a
            // failed parse restores the cursor and keeps the opener as
            // text rather than failing the page.
            let saved = cursor.position;
            match parse_template(cursor) {
                Ok(template) => {
                    flush(&mut pending, &mut inlines);
                    inlines.push(Inline::Template(template));
                }
                Err(_) => {
                    cursor.position = saved + 2;
                    pending.push_str("{{");
                }
            }
            continue;
        }
        if cursor.starts_with("[[") {
            let saved = cursor.position;
            match parse_link(cursor) {
                Ok(link) => {
                    flush(&mut pending, &mut inlines);
                    inlines.push(Inline::Link(link));
                }
                Err(_) => {
                    cursor.position = saved + 2;
                    pending.push_str("[[");
                }
            }
            continue;
        }
        if byte == b'[' {
            if let Some(external) = parse_external_link(cursor) {
                flush(&mut pending, &mut inlines);
                inlines.push(external);
                continue;
            }
            pending.push('[');
            cursor.position += 1;
            continue;
        }
        if byte == b'\'' {
            let run = cursor.run_of(b'\'');
            if run >= 2 {
                flush(&mut pending, &mut inlines);
                let (extra, emphasis) = match run {
                    2 => (0, Emphasis::Italic),
                    3 | 4 => (run - 3, Emphasis::Bold),
                    _ => (run - 5, Emphasis::BoldItalic),
                };
                for _ in 0..extra {
                    pending.push('\'');
                }
                flush(&mut pending, &mut inlines);
                inlines.push(Inline::Emphasis(emphasis));
                cursor.position += run;
                continue;
            }
            pending.push('\'');
            cursor.position += 1;
            continue;
        }
        if cursor.starts_with("<nowiki>") {
            let opener_end = cursor.position + "<nowiki>".len();
            match cursor.text[opener_end..]
                .find("</nowiki>")
                .map(|offset| opener_end + offset)
            {
                Some(end) => {
                    flush(&mut pending, &mut inlines);
                    inlines.push(Inline::Nowiki(
                        cursor.text[opener_end..end].to_string(),
                    ));
                    cursor.position = end + "</nowiki>".len();
                }
                None => {
                    // Unterminated: the opener is literal text.
                    pending.push('<');
                    cursor.position += 1;
                }
            }
            continue;
        }
        if cursor.starts_with("<ref") {
            let saved = cursor.position;
            match parse_ref(cursor) {
                Ok(reference) => {
                    flush(&mut pending, &mut inlines);
                    inlines.push(reference);
                }
                Err(_) => {
                    cursor.position = saved + 1;
                    pending.push('<');
                }
            }
            continue;
        }
        if byte == b'<' {
            if let Some(html) = parse_html_tag(cursor) {
                flush(&mut pending, &mut inlines);
                inlines.push(html);
                continue;
            }
            pending.push('<');
            cursor.position += 1;
            continue;
        }
        if byte == b'&' {
            if let Some(resolved) = parse_entity(cursor) {
                pending.push(resolved);
                continue;
            }
            pending.push('&');
            cursor.position += 1;
            continue;
        }
        let rest = &cursor.text[cursor.position..];
        let c = rest.chars().next().expect("cursor is not done");
        pending.push(c);
        cursor.position += c.len_utf8();
    }
    flush(&mut pending, &mut inlines);
    Ok(inlines)
}

/// Skip `<!-- -->`; an unterminated comment swallows to the end, the
/// MediaWiki behavior.
fn skip_comment(cursor: &mut Cursor<'_>) {
    cursor.position += "<!--".len();
    match cursor.find("-->") {
        Some(end) => cursor.position = end + "-->".len(),
        None => cursor.position = cursor.text.len(),
    }
}

/// A segment scan for templates and links: raw spans between
/// top-level pipes, nesting-aware ({{ }}, [[ ]]), comment- and
/// nowiki-blind, newline-crossing. Returns the segments and leaves
/// the cursor after the closer.
fn scan_segments(
    cursor: &mut Cursor<'_>,
    closer: &str,
) -> BiquestResult<Vec<String>> {
    let opened_at = cursor.position;
    let mut segments = Vec::new();
    let mut segment_start = cursor.position;
    let mut brace_depth = 0usize;
    let mut bracket_depth = 0usize;
    loop {
        snafu::ensure_whatever!(
            !cursor.done(),
            "the construct opened at byte {opened_at} never closes"
        );
        if cursor.starts_with("<!--") {
            skip_comment(cursor);
            continue;
        }
        if cursor.starts_with("<nowiki>") {
            match cursor.find("</nowiki>") {
                Some(end) => cursor.position = end + "</nowiki>".len(),
                None => snafu::whatever!("a nowiki span never closes"),
            }
            continue;
        }
        if brace_depth == 0 && bracket_depth == 0 && cursor.starts_with(closer) {
            segments.push(cursor.text[segment_start..cursor.position].to_string());
            cursor.position += closer.len();
            return Ok(segments);
        }
        if cursor.starts_with("{{") {
            brace_depth += 1;
            cursor.position += 2;
            continue;
        }
        if brace_depth > 0 && cursor.starts_with("}}") {
            brace_depth -= 1;
            cursor.position += 2;
            continue;
        }
        if cursor.starts_with("[[") {
            bracket_depth += 1;
            cursor.position += 2;
            continue;
        }
        if bracket_depth > 0 && cursor.starts_with("]]") {
            bracket_depth -= 1;
            cursor.position += 2;
            continue;
        }
        if brace_depth == 0 && bracket_depth == 0 && cursor.byte() == b'|' {
            segments.push(cursor.text[segment_start..cursor.position].to_string());
            cursor.position += 1;
            segment_start = cursor.position;
            continue;
        }
        cursor.position += 1;
    }
}

/// Parse `{{name|...}}` from its opener. The name trims; a segment
/// with a top-level `=` is named (both sides trimmed), the rest are
/// positional, verbatim.
fn parse_template(cursor: &mut Cursor<'_>) -> BiquestResult<Template> {
    cursor.position += 2;
    let segments = scan_segments(cursor, "}}")?;
    let mut iterator = segments.into_iter();
    let name = iterator.next().unwrap_or_default().trim().to_string();
    let mut positional = Vec::new();
    let mut named = Vec::new();
    for segment in iterator {
        match top_level_equals(&segment) {
            Some(split) => {
                let key = segment[..split].trim().to_string();
                let value = segment[split + 1..].trim().to_string();
                named.push((key, value));
            }
            None => positional.push(segment),
        }
    }
    Ok(Template { name, positional, named })
}

/// The first `=` outside any nesting, or None for a positional.
/// Byte-based throughout for mid-character safety.
fn top_level_equals(segment: &str) -> Option<usize> {
    let bytes = segment.as_bytes();
    let mut brace_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index..].starts_with(b"{{") {
            brace_depth += 1;
            index += 2;
        } else if brace_depth > 0 && bytes[index..].starts_with(b"}}") {
            brace_depth -= 1;
            index += 2;
        } else if bytes[index..].starts_with(b"[[") {
            bracket_depth += 1;
            index += 2;
        } else if bracket_depth > 0 && bytes[index..].starts_with(b"]]") {
            bracket_depth -= 1;
            index += 2;
        } else {
            if bytes[index] == b'=' && brace_depth == 0 && bracket_depth == 0 {
                return Some(index);
            }
            index += 1;
        }
    }
    None
}

/// Parse `[[target|display]]trail` from its opener. The display is
/// everything past the first top-level pipe, raw; the trail is the
/// ASCII-lowercase run after the closer (the enwiki linktrail rule).
fn parse_link(cursor: &mut Cursor<'_>) -> BiquestResult<Link> {
    cursor.position += 2;
    let segments = scan_segments(cursor, "]]")?;
    let mut iterator = segments.into_iter();
    let target = iterator.next().unwrap_or_default().trim().to_string();
    let display_parts: Vec<String> = iterator.collect();
    let display = if display_parts.is_empty() {
        None
    } else {
        Some(display_parts.join("|"))
    };
    let trail_length = cursor.run_of_lowercase();
    let trail = cursor.text[cursor.position..cursor.position + trail_length].to_string();
    cursor.position += trail_length;
    Ok(Link { target, display, trail })
}

impl Cursor<'_> {
    /// The ASCII-lowercase run at the cursor (the linktrail).
    fn run_of_lowercase(&self) -> usize {
        self.text.as_bytes()[self.position..]
            .iter()
            .take_while(|b| b.is_ascii_lowercase())
            .count()
    }
}

/// `[scheme://...]` as an external link; anything else is not one.
/// Confined to its line: an unclosed bracket falls back to text.
fn parse_external_link(cursor: &mut Cursor<'_>) -> Option<Inline> {
    let rest = &cursor.text[cursor.position + 1..];
    let is_url = ["http://", "https://", "ftp://", "//"]
        .iter()
        .any(|scheme| rest.starts_with(scheme));
    if !is_url {
        return None;
    }
    let line_end = rest.find('\n').unwrap_or(rest.len());
    let close = rest[..line_end].find(']')?;
    let interior = &rest[..close];
    let (url, label) = match interior.find(' ') {
        Some(space) => (
            interior[..space].to_string(),
            Some(interior[space + 1..].to_string()),
        ),
        None => (interior.to_string(), None),
    };
    cursor.position += 1 + close + 1;
    Some(Inline::ExternalLink { url, label })
}

/// Parse `<ref ...>content</ref>` or the void `<ref ... />`.
fn parse_ref(cursor: &mut Cursor<'_>) -> BiquestResult<Inline> {
    let opened_at = cursor.position;
    cursor.position += "<ref".len();
    let Some(open_end) = cursor.find(">") else {
        snafu::whatever!("a ref tag opened at byte {opened_at} never closes its opener");
    };
    let attrs_raw = &cursor.text[cursor.position..open_end];
    let void = attrs_raw.trim_end().ends_with('/');
    let attrs = attrs_raw.trim().trim_end_matches('/').trim().to_string();
    cursor.position = open_end + 1;
    if void {
        return Ok(Inline::Ref { attrs, content: None });
    }
    let Some(close) = cursor.find("</ref>") else {
        snafu::whatever!("the ref opened at byte {opened_at} never closes");
    };
    let content = cursor.text[cursor.position..close].to_string();
    cursor.position = close + "</ref>".len();
    Ok(Inline::Ref { attrs, content: Some(content) })
}

/// An HTML-shaped tag token (`<name ...>`, `</name>`, `<name/>`),
/// confined to its line; pairing is renderer policy. Not-a-tag
/// returns None and the `<` stays text.
fn parse_html_tag(cursor: &mut Cursor<'_>) -> Option<Inline> {
    let rest = &cursor.text[cursor.position + 1..];
    let closing = rest.starts_with('/');
    let name_start = if closing { 1 } else { 0 };
    let name_length = rest[name_start..]
        .bytes()
        .take_while(|b| b.is_ascii_alphanumeric())
        .count();
    if name_length == 0 {
        return None;
    }
    let line_end = rest.find('\n').unwrap_or(rest.len());
    let close = rest[..line_end].find('>')?;
    if close < name_start + name_length {
        return None;
    }
    let tag = rest[name_start..name_start + name_length].to_string();
    let interior = &rest[name_start + name_length..close];
    let void = interior.trim_end().ends_with('/');
    let attrs = interior.trim().trim_end_matches('/').trim().to_string();
    cursor.position += 1 + close + 1;
    Some(Inline::Html { tag, attrs, closing, void })
}

/// Resolve `&name;`, `&#nnn;`, or `&#xhh;` at the cursor; an unknown
/// name stays literal for the audit to see. The lookahead scans
/// bytes - a length-capped str slice can land inside a multibyte
/// character.
fn parse_entity(cursor: &mut Cursor<'_>) -> Option<char> {
    let rest = &cursor.text[cursor.position + 1..];
    let bytes = rest.as_bytes();
    let semicolon = bytes[..bytes.len().min(12)]
        .iter()
        .position(|&b| b == b';')?;
    let body = &rest[..semicolon];
    let resolved = if let Some(digits) = body.strip_prefix("#x").or(body.strip_prefix("#X")) {
        char::from_u32(u32::from_str_radix(digits, 16).ok()?)?
    } else if let Some(digits) = body.strip_prefix('#') {
        char::from_u32(digits.parse::<u32>().ok()?)?
    } else {
        NAMED_ENTITIES
            .iter()
            .find(|(name, _)| *name == body)
            .map(|&(_, c)| c)?
    };
    cursor.position += 1 + semicolon + 1;
    Some(resolved)
}
