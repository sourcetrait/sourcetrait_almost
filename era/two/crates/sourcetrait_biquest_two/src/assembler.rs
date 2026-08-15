//! The QuillAssembler: assembly text to wire tokens and back.
use crate::*;

use crate::bucket::BucketTable;
use crate::dictionary::read_words_ordered;
use crate::lexer::HARDCODED_BLOCK_FLOOR;
use crate::lexer::KEYWORD_BEGIN_REPEAT;
use crate::lexer::KEYWORD_CAPITALIZED;
use crate::lexer::KEYWORD_CASE;
use crate::lexer::KEYWORD_CASED;
use crate::lexer::KEYWORD_END_REPEAT;
use crate::lexer::KEYWORD_PAGE_SIZE;
use crate::lexer::KEYWORD_REPEAT;
use crate::lexer::KEYWORD_REPETITION;
use crate::lexer::KEYWORD_UNICODE;
use crate::lexer::KEYWORD_UNICODED;
use crate::lexer::KEYWORD_UPPERCASED;
use crate::lexer::Segmenter;
use crate::ucd::CharClass;
use crate::ucd::CharacterTable;

/// One indentation level of the assembly surface.
const INDENT: &str = "  ";

/// The four structural heads, resolved from the table BY NAME.
const HEAD_OPEN: &str = "OPEN";
const HEAD_CLOSE: &str = "CLOSE";
const HEAD_BEGIN: &str = "BEGIN";
const HEAD_END: &str = "END";

/// The REIGN meta line: an authorship marker the assembler validates
/// and drops - it produces no wire (TheUser's ruling).
const REIGN_HEAD: &str = "REIGN";
const REIGN_VALUES: [&str; 3] = ["human", "ai", "auto"];

/// The escape mechanics resolve by name when bound; a table without
/// them simply cannot express escape spans.
const NAME_ESCAPE: &str = "ESCAPE";
const NAME_ESCAPED: &str = "ESCAPED";
const NAME_TRAIN: &str = "TRAIN";
const NAME_INPUT: &str = "INPUT";

/// The hardcoded block's assembly-surface names: fixed by the codec,
/// resolved and rendered beside the user table (which may not bind
/// them). The span pair lives here because forced tokenization is a
/// tokenizer property, not portable language syntax (TheUser).
const HARDCODED_NAMES: [(u32, &str); 10] = [
    (KEYWORD_UNICODE, "UNICODE"),
    (KEYWORD_UNICODED, "UNICODED"),
    (KEYWORD_CASED, "CASED"),
    (KEYWORD_CASE, "CASE"),
    (KEYWORD_CAPITALIZED, "CAPITALIZED"),
    (KEYWORD_UPPERCASED, "UPPERCASED"),
    (KEYWORD_BEGIN_REPEAT, "BEGIN_REPEAT"),
    (KEYWORD_END_REPEAT, "END_REPEAT"),
    (KEYWORD_REPEAT, "REPEAT"),
    (KEYWORD_REPETITION, "REPETITION"),
];

/// The vendored default binding table (TheUser's Syntax.nuon draft),
/// embedded so the bare verbs need no --syntax; re-vendor from the
/// reference copy when the draft moves.
const DEFAULT_SYNTAX: &str = include_str!("../data/syntax/Syntax.nuon");

/// The user-binding half of the keyword page: Syntax.nuon remapped
/// onto the page in order, NULL at 0x00. Bindings stop below the
/// hardcoded operator block.
pub(crate) struct SyntaxTable {
    names: Vec<String>,
    ids: HashMap<String, u32>,
    open: u32,
    close: u32,
    begin: u32,
    end: u32,
    escape: Option<u32>,
    escaped: Option<u32>,
    train: Option<u32>,
    input: Option<u32>,
}

impl SyntaxTable {
    /// Build from the table's name column, in page order.
    pub(crate) fn from_names(names: Vec<String>) -> BiquestResult<Self> {
        snafu::ensure_whatever!(
            names.len() <= HARDCODED_BLOCK_FLOOR as usize,
            "the syntax table carries {} bindings; the page holds {} below the \
             hardcoded operator block",
            names.len(),
            HARDCODED_BLOCK_FLOOR
        );
        let mut ids = HashMap::with_capacity(names.len());
        for (index, name) in names.iter().enumerate() {
            snafu::ensure_whatever!(
                !HARDCODED_NAMES.iter().any(|(_, hardcoded)| hardcoded == name),
                "the syntax table binds {name}, a hardcoded tokenizer operator's \
                 fixed name"
            );
            snafu::ensure_whatever!(
                ids.insert(name.clone(), index as u32).is_none(),
                "the syntax table binds {name} twice"
            );
        }
        let head = |name: &str| -> BiquestResult<u32> {
            match ids.get(name) {
                Some(&id) => Ok(id),
                None => snafu::whatever!("the syntax table lacks the {name} head"),
            }
        };
        Ok(Self {
            open: head(HEAD_OPEN)?,
            close: head(HEAD_CLOSE)?,
            begin: head(HEAD_BEGIN)?,
            end: head(HEAD_END)?,
            escape: ids.get(NAME_ESCAPE).copied(),
            escaped: ids.get(NAME_ESCAPED).copied(),
            train: ids.get(NAME_TRAIN).copied(),
            input: ids.get(NAME_INPUT).copied(),
            names,
            ids,
        })
    }

    /// Parse a binding table's NUON text: a table with a name column.
    pub(crate) fn from_text(text: &str) -> BiquestResult<Self> {
        let value = harness::nu::from_nuon_text(text)?;
        let rows = match value.as_list() {
            Ok(rows) => rows,
            Err(e) => snafu::whatever!("the syntax table is not a table: {e}"),
        };
        let mut names = Vec::with_capacity(rows.len());
        for row in rows {
            let record = match row.as_record() {
                Ok(record) => record,
                Err(e) => snafu::whatever!("syntax row: {e}"),
            };
            names.push(field_str(record, "name")?);
        }
        Self::from_names(names)
    }

    /// The embedded default table.
    pub(crate) fn embedded() -> BiquestResult<Self> {
        Self::from_text(DEFAULT_SYNTAX)
    }

    /// Load the `> HUMAN` draft table from a file.
    pub(crate) fn load(path: &Path) -> BiquestResult<Self> {
        let text = fs::read_to_string(path)?;
        Self::from_text(&text)
    }

    fn id_of(&self, name: &str) -> Option<u32> {
        self.ids.get(name).copied()
    }

    fn name_of(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(String::as_str)
    }
}

/// A raw-string opener for an interior, escalated until collision-free:
/// `#{`, `##{`, and so on, closing with the mirrored `}#` form.
fn raw_delimiters(interior: &str) -> (String, String) {
    let mut hashes = 1usize;
    loop {
        let closer = format!("}}{}", "#".repeat(hashes));
        let collides = interior
            .lines()
            .any(|line| line.trim() == closer || line.trim() == format!("{}{{", "#".repeat(hashes)));
        if !collides {
            return (format!("{}{{", "#".repeat(hashes)), closer);
        }
        hashes += 1;
    }
}

/// Whether a rendered interior needs raw-string protection: any line
/// that would read as a structural line, or as a raw delimiter.
fn interior_needs_raw(interior: &str) -> bool {
    interior.lines().any(|line| {
        let trimmed = line.trim_start();
        [HEAD_OPEN, HEAD_CLOSE, HEAD_BEGIN, HEAD_END]
            .iter()
            .any(|head| trimmed.strip_prefix(head).is_some_and(|rest| rest.starts_with(' ')))
            || (trimmed.starts_with('#') && trimmed.trim_end().ends_with('{'))
            || (trimmed.starts_with('}') && trimmed.trim_end().ends_with('#'))
    })
}

/// Strip exactly `depth` indentation levels off a non-empty content
/// line; excess indentation is the content's own, and a shortfall
/// faults - assembly indentation is enforced for parse precision
/// (TheUser's ruling: two spaces per level, as Syntax.md shows).
fn strip_indent(line: &str, depth: usize, number: usize) -> BiquestResult<&str> {
    if line.is_empty() {
        return Ok(line);
    }
    let mut rest = line;
    for _ in 0..depth {
        match rest.strip_prefix(INDENT) {
            Some(stripped) => rest = stripped,
            None => snafu::whatever!(
                "line {number}: content wants {depth} indentation levels \
                 ({} spaces)",
                depth * INDENT.len()
            ),
        }
    }
    Ok(rest)
}

/// Require a structural line to sit at exactly `depth` levels: the
/// expected indent, then a non-space head.
fn require_depth(line: &str, depth: usize, number: usize) -> BiquestResult<()> {
    let expected = INDENT.len() * depth;
    let leading = line.len() - line.trim_start_matches(' ').len();
    snafu::ensure_whatever!(
        leading == expected,
        "line {number}: expected {expected} spaces of indentation ({depth} \
         levels), got {leading}"
    );
    Ok(())
}

/// The Quill assembler over one syntax table and content tokenizer.
pub(crate) struct Assembler<'a> {
    syntax: &'a SyntaxTable,
    segmenter: &'a Segmenter<'a>,
    table: &'a CharacterTable,
    buckets: &'a BucketTable,
    admitted: &'a [String],
}

/// What one line of assembly parses to, structurally.
enum AssemblyLine {
    Structural { head: u32, noun: u32 },
    Bare(u32),
    /// A validated REIGN authorship line; it produces no wire.
    Reign,
    Blank,
}

impl<'a> Assembler<'a> {
    pub(crate) fn new(
        syntax: &'a SyntaxTable,
        segmenter: &'a Segmenter<'a>,
        table: &'a CharacterTable,
        buckets: &'a BucketTable,
        admitted: &'a [String],
    ) -> Self {
        Self { syntax, segmenter, table, buckets, admitted }
    }

    fn parse_line(&self, line: &str, number: usize) -> BiquestResult<AssemblyLine> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(AssemblyLine::Blank);
        }
        let mut words = trimmed.split_whitespace();
        let head = words.next().expect("non-empty line");
        let noun = words.next();
        snafu::ensure_whatever!(
            words.next().is_none(),
            "line {number}: an assembly line carries at most a head and a noun"
        );
        if head == REIGN_HEAD {
            match noun {
                Some(value) if REIGN_VALUES.contains(&value) => {
                    return Ok(AssemblyLine::Reign);
                }
                Some(value) => snafu::whatever!(
                    "line {number}: REIGN wants human, ai, or auto, got {value}"
                ),
                None => snafu::whatever!(
                    "line {number}: REIGN wants human, ai, or auto"
                ),
            }
        }
        let Some(head_id) = self.syntax.id_of(head) else {
            snafu::whatever!("line {number}: {head} is not a bound keyword");
        };
        match noun {
            Some(noun) => {
                let structural = [
                    self.syntax.open,
                    self.syntax.close,
                    self.syntax.begin,
                    self.syntax.end,
                ]
                .contains(&head_id);
                snafu::ensure_whatever!(
                    structural,
                    "line {number}: {head} takes no noun (only OPEN/CLOSE/BEGIN/END do)"
                );
                let Some(noun_id) = self.syntax.id_of(noun) else {
                    snafu::whatever!("line {number}: {noun} is not a bound keyword");
                };
                Ok(AssemblyLine::Structural { head: head_id, noun: noun_id })
            }
            None => Ok(AssemblyLine::Bare(head_id)),
        }
    }

    /// Assembly text to the wire: indentation is enforced at exactly
    /// two spaces per level and drops (so no structural whitespace
    /// reaches the wire), structure checks as it encodes, and
    /// serialization interiors pass through content tokenization
    /// (raw-string wrappers strip). Blank lines are legal only
    /// outside any open block.
    pub(crate) fn encode(&self, text: &str) -> BiquestResult<Vec<u32>> {
        let mut wire: Vec<u32> = Vec::new();
        let mut stack: Vec<u32> = Vec::new();
        let mut lines = text.lines().enumerate().peekable();
        while let Some((index, line)) = lines.next() {
            let number = index + 1;
            match self.parse_line(line, number)? {
                AssemblyLine::Blank => {
                    // Newlines between things are enforced: a blank
                    // line is legal only outside any open block (the
                    // REIGN header region).
                    snafu::ensure_whatever!(
                        stack.is_empty(),
                        "line {number}: a blank line inside an open block"
                    );
                }
                AssemblyLine::Reign => {
                    require_depth(line, stack.len(), number)?;
                }
                AssemblyLine::Bare(id) => {
                    require_depth(line, stack.len(), number)?;
                    wire.push(id);
                }
                AssemblyLine::Structural { head, noun } => {
                    if head == self.syntax.open {
                        require_depth(line, stack.len(), number)?;
                        wire.push(head);
                        wire.push(noun);
                        stack.push(noun);
                    } else if head == self.syntax.close {
                        let Some(expected) = stack.pop() else {
                            snafu::whatever!("line {number}: CLOSE with nothing open");
                        };
                        require_depth(line, stack.len(), number)?;
                        snafu::ensure_whatever!(
                            expected == noun,
                            "line {number}: CLOSE {} against open {}",
                            self.syntax.name_of(noun).unwrap_or("?"),
                            self.syntax.name_of(expected).unwrap_or("?")
                        );
                        wire.push(head);
                        wire.push(noun);
                    } else if head == self.syntax.begin {
                        require_depth(line, stack.len(), number)?;
                        wire.push(head);
                        wire.push(noun);
                        let depth = stack.len() + 1;
                        let interior =
                            self.collect_interior(&mut lines, noun, depth, number)?;
                        let spans_legal = self.spans_legal(&stack);
                        self.encode_interior(&interior, spans_legal, &mut wire)?;
                        wire.push(self.syntax.end);
                        wire.push(noun);
                    } else {
                        // A lone END line: its BEGIN collected it above.
                        snafu::whatever!(
                            "line {number}: END {} closes no open serialization",
                            self.syntax.name_of(noun).unwrap_or("?")
                        );
                    }
                }
            }
        }
        snafu::ensure_whatever!(
            stack.is_empty(),
            "assembly ends with {} block(s) open",
            stack.len()
        );
        Ok(wire)
    }

    /// Collect a serialization interior through its END line. The
    /// terminator is recognized by POSITION - the only line that may
    /// sit at the BEGIN's own depth - so a data line spelling END at
    /// the data indent stays data. Non-empty interior lines carry
    /// the base indent exactly (stripped; a shortfall faults) and
    /// everything past it is data verbatim; a raw-string wrapper is
    /// honored at the base indent. The precision is the point
    /// (TheUser): the surface indent strips reliably, so no
    /// structural tabs or spaces reach the wire.
    fn collect_interior(
        &self,
        lines: &mut std::iter::Peekable<
            std::iter::Enumerate<std::str::Lines<'_>>,
        >,
        format: u32,
        depth: usize,
        begin_number: usize,
    ) -> BiquestResult<String> {
        let mut collected: Vec<String> = Vec::new();
        let mut raw_closer: Option<String> = None;
        let mut first = true;
        for (index, line) in lines.by_ref() {
            let number = index + 1;
            if let Some(closer) = &raw_closer {
                let content = strip_indent(line, depth, number)?;
                if content.trim() == closer.as_str() {
                    raw_closer = None;
                    continue;
                }
                collected.push(content.to_string());
                continue;
            }
            let leading = line.len() - line.trim_start_matches(' ').len();
            if !line.is_empty() && leading == INDENT.len() * (depth - 1) {
                if let Ok(AssemblyLine::Structural { head, noun }) =
                    self.parse_line(line, number)
                    && head == self.syntax.end
                {
                    snafu::ensure_whatever!(
                        noun == format,
                        "line {number}: END {} against BEGIN {}",
                        self.syntax.name_of(noun).unwrap_or("?"),
                        self.syntax.name_of(format).unwrap_or("?")
                    );
                    return Ok(collected.join("\n"));
                }
                snafu::whatever!(
                    "line {number}: a serialization admits only its END line at \
                     the parent depth"
                );
            }
            let content = strip_indent(line, depth, number)?;
            let trimmed = content.trim();
            if first
                && trimmed.starts_with('#')
                && trimmed.ends_with('{')
                && trimmed[..trimmed.len() - 1].chars().all(|c| c == '#')
            {
                let hashes = trimmed.len() - 1;
                raw_closer = Some(format!("}}{}", "#".repeat(hashes)));
                first = false;
                continue;
            }
            first = false;
            collected.push(content.to_string());
        }
        snafu::whatever!(
            "the serialization opened at line {begin_number} never ends (missing END {})",
            self.syntax.name_of(format).unwrap_or("?")
        )
    }

    /// Whether escape and unicode spans are legal here: inside TRAIN,
    /// within an INPUT block's serialization (TheUser's ruling, one
    /// shared predicate), with TRAIN and INPUT bound.
    fn spans_legal(&self, stack: &[u32]) -> bool {
        match (self.syntax.train, self.syntax.input) {
            (Some(train), Some(input)) => {
                stack.contains(&train) && stack.last() == Some(&input)
            }
            _ => false,
        }
    }

    /// Resolve a marker spelling's interior: a bound keyword name, a
    /// hardcoded operator's fixed name, or a two-hex-digit page
    /// address.
    fn spelling_id(&self, body: &str) -> Option<u32> {
        if let Some(id) = self.syntax.id_of(body) {
            return Some(id);
        }
        if let Some(&(id, _)) = HARDCODED_NAMES.iter().find(|(_, name)| *name == body) {
            return Some(id);
        }
        if body.len() == 2 && body.chars().all(|c| c.is_ascii_hexdigit()) {
            return u32::from_str_radix(body, 16).ok();
        }
        None
    }

    /// Encode a serialization interior: content tokenizes, and marker
    /// spellings are invalid except as span openers and inside spans.
    /// An ESCAPE span converts spellings to their keyword ids (prose
    /// rides along; nesting and an unterminated span fault); a
    /// UNICODE span forces per-character tokenization
    /// (encode_unicode_span).
    fn encode_interior(
        &self,
        interior: &str,
        spans_legal: bool,
        wire: &mut Vec<u32>,
    ) -> BiquestResult<()> {
        let escape = self.syntax.escape;
        let escaped = self.syntax.escaped;
        let mut rest = interior;
        let mut in_span = false;
        loop {
            let Some((start, end, body)) = find_marker_spelling(rest) else {
                snafu::ensure_whatever!(
                    !in_span,
                    "an escape span never closes (missing <|ESCAPED|>)"
                );
                if !rest.is_empty() {
                    for token in self.segmenter.segment(rest)? {
                        wire.push(token.id);
                    }
                }
                return Ok(());
            };
            let before = &rest[..start];
            if !before.is_empty() {
                for token in self.segmenter.segment(before)? {
                    wire.push(token.id);
                }
            }
            let Some(id) = self.spelling_id(body) else {
                snafu::whatever!("<|{body}|> is not a bound keyword or page address");
            };
            if !in_span {
                if id == KEYWORD_UNICODE {
                    snafu::ensure_whatever!(
                        spans_legal,
                        "a unicode span is legal only inside TRAIN, within an INPUT \
                         serialization"
                    );
                    wire.push(id);
                    rest = self.encode_unicode_span(&rest[end..], wire)?;
                    continue;
                }
                snafu::ensure_whatever!(
                    Some(id) == escape,
                    "a marker spelling (<|{body}|>) outside a span is invalid"
                );
                snafu::ensure_whatever!(
                    spans_legal,
                    "an escape span is legal only inside TRAIN, within an INPUT \
                     serialization"
                );
                snafu::ensure_whatever!(
                    escaped.is_some(),
                    "the syntax table lacks the ESCAPED binding"
                );
                wire.push(id);
                in_span = true;
            } else if Some(id) == escape {
                snafu::whatever!("an escape span cannot nest another <|ESCAPE|>");
            } else {
                wire.push(id);
                if Some(id) == escaped {
                    in_span = false;
                }
            }
            rest = &rest[end..];
        }
    }

    /// Escape control characters for the debugging table's value
    /// column - a visual representation, one row per token (TheUser).
    fn visual_text(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for c in text.chars() {
            match c {
                '\n' => out.push_str("\\n"),
                '\t' => out.push_str("\\t"),
                '\r' => out.push_str("\\r"),
                c if (c as u32) < 0x20 || c == '\u{7F}' => {
                    out.push_str(&format!("\\u{{{:02X}}}", c as u32));
                }
                c => out.push(c),
            }
        }
        out
    }

    /// The wire as tokenize-style rows: keyword ids as their mention
    /// spellings, character tokens with their U+ column, rows and
    /// words with their text - the assembler's test surface.
    fn wire_table_rows(&self, wire: &[u32]) -> BiquestResult<Vec<harness::nu::Value>> {
        let character_offset = KEYWORD_PAGE_SIZE;
        let keyboard_offset = character_offset + self.table.assigned_count() as u32;
        let mut rows = Vec::with_capacity(wire.len());
        for (position, &id) in wire.iter().enumerate() {
            let (unicode, value) = if id < KEYWORD_PAGE_SIZE {
                (harness::nu::Value::nothing(span()), self.mention_spelling(id))
            } else if id < keyboard_offset {
                let row = &self.table.rows[(id - character_offset) as usize];
                let Some(c) = char::from_u32(row.code_point) else {
                    snafu::whatever!(
                        "wire fault at token {position}: unrenderable code point \
                         U+{:04X}",
                        row.code_point
                    );
                };
                (v_str(&format!("U+{:04X}", c as u32)), c.to_string())
            } else {
                (
                    harness::nu::Value::nothing(span()),
                    self.content_text(id, position)?,
                )
            };
            rows.push(harness::nu::Value::record(
                harness::nu::record! {
                    "token" => v_int(id as i64),
                    "unicode" => unicode,
                    "value" => v_str(&Self::visual_text(&value)),
                },
                span(),
            ));
        }
        Ok(rows)
    }

    /// A character's layer token id, or the ingestion refusal.
    fn character_id(&self, c: char) -> BiquestResult<u32> {
        match self.table.index_of(c as u32) {
            Some(index) => Ok(KEYWORD_PAGE_SIZE + index),
            None => snafu::whatever!(
                "ingestion refusal: unassigned code point U+{:04X}",
                c as u32
            ),
        }
    }

    /// Encode a forced per-character span's content: every character
    /// until the UNICODED spelling is one character token - no
    /// dictionary matching, no case modifiers, no keyboard rows, no
    /// repeat bands (the point is the exact surface). UNICODED is the
    /// sole terminator: any other marker-shaped spelling char-splits
    /// as raw content; a span that never terminates faults. Returns
    /// the text after the terminator.
    fn encode_unicode_span<'t>(
        &self,
        text: &'t str,
        wire: &mut Vec<u32>,
    ) -> BiquestResult<&'t str> {
        let mut rest = text;
        loop {
            let Some((start, end, body)) = find_marker_spelling(rest) else {
                snafu::whatever!("a unicode span never closes (missing <|UNICODED|>)");
            };
            if self.spelling_id(body) == Some(KEYWORD_UNICODED) {
                for c in rest[..start].chars() {
                    wire.push(self.character_id(c)?);
                }
                wire.push(KEYWORD_UNICODED);
                return Ok(&rest[end..]);
            }
            for c in rest[..end].chars() {
                wire.push(self.character_id(c)?);
            }
            rest = &rest[end..];
        }
    }

    /// A keyword id's mention spelling: the bound name, the hardcoded
    /// operator's fixed name, else the two-hex page address.
    fn mention_spelling(&self, id: u32) -> String {
        if let Some(name) = self.syntax.name_of(id) {
            return format!("<|{name}|>");
        }
        match HARDCODED_NAMES.iter().find(|(hardcoded, _)| *hardcoded == id) {
            Some(&(_, name)) => format!("<|{name}|>"),
            None => format!("<|{id:02X}|>"),
        }
    }

    /// A case operator's dictionary follower, or a positioned fault.
    fn dictionary_row_text(&self, id: u32, position: usize) -> BiquestResult<&str> {
        let dictionary_offset = KEYWORD_PAGE_SIZE
            + self.table.assigned_count() as u32
            + self.buckets.count() as u32;
        let reserve_offset = dictionary_offset + self.admitted.len() as u32;
        snafu::ensure_whatever!(
            id >= dictionary_offset && id < reserve_offset,
            "wire fault at token {position}: a case token follows a dictionary \
             word only"
        );
        Ok(&self.admitted[(id - dictionary_offset) as usize])
    }

    /// A character's UCD simple uppercase, itself when unmapped.
    fn upper_char(&self, c: char) -> char {
        self.table
            .simple_uppercase
            .get(&(c as u32))
            .and_then(|&target| char::from_u32(target))
            .unwrap_or(c)
    }

    /// One content token id's text, or a positioned fault.
    fn content_text(&self, id: u32, position: usize) -> BiquestResult<String> {
        let character_offset = KEYWORD_PAGE_SIZE;
        let keyboard_offset = character_offset + self.table.assigned_count() as u32;
        let dictionary_offset = keyboard_offset + self.buckets.count() as u32;
        let reserve_offset = dictionary_offset + self.admitted.len() as u32;
        if id < character_offset {
            snafu::whatever!(
                "wire fault at token {position}: keyword id {id} inside a serialization"
            );
        }
        if id < keyboard_offset {
            let row = &self.table.rows[(id - character_offset) as usize];
            match char::from_u32(row.code_point) {
                Some(c) => Ok(c.to_string()),
                None => snafu::whatever!(
                    "wire fault at token {position}: unrenderable code point U+{:04X}",
                    row.code_point
                ),
            }
        } else if id < dictionary_offset {
            Ok(self
                .buckets
                .entry((id - keyboard_offset) as usize)
                .sequence())
        } else if id < reserve_offset {
            Ok(self.admitted[(id - dictionary_offset) as usize].clone())
        } else {
            snafu::whatever!("wire fault at token {position}: id {id} is beyond the vocabulary")
        }
    }

    /// Whether a content token may head a repetition run: a
    /// character-layer token that is not a Digit (a number is
    /// place-value content, never repetition), and never a keyboard
    /// row (rows never compose with the operators).
    fn repeatable(&self, id: u32) -> bool {
        let character_offset = KEYWORD_PAGE_SIZE;
        let keyboard_offset = character_offset + self.table.assigned_count() as u32;
        if id < character_offset || id >= keyboard_offset {
            return false;
        }
        let row = &self.table.rows[(id - character_offset) as usize];
        self.table.class_of_row(row) != CharClass::Digit
    }

    /// A count digit's value off a character-layer token.
    fn digit_value(&self, id: u32, position: usize) -> BiquestResult<u32> {
        let text = self.content_text(id, position)?;
        let mut chars = text.chars();
        match (chars.next(), chars.next()) {
            (Some(digit), None) if digit.is_ascii_digit() => {
                Ok(digit as u32 - '0' as u32)
            }
            _ => snafu::whatever!(
                "wire fault at token {position}: a repetition count wants a digit, got {text:?}"
            ),
        }
    }

    /// Decode a serialization interior: band expansion lives here,
    /// REPETITION's wire illegality faults here, escape spans render
    /// their keyword ids back as mention spellings, and unicode spans
    /// render their characters verbatim between the two markers.
    /// Returns the interior text and the position of the END keyword.
    fn decode_interior(
        &self,
        wire: &[u32],
        mut position: usize,
        spans_legal: bool,
    ) -> BiquestResult<(String, usize)> {
        let mut interior = String::new();
        let mut in_span = false;
        while position < wire.len() {
            let id = wire[position];
            if in_span && id < KEYWORD_PAGE_SIZE {
                if Some(id) == self.syntax.escape {
                    snafu::whatever!(
                        "wire fault at token {position}: an escape span cannot nest \
                         another <|ESCAPE|>"
                    );
                }
                interior.push_str(&self.mention_spelling(id));
                if Some(id) == self.syntax.escaped {
                    in_span = false;
                }
                position += 1;
                continue;
            }
            if Some(id) == self.syntax.escape {
                snafu::ensure_whatever!(
                    spans_legal,
                    "wire fault at token {position}: an escape span is legal only \
                     inside TRAIN, within an INPUT serialization"
                );
                interior.push_str(&self.mention_spelling(id));
                in_span = true;
                position += 1;
                continue;
            }
            if id == KEYWORD_UNICODE {
                snafu::ensure_whatever!(
                    spans_legal,
                    "wire fault at token {position}: a unicode span is legal only \
                     inside TRAIN, within an INPUT serialization"
                );
                interior.push_str(&self.mention_spelling(id));
                position += 1;
                // The span's characters render verbatim: no band
                // expansion, no case handling - the markers bracket
                // an exact surface.
                let keyboard_offset =
                    KEYWORD_PAGE_SIZE + self.table.assigned_count() as u32;
                loop {
                    let Some(&span_id) = wire.get(position) else {
                        snafu::whatever!(
                            "wire fault at token {position}: a unicode span never \
                             closes (missing <|UNICODED|>)"
                        );
                    };
                    if span_id == KEYWORD_UNICODED {
                        interior.push_str(&self.mention_spelling(span_id));
                        position += 1;
                        break;
                    }
                    snafu::ensure_whatever!(
                        span_id >= KEYWORD_PAGE_SIZE,
                        "wire fault at token {position}: a unicode span never \
                         closes (missing <|UNICODED|>)"
                    );
                    snafu::ensure_whatever!(
                        span_id < keyboard_offset,
                        "wire fault at token {position}: a unicode span admits \
                         character tokens only"
                    );
                    let row = &self.table.rows[(span_id - KEYWORD_PAGE_SIZE) as usize];
                    let Some(c) = char::from_u32(row.code_point) else {
                        snafu::whatever!(
                            "wire fault at token {position}: unrenderable code \
                             point U+{:04X}",
                            row.code_point
                        );
                    };
                    interior.push(c);
                    position += 1;
                }
                continue;
            }
            if id == KEYWORD_REPETITION {
                snafu::whatever!(
                    "wire fault at token {position}: <|repetition|> is an \
                     AbstractConceptMarker and never legal wire"
                );
            }
            if id == KEYWORD_CASE {
                snafu::whatever!(
                    "wire fault at token {position}: <|case|> is an \
                     AbstractConceptMarker and never legal wire"
                );
            }
            if id == KEYWORD_CASED || id == KEYWORD_CAPITALIZED || id == KEYWORD_UPPERCASED {
                snafu::whatever!(
                    "wire fault at token {position}: a case operator with no word \
                     before it"
                );
            }
            if id == KEYWORD_REPEAT || id == KEYWORD_BEGIN_REPEAT || id == KEYWORD_END_REPEAT {
                snafu::whatever!(
                    "wire fault at token {position}: a repetition operator with no unit \
                     before it"
                );
            }
            if id == KEYWORD_UNICODED {
                snafu::whatever!(
                    "wire fault at token {position}: a unicode span closer with no \
                     open span"
                );
            }
            if id < KEYWORD_PAGE_SIZE {
                return Ok((interior, position));
            }
            let unit = self.content_text(id, position)?;
            let after = wire.get(position + 1).copied();
            if after == Some(KEYWORD_CAPITALIZED) || after == Some(KEYWORD_UPPERCASED) {
                // Postfix (the match first - order matters for the
                // model): the row, then its case token.
                let word = self.dictionary_row_text(id, position)?.to_string();
                if after == Some(KEYWORD_UPPERCASED) {
                    for c in word.chars() {
                        interior.push(self.upper_char(c));
                    }
                } else {
                    let mut chars = word.chars();
                    if let Some(first) = chars.next() {
                        interior.push(self.upper_char(first));
                        interior.push_str(chars.as_str());
                    }
                }
                position += 2;
                continue;
            }
            if after == Some(KEYWORD_CASED) {
                // The overlay is bounded by the row's length in
                // DECODED characters. A character token folds to its
                // row character; a dictionary token stands in for its
                // folded slice, its optional case operator shaping
                // the rendered surface (TheUser's combination
                // design: microsoft CASED micro CAPITALIZED soft
                // CAPITALIZED).
                let word = self.dictionary_row_text(id, position)?.to_string();
                let expected: Vec<char> = word.chars().collect();
                let mut consumed = 0usize;
                let mut cursor = position + 2;
                let character_offset = KEYWORD_PAGE_SIZE;
                let keyboard_offset =
                    character_offset + self.table.assigned_count() as u32;
                let dictionary_offset =
                    keyboard_offset + self.buckets.count() as u32;
                let reserve_offset = dictionary_offset + self.admitted.len() as u32;
                while consumed < expected.len() {
                    let Some(&overlay_id) = wire.get(cursor) else {
                        snafu::whatever!(
                            "wire fault at token {cursor}: a cased overlay ends \
                             before its row's length"
                        );
                    };
                    if overlay_id >= character_offset && overlay_id < keyboard_offset {
                        let row =
                            &self.table.rows[(overlay_id - character_offset) as usize];
                        let Some(c) = char::from_u32(row.code_point) else {
                            snafu::whatever!(
                                "wire fault at token {cursor}: unrenderable code point"
                            );
                        };
                        snafu::ensure_whatever!(
                            self.table.fold(c) == Some(expected[consumed]),
                            "wire fault at token {cursor}: a cased overlay character \
                             does not fold to its row's character"
                        );
                        interior.push(c);
                        consumed += 1;
                        cursor += 1;
                        continue;
                    }
                    if overlay_id >= dictionary_offset && overlay_id < reserve_offset {
                        let text = self.admitted
                            [(overlay_id - dictionary_offset) as usize]
                            .clone();
                        let piece_length = text.chars().count();
                        let matches_slice = consumed + piece_length <= expected.len()
                            && text
                                .chars()
                                .zip(expected[consumed..consumed + piece_length].iter())
                                .all(|(piece_char, &slice_char)| piece_char == slice_char);
                        snafu::ensure_whatever!(
                            matches_slice,
                            "wire fault at token {cursor}: a cased overlay word does \
                             not match its row's slice"
                        );
                        let operator = wire.get(cursor + 1).copied();
                        if operator == Some(KEYWORD_UPPERCASED) {
                            for c in text.chars() {
                                interior.push(self.upper_char(c));
                            }
                            cursor += 2;
                        } else if operator == Some(KEYWORD_CAPITALIZED) {
                            let mut chars = text.chars();
                            if let Some(first) = chars.next() {
                                interior.push(self.upper_char(first));
                                interior.push_str(chars.as_str());
                            }
                            cursor += 2;
                        } else {
                            interior.push_str(&text);
                            cursor += 1;
                        }
                        consumed += piece_length;
                        continue;
                    }
                    snafu::whatever!(
                        "wire fault at token {cursor}: a cased overlay wants \
                         character or dictionary tokens"
                    );
                }
                position = cursor;
                continue;
            }
            if after == Some(KEYWORD_REPEAT) {
                snafu::ensure_whatever!(
                    self.repeatable(id),
                    "wire fault at token {}: REPEAT after a non-repeatable token",
                    position + 1
                );
                let Some(&count_id) = wire.get(position + 2) else {
                    snafu::whatever!(
                        "wire fault at token {}: REPEAT with no count digit",
                        position + 1
                    );
                };
                let count = self.digit_value(count_id, position + 2)?;
                snafu::ensure_whatever!(
                    count >= 3,
                    "wire fault at token {}: a REPEAT count below the band (got {count})",
                    position + 2
                );
                for _ in 0..count {
                    interior.push_str(&unit);
                }
                position += 3;
                continue;
            }
            if after == Some(KEYWORD_BEGIN_REPEAT) {
                snafu::ensure_whatever!(
                    self.repeatable(id),
                    "wire fault at token {}: BEGIN_REPEAT after a non-repeatable token",
                    position + 1
                );
                let mut cursor = position + 2;
                let mut count = 0u32;
                let mut digits = 0usize;
                while cursor < wire.len() && wire[cursor] != KEYWORD_END_REPEAT {
                    count = count * 10 + self.digit_value(wire[cursor], cursor)?;
                    digits += 1;
                    snafu::ensure_whatever!(
                        digits <= 9,
                        "wire fault at token {cursor}: a bracketed count past nine digits"
                    );
                    cursor += 1;
                }
                snafu::ensure_whatever!(
                    cursor < wire.len(),
                    "wire fault at token {}: BEGIN_REPEAT never ends",
                    position + 1
                );
                snafu::ensure_whatever!(
                    digits > 0,
                    "wire fault at token {}: BEGIN_REPEAT with no count digits",
                    position + 1
                );
                snafu::ensure_whatever!(
                    count >= 10,
                    "wire fault at token {}: a bracketed count below the band (got {count})",
                    position + 1
                );
                for _ in 0..count {
                    interior.push_str(&unit);
                }
                position = cursor + 1;
                continue;
            }
            interior.push_str(&unit);
            position += 1;
        }
        snafu::whatever!("wire fault at token {position}: a serialization never ends")
    }

    /// The wire back to assembly: indentation regenerates, structure
    /// checks as it renders, interiors re-wrap in raw strings where
    /// their lines would read as assembly.
    pub(crate) fn decode(&self, wire: &[u32]) -> BiquestResult<String> {
        let mut out = String::new();
        let mut stack: Vec<u32> = Vec::new();
        let mut position = 0usize;
        let keyword_name = |id: u32, position: usize| -> BiquestResult<&str> {
            match self.syntax.name_of(id) {
                Some(name) => Ok(name),
                None => snafu::whatever!(
                    "wire fault at token {position}: keyword id {id} is unbound"
                ),
            }
        };
        while position < wire.len() {
            let id = wire[position];
            if id == KEYWORD_REPETITION {
                snafu::whatever!(
                    "wire fault at token {position}: <|repetition|> is an \
                     AbstractConceptMarker and never legal wire"
                );
            }
            if id >= KEYWORD_PAGE_SIZE {
                snafu::whatever!(
                    "wire fault at token {position}: content id {id} outside a serialization"
                );
            }
            if id >= HARDCODED_BLOCK_FLOOR {
                snafu::whatever!(
                    "wire fault at token {position}: a hardcoded operator or marker \
                     outside a serialization"
                );
            }
            let indent = INDENT.repeat(stack.len());
            if id == self.syntax.open || id == self.syntax.close || id == self.syntax.begin {
                let Some(&noun) = wire.get(position + 1) else {
                    snafu::whatever!(
                        "wire fault at token {position}: {} with no noun",
                        keyword_name(id, position)?
                    );
                };
                let noun_name = keyword_name(noun, position + 1)?.to_string();
                if id == self.syntax.open {
                    out.push_str(&format!("{indent}{HEAD_OPEN} {noun_name}\n"));
                    stack.push(noun);
                    position += 2;
                } else if id == self.syntax.close {
                    let Some(expected) = stack.pop() else {
                        snafu::whatever!(
                            "wire fault at token {position}: CLOSE {noun_name} with \
                             nothing open"
                        );
                    };
                    snafu::ensure_whatever!(
                        expected == noun,
                        "wire fault at token {}: CLOSE {noun_name} against open {}",
                        position + 1,
                        self.syntax.name_of(expected).unwrap_or("?")
                    );
                    let indent = INDENT.repeat(stack.len());
                    out.push_str(&format!("{indent}{HEAD_CLOSE} {noun_name}\n"));
                    position += 2;
                } else {
                    out.push_str(&format!("{indent}{HEAD_BEGIN} {noun_name}\n"));
                    let spans_legal = self.spans_legal(&stack);
                    let (interior, end_position) =
                        self.decode_interior(wire, position + 2, spans_legal)?;
                    snafu::ensure_whatever!(
                        wire[end_position] == self.syntax.end,
                        "wire fault at token {end_position}: a serialization interrupted \
                         by {} (only END closes it)",
                        keyword_name(wire[end_position], end_position)?
                    );
                    let Some(&end_noun) = wire.get(end_position + 1) else {
                        snafu::whatever!(
                            "wire fault at token {end_position}: END with no noun"
                        );
                    };
                    snafu::ensure_whatever!(
                        end_noun == noun,
                        "wire fault at token {}: END {} against BEGIN {noun_name}",
                        end_position + 1,
                        keyword_name(end_noun, end_position + 1)?
                    );
                    let content_indent = INDENT.repeat(stack.len() + 1);
                    // A blank interior line stays byte-empty: indenting
                    // it would decode a whitespace line the author
                    // never wrote.
                    let indented = |line: &str| -> String {
                        if line.is_empty() {
                            String::from("\n")
                        } else {
                            format!("{content_indent}{line}\n")
                        }
                    };
                    if interior_needs_raw(&interior) {
                        let (opener, closer) = raw_delimiters(&interior);
                        out.push_str(&format!("{content_indent}{opener}\n"));
                        for line in interior.split('\n') {
                            out.push_str(&indented(line));
                        }
                        out.push_str(&format!("{content_indent}{closer}\n"));
                    } else if !interior.is_empty() {
                        for line in interior.split('\n') {
                            out.push_str(&indented(line));
                        }
                    }
                    out.push_str(&format!("{indent}{HEAD_END} {noun_name}\n"));
                    position = end_position + 2;
                }
            } else if id == self.syntax.end {
                snafu::whatever!(
                    "wire fault at token {position}: a stray END (unopened closer)"
                );
            } else {
                out.push_str(&format!("{indent}{}\n", keyword_name(id, position)?));
                position += 1;
            }
        }
        snafu::ensure_whatever!(
            stack.is_empty(),
            "wire fault at token {position}: the wire ends with {} block(s) open",
            stack.len()
        );
        Ok(out)
    }
}

/// The first `<|BODY|>` marker spelling in a text: its byte range and
/// interior. Bodies are keyword names or page addresses - short runs
/// of name characters, never crossing a line.
fn find_marker_spelling(text: &str) -> Option<(usize, usize, &str)> {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    while index + 4 <= bytes.len() {
        if bytes[index] == b'<' && bytes[index + 1] == b'|' {
            let body_start = index + 2;
            let mut cursor = body_start;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric()
                    || bytes[cursor] == b'_'
                    || bytes[cursor] == b'-')
            {
                cursor += 1;
            }
            if cursor > body_start
                && cursor + 2 <= bytes.len()
                && bytes[cursor] == b'|'
                && bytes[cursor + 1] == b'>'
            {
                return Some((index, cursor + 2, &text[body_start..cursor]));
            }
        }
        index += 1;
    }
    None
}

/// Shared verb setup: the embedded layers plus table and dictionary.
struct AssemblerParts {
    table: CharacterTable,
    buckets: BucketTable,
    admitted: Vec<String>,
    syntax: SyntaxTable,
}

fn assembler_parts(
    syntax_path: Option<&PathBuf>,
    words_path: Option<&PathBuf>,
) -> BiquestResult<AssemblerParts> {
    Ok(AssemblerParts {
        table: CharacterTable::embedded()?,
        buckets: BucketTable::new(),
        admitted: match words_path {
            Some(path) => read_words_ordered(path)?,
            None => crate::dictionary::embedded_words(),
        },
        syntax: match syntax_path {
            Some(path) => SyntaxTable::load(path)?,
            None => SyntaxTable::embedded()?,
        },
    })
}

/// `biquest assembler`: TheUser's test surface - one file path in,
/// the tokenize-style [token, unicode, value] table out, everything
/// else embedded defaults.
pub(crate) fn assembler_table(args: &AssemblerArgs) -> BiquestResult<()> {
    let parts = assembler_parts(None, None)?;
    let segmenter = Segmenter::new(&parts.table, &parts.buckets, &parts.admitted);
    let assembler = Assembler::new(
        &parts.syntax,
        &segmenter,
        &parts.table,
        &parts.buckets,
        &parts.admitted,
    );
    let text = fs::read_to_string(&args.file)?;
    let wire = assembler.encode(&text)?;
    let rows = assembler.wire_table_rows(&wire)?;
    let rendered =
        harness::nu::to_nuon_pretty(&harness::nu::Value::list(rows, span()))?;
    println!("{rendered}");
    Ok(())
}

/// `biquest assemble`: assembly text to the wire id list, as NUON.
pub(crate) fn assemble_text(args: &AssembleArgs) -> BiquestResult<()> {
    let parts = assembler_parts(args.syntax.as_ref(), args.words.as_ref())?;
    let segmenter = Segmenter::new(&parts.table, &parts.buckets, &parts.admitted);
    let assembler = Assembler::new(
        &parts.syntax,
        &segmenter,
        &parts.table,
        &parts.buckets,
        &parts.admitted,
    );
    let text = match &args.file {
        Some(path) => fs::read_to_string(path)?,
        None => match &args.text {
            Some(text) => text.clone(),
            None => snafu::whatever!("assemble wants TEXT or --file"),
        },
    };
    let wire = assembler.encode(&text)?;
    let rendered = harness::nu::to_nuon_text(&harness::nu::Value::list(
        wire.iter().map(|&id| v_int(id as i64)).collect(),
        span(),
    ))?;
    println!("{rendered}");
    Ok(())
}

/// `biquest disassemble`: a NUON wire id list back to assembly text.
pub(crate) fn disassemble_wire(args: &DisassembleArgs) -> BiquestResult<()> {
    let parts = assembler_parts(args.syntax.as_ref(), args.words.as_ref())?;
    let segmenter = Segmenter::new(&parts.table, &parts.buckets, &parts.admitted);
    let assembler = Assembler::new(
        &parts.syntax,
        &segmenter,
        &parts.table,
        &parts.buckets,
        &parts.admitted,
    );
    let text = match &args.file {
        Some(path) => fs::read_to_string(path)?,
        None => match &args.ids {
            Some(ids) => ids.clone(),
            None => snafu::whatever!("disassemble wants IDS or --file"),
        },
    };
    let value = harness::nu::from_nuon_text(&text)?;
    let rows = match value.as_list() {
        Ok(rows) => rows,
        Err(e) => snafu::whatever!("the wire wants a NUON int list: {e}"),
    };
    let mut wire = Vec::with_capacity(rows.len());
    for row in rows {
        match row.as_int() {
            Ok(id) if (0..=u32::MAX as i64).contains(&id) => wire.push(id as u32),
            Ok(id) => snafu::whatever!("wire id {id} is outside u32"),
            Err(e) => snafu::whatever!("wire id: {e}"),
        }
    }
    print!("{}", assembler.decode(&wire)?);
    Ok(())
}
