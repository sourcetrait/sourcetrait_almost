//! The QuillAssembler: assembly text to wire tokens and back.
use crate::*;

use crate::bucket::BucketTable;
use crate::dictionary::read_words_ordered;
use crate::lexer::KEYWORD_BEGIN_REPEAT;
use crate::lexer::KEYWORD_END_REPEAT;
use crate::lexer::KEYWORD_PAGE_SIZE;
use crate::lexer::KEYWORD_REPEAT;
use crate::lexer::KEYWORD_REPETITION;
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
}

impl SyntaxTable {
    /// Build from the table's name column, in page order.
    pub(crate) fn from_names(names: Vec<String>) -> BiquestResult<Self> {
        snafu::ensure_whatever!(
            names.len() <= KEYWORD_BEGIN_REPEAT as usize,
            "the syntax table carries {} bindings; the page holds {} below the \
             hardcoded operator block",
            names.len(),
            KEYWORD_BEGIN_REPEAT
        );
        let mut ids = HashMap::with_capacity(names.len());
        for (index, name) in names.iter().enumerate() {
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
            names,
            ids,
        })
    }

    /// Load the `> HUMAN` draft table: a NUON table with a name column.
    pub(crate) fn load(path: &Path) -> BiquestResult<Self> {
        let value = harness::nu::load_value(path)?;
        let rows = match value.as_list() {
            Ok(rows) => rows,
            Err(e) => snafu::whatever!("{}: not a table: {e}", path.display()),
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

/// Strip up to `depth` indentation levels off a content line; excess
/// indentation is the content's own.
fn strip_indent(line: &str, depth: usize) -> &str {
    let mut rest = line;
    for _ in 0..depth {
        match rest.strip_prefix(INDENT) {
            Some(stripped) => rest = stripped,
            None => break,
        }
    }
    rest
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

    /// Assembly text to the wire: indentation drops, structure checks
    /// as it encodes, serialization interiors pass through content
    /// tokenization (raw-string wrappers strip).
    pub(crate) fn encode(&self, text: &str) -> BiquestResult<Vec<u32>> {
        let mut wire: Vec<u32> = Vec::new();
        let mut stack: Vec<u32> = Vec::new();
        let mut lines = text.lines().enumerate().peekable();
        while let Some((index, line)) = lines.next() {
            let number = index + 1;
            match self.parse_line(line, number)? {
                AssemblyLine::Blank => {}
                AssemblyLine::Bare(id) => wire.push(id),
                AssemblyLine::Structural { head, noun } => {
                    wire.push(head);
                    wire.push(noun);
                    if head == self.syntax.open {
                        stack.push(noun);
                    } else if head == self.syntax.close {
                        let Some(expected) = stack.pop() else {
                            snafu::whatever!("line {number}: CLOSE with nothing open");
                        };
                        snafu::ensure_whatever!(
                            expected == noun,
                            "line {number}: CLOSE {} against open {}",
                            self.syntax.name_of(noun).unwrap_or("?"),
                            self.syntax.name_of(expected).unwrap_or("?")
                        );
                    } else if head == self.syntax.begin {
                        let depth = stack.len() + 1;
                        let interior =
                            self.collect_interior(&mut lines, noun, depth, number)?;
                        for token in self.segmenter.segment(&interior)? {
                            wire.push(token.id);
                        }
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

    /// Collect a serialization interior through its END line, indent
    /// stripped, raw-string wrapper honored.
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
            let content = strip_indent(line, depth);
            if let Some(closer) = &raw_closer {
                if content.trim() == closer.as_str() {
                    raw_closer = None;
                    continue;
                }
                collected.push(content.to_string());
                continue;
            }
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
            // The END line terminates the interior; anything else,
            // structural-looking or not, is content by position.
            if let Ok(AssemblyLine::Structural { head, noun }) = self.parse_line(line, number)
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
            collected.push(content.to_string());
        }
        snafu::whatever!(
            "the serialization opened at line {begin_number} never ends (missing END {})",
            self.syntax.name_of(format).unwrap_or("?")
        )
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

    /// Decode a serialization interior: band expansion lives here, and
    /// REPETITION's wire illegality faults here. Returns the interior
    /// text and the position of the END keyword.
    fn decode_interior(
        &self,
        wire: &[u32],
        mut position: usize,
    ) -> BiquestResult<(String, usize)> {
        let mut interior = String::new();
        while position < wire.len() {
            let id = wire[position];
            if id == KEYWORD_REPETITION {
                snafu::whatever!(
                    "wire fault at token {position}: <|repetition|> is an \
                     AbstractConceptMarker and never legal wire"
                );
            }
            if id == KEYWORD_REPEAT || id == KEYWORD_BEGIN_REPEAT || id == KEYWORD_END_REPEAT {
                snafu::whatever!(
                    "wire fault at token {position}: a repetition operator with no unit \
                     before it"
                );
            }
            if id < KEYWORD_PAGE_SIZE {
                return Ok((interior, position));
            }
            let unit = self.content_text(id, position)?;
            let after = wire.get(position + 1).copied();
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
            if id >= KEYWORD_BEGIN_REPEAT {
                snafu::whatever!(
                    "wire fault at token {position}: a repetition operator outside a \
                     serialization"
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
                    let (interior, end_position) =
                        self.decode_interior(wire, position + 2)?;
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
                    if interior_needs_raw(&interior) {
                        let (opener, closer) = raw_delimiters(&interior);
                        out.push_str(&format!("{content_indent}{opener}\n"));
                        for line in interior.split('\n') {
                            out.push_str(&format!("{content_indent}{line}\n"));
                        }
                        out.push_str(&format!("{content_indent}{closer}\n"));
                    } else if !interior.is_empty() {
                        for line in interior.split('\n') {
                            out.push_str(&format!("{content_indent}{line}\n"));
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

/// Shared verb setup: the embedded layers plus table and dictionary.
struct AssemblerParts {
    table: CharacterTable,
    buckets: BucketTable,
    admitted: Vec<String>,
    syntax: SyntaxTable,
}

fn assembler_parts(
    syntax_path: &Path,
    words_path: Option<&PathBuf>,
) -> BiquestResult<AssemblerParts> {
    Ok(AssemblerParts {
        table: CharacterTable::embedded()?,
        buckets: BucketTable::new(),
        admitted: match words_path {
            Some(path) => read_words_ordered(path)?,
            None => crate::dictionary::embedded_words(),
        },
        syntax: SyntaxTable::load(syntax_path)?,
    })
}

/// `biquest assemble`: assembly text to the wire id list, as NUON.
pub(crate) fn assemble_text(args: &AssembleArgs) -> BiquestResult<()> {
    let parts = assembler_parts(&args.syntax, args.words.as_ref())?;
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
    let parts = assembler_parts(&args.syntax, args.words.as_ref())?;
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
