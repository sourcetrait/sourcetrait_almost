//! The typed-channel block grammar: tags, framing, and diagnostics.
use crate::*;

/// The reserved anchor the config channel rides.
pub const TOKEN_EXTRA_ID_0: u32 = 100_256;
/// The tool-marker pair the boundary aliases until routing is trained.
pub const TOKEN_FUNCTION_CALLS_OPEN: u32 = 100_268;
pub const TOKEN_FUNCTION_CALLS_CLOSE: u32 = 100_269;
/// The first of the contiguous opener run (`extra_id_1`).
pub const TOKEN_EXTRA_ID_1: u32 = 100_270;
/// The insufficiency signal; special, so detect by id and never by text.
pub const TOKEN_ENDOFPROMPT: u32 = 100_276;

/// A channel tag. The id is what is real; the spelling is data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    Config,
    Input,
    Output,
    Pass,
    Nu,
    Liquid,
    Close,
}

/// The tags in emission order, closer last.
pub const TAGS: [Tag; 7] = [
    Tag::Config,
    Tag::Input,
    Tag::Output,
    Tag::Pass,
    Tag::Nu,
    Tag::Liquid,
    Tag::Close,
];

impl Tag {
    /// The reserved token id this tag occupies; the binding fact.
    pub fn token_id(&self) -> u32 {
        match self {
            Self::Config => TOKEN_EXTRA_ID_0,
            Self::Input => TOKEN_EXTRA_ID_1,
            Self::Output => TOKEN_EXTRA_ID_1 + 1,
            Self::Pass => TOKEN_EXTRA_ID_1 + 2,
            Self::Nu => TOKEN_EXTRA_ID_1 + 3,
            Self::Liquid => TOKEN_EXTRA_ID_1 + 4,
            Self::Close => TOKEN_EXTRA_ID_1 + 5,
        }
    }

    /// What the shipped tokenizer encodes to this tag's id today.
    pub fn spelling(&self) -> &'static str {
        match self {
            Self::Config => "<|extra_id_0|>",
            Self::Input => "<|extra_id_1|>",
            Self::Output => "<|extra_id_2|>",
            Self::Pass => "<|extra_id_3|>",
            Self::Nu => "<|extra_id_4|>",
            Self::Liquid => "<|extra_id_5|>",
            Self::Close => "<|extra_id_6|>",
        }
    }

    /// The authoring alias for this tag's wire token; human-only, never a token.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Config => "<|config|>",
            Self::Input => "<|input|>",
            Self::Output => "<|output|>",
            Self::Pass => "<|pass|>",
            Self::Nu => "<|nu|>",
            Self::Liquid => "<|liquid|>",
            Self::Close => "<|/|>",
        }
    }

    /// Whether this tag opens a block; only the closer does not.
    pub fn opens(&self) -> bool {
        !matches!(self, Self::Close)
    }

    /// Whether this tag renders opener, payload and closer on one line.
    pub fn inline(&self) -> bool {
        matches!(self, Self::Pass)
    }
}

/// The word a `nu`-format block's type slot carries.
const TYPEDEF_SLOT: &str = "type";

/// The notation a data block's content is in; closed to what we read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Nuon,
    Nu,
    String,
}

/// The formats in the order a diagnostic lists them.
pub const FORMATS: [Format; 3] = [Format::Nuon, Format::Nu, Format::String];

impl Format {
    pub fn spelling(&self) -> &'static str {
        match self {
            Self::Nuon => "nuon",
            Self::Nu => "nu",
            Self::String => "string",
        }
    }

    /// Read one back; anything outside the vocabulary is refused.
    pub fn parse(spelling: &str) -> HarnessQuestResult<Self> {
        match FORMATS.into_iter().find(|format| format.spelling() == spelling) {
            Some(format) => Ok(format),
            None => snafu::whatever!(
                "`{spelling}` is not a format; a data block carries {}",
                FORMATS.map(|format| format.spelling()).join(", ")
            ),
        }
    }
}

/// What a data block's type slot says about its content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Declared {
    /// A nushell type the content conforms to.
    Conforms(nu::Type),
    /// The content IS a typedef, carried as a string for now.
    Typedef,
    /// Nothing to check, because the content is text.
    Untyped,
}

/// A data block's opener: the format, then what the type slot says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub format: Format,
    pub declared: Declared,
}

impl Descriptor {
    /// The descriptor a value travels under as NUON.
    pub fn nuon(declared: nu::Type) -> Self {
        Self {
            format: Format::Nuon,
            declared: Declared::Conforms(declared),
        }
    }

    /// The descriptor a typedef travels under.
    pub fn typedef() -> Self {
        Self {
            format: Format::Nu,
            declared: Declared::Typedef,
        }
    }

    /// The descriptor plain text travels under.
    pub fn text() -> Self {
        Self {
            format: Format::String,
            declared: Declared::Untyped,
        }
    }

    /// Read an opener's header; the pairing is part of the vocabulary.
    pub fn parse(header: &str) -> HarnessQuestResult<Self> {
        let header = header.trim();
        snafu::ensure_whatever!(
            !header.is_empty(),
            "a data block declares a format before its type"
        );
        let (word, rest) = match header.split_once(char::is_whitespace) {
            Some((word, rest)) => (word, rest.trim()),
            None => (header, ""),
        };
        let format = Format::parse(word)?;
        let declared = match format {
            Format::Nuon => {
                snafu::ensure_whatever!(
                    !rest.is_empty(),
                    "`nuon` declares the type its content conforms to"
                );
                Declared::Conforms(nu::parse_typedef(rest)?)
            }
            Format::Nu => {
                snafu::ensure_whatever!(
                    rest == TYPEDEF_SLOT,
                    "`nu` carries `{TYPEDEF_SLOT}`; got {rest:?}"
                );
                Declared::Typedef
            }
            Format::String => {
                snafu::ensure_whatever!(
                    rest.is_empty(),
                    "`string` carries no type; got {rest:?}"
                );
                Declared::Untyped
            }
        };
        Ok(Self { format, declared })
    }

    /// The header this descriptor renders as.
    pub fn render(&self) -> String {
        match &self.declared {
            Declared::Conforms(declared) => {
                format!("{} {declared}", self.format.spelling())
            }
            Declared::Typedef => format!("{} {TYPEDEF_SLOT}", self.format.spelling()),
            Declared::Untyped => self.format.spelling().to_string(),
        }
    }
}

/// Whether the boundary accepts the trained tool-marker attractor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aliasing {
    /// The grammar as locked; anything else is a protocol error.
    Strict,
    /// Also accept `function_calls` and an unmarked body as output.
    ToolMarkers,
}

/// One parsed block: its tag, its opener-line header, and its content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub tag: Tag,
    /// The rest of the opener line: a typedef, a mode, or empty.
    pub header: String,
    pub content: String,
}

impl Block {
    pub fn new(tag: Tag, header: &str, content: &str) -> Self {
        Self {
            tag,
            header: header.trim().to_string(),
            content: content.to_string(),
        }
    }
}

/// Escape the newlines a compact NUON render leaves raw.
pub fn escape_content(content: &str) -> String {
    content.replace('\\', "\\\\").replace('\r', "\\r").replace('\n', "\\n")
}

/// The inverse of escape_content, left-to-right so `\\n` survives.
pub fn unescape_content(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut chars = content.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Render one block in its canonical form; `pass` renders inline.
pub fn render_block(block: &Block) -> String {
    if block.tag.inline() {
        return format!(
            "{}{}{}",
            block.tag.spelling(),
            block.content,
            Tag::Close.spelling()
        );
    }
    let mut out = String::new();
    out.push_str(block.tag.spelling());
    if !block.header.is_empty() {
        out.push(' ');
        out.push_str(&block.header);
    }
    out.push('\n');
    if !block.content.is_empty() {
        out.push_str(&block.content);
        out.push('\n');
    }
    out.push_str(Tag::Close.spelling());
    out
}

/// Render a block sequence, one block per line-anchored run.
pub fn render_blocks(blocks: &[Block]) -> String {
    blocks
        .iter()
        .map(render_block)
        .collect::<Vec<String>>()
        .join("\n")
}

/// Translate authoring aliases to their wire token spellings.
pub fn authoring_to_wire(text: &str) -> String {
    let mut out = text.to_string();
    for tag in TAGS {
        out = out.replace(tag.name(), tag.spelling());
    }
    out
}

/// Whether text carries any channel-token spelling; the prose guard.
pub fn carries_marker(text: &str) -> bool {
    TAGS.iter().any(|tag| text.contains(tag.spelling()))
}

/// Parse a decoded turn into its blocks, line-anchored.
pub fn parse_blocks(text: &str, aliasing: Aliasing) -> HarnessQuestResult<Vec<Block>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut blocks = Vec::new();
    let mut index = 0usize;
    let mut unmarked: Vec<&str> = Vec::new();

    while index < lines.len() {
        let line = lines[index];
        let Some((tag, rest)) = opener_at(line, aliasing) else {
            if !line.trim().is_empty() {
                unmarked.push(line);
            }
            index += 1;
            continue;
        };
        if !unmarked.is_empty() {
            match aliasing {
                Aliasing::ToolMarkers => unmarked.clear(),
                Aliasing::Strict => snafu::whatever!(
                    "unmarked content before a block opener at line {}",
                    index + 1
                ),
            }
        }
        if let Some(payload) = inline_payload(rest, aliasing) {
            blocks.push(Block::new(tag, "", payload));
            index += 1;
            continue;
        }
        let mut content: Vec<&str> = Vec::new();
        let mut cursor = index + 1;
        let mut closed = false;
        while cursor < lines.len() {
            if is_closer(lines[cursor], aliasing) {
                closed = true;
                break;
            }
            content.push(lines[cursor]);
            cursor += 1;
        }
        snafu::ensure_whatever!(
            closed,
            "block opened by {} at line {} is never closed",
            tag.name(),
            index + 1
        );
        blocks.push(Block::new(tag, rest, &content.join("\n")));
        index = cursor + 1;
    }

    if !unmarked.is_empty() {
        match aliasing {
            Aliasing::ToolMarkers if blocks.is_empty() => {
                blocks.push(Block::new(Tag::Output, "", &unmarked.join("\n")));
            }
            Aliasing::ToolMarkers => {}
            Aliasing::Strict => snafu::whatever!("unmarked content outside every block"),
        }
    }
    Ok(blocks)
}

/// The opener a line carries, with whatever follows it on that line.
fn opener_at(line: &str, aliasing: Aliasing) -> Option<(Tag, &str)> {
    for tag in TAGS.into_iter().filter(Tag::opens) {
        if let Some(rest) = line.strip_prefix(tag.spelling()) {
            return Some((tag, rest.trim()));
        }
    }
    if aliasing == Aliasing::ToolMarkers
        && let Some(rest) = line.strip_prefix("<function_calls>")
    {
        return Some((Tag::Output, rest.trim()));
    }
    None
}

/// A closer at column zero, alone on its line.
fn is_closer(line: &str, aliasing: Aliasing) -> bool {
    let trimmed = line.trim_end();
    if trimmed == Tag::Close.spelling() {
        return true;
    }
    aliasing == Aliasing::ToolMarkers && trimmed == "</function_calls>"
}

/// The payload of an inline block, when the rest closes on its line.
fn inline_payload(rest: &str, aliasing: Aliasing) -> Option<&str> {
    if let Some(payload) = rest.strip_suffix(Tag::Close.spelling()) {
        return Some(payload);
    }
    if aliasing == Aliasing::ToolMarkers {
        return rest.strip_suffix("</function_calls>");
    }
    None
}

/// One diagnostic row, addressed by cell-path rather than by span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub kind: String,
    /// The cell-path into the offending value, where one applies.
    pub source: Option<String>,
    pub message: String,
}

impl Diagnostic {
    pub fn new(kind: &str, source: Option<&str>, message: &str) -> Self {
        Self {
            kind: kind.to_string(),
            source: source.map(str::to_string),
            message: message.to_string(),
        }
    }

    fn to_value(&self) -> nu::Value {
        let span = nu::Span::unknown();
        nu::Value::record(
            nu::record! {
                "kind" => nu::Value::string(self.kind.clone(), span),
                "source" => match &self.source {
                    Some(source) => nu::Value::string(source.clone(), span),
                    None => nu::Value::nothing(span),
                },
                "message" => nu::Value::string(self.message.clone(), span),
            },
            span,
        )
    }
}

/// The repair loop's feedback: collected rows, never a first-error bail.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Envelope {
    pub errors: Vec<Diagnostic>,
    pub warnings: Vec<Diagnostic>,
}

impl Envelope {
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn error(&mut self, kind: &str, source: Option<&str>, message: &str) {
        self.errors.push(Diagnostic::new(kind, source, message));
    }

    pub fn warn(&mut self, kind: &str, source: Option<&str>, message: &str) {
        self.warnings.push(Diagnostic::new(kind, source, message));
    }

    /// The envelope as a nu value, for the environment turn it rides.
    pub fn to_value(&self) -> nu::Value {
        let span = nu::Span::unknown();
        let rows = |set: &[Diagnostic]| {
            nu::Value::list(set.iter().map(Diagnostic::to_value).collect(), span)
        };
        nu::Value::record(
            nu::record! {
                "errors" => rows(&self.errors),
                "warnings" => rows(&self.warnings),
            },
            span,
        )
    }
}
