//! The bucket layer: designed sequence tokens, each with a stated reason.
//!
//! Buckets hold only semantic base units; run magnitude rides the
//! REPEAT operator (lexer.rs), never depth enumerations.
use crate::*;

/// Indentation units: the two conventions. Deeper indents ride
/// REPEAT; tab indents ride the tab character row plus REPEAT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndentBucket {
    /// A two-space indent step.
    TwoSpaces,
    /// A four-space indent step.
    FourSpaces,
}

/// Markdown sequences not already covered by the layers below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MarkdownBucket {
    /// The bold marker; also claims the comment-star pair, one id.
    Bold,
    /// An ATX heading at depth 2..=6, CommonMark's ceiling; depth 1
    /// is the hash character row. Levels are semantic, not
    /// magnitude, so they stay enumerated.
    HeadingDepth(u8),
}

/// ASCII line-drawing units over `-`, `=`, `_`, and `~`; longer
/// lines ride REPEAT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AsciiBucket {
    /// The two-character pair (`--`, `==`, `__`, `~~`).
    LinePair(char),
    /// The four-character quad (`----`, `====`, `____`, `~~~~`).
    LineQuad(char),
}

/// Comment openers and closers not covered by the buckets above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommentBucket {
    /// `//`
    LineComment,
    /// `/*`
    BlockOpen,
    /// `/**`
    DocBlockOpen,
    /// `*/`
    BlockClose,
}

/// One bucket entry; the category is the entry's stated reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bucket {
    Indent(IndentBucket),
    Markdown(MarkdownBucket),
    Ascii(AsciiBucket),
    Comment(CommentBucket),
}

/// The line-drawing characters, in canonical order.
const LINE_CHARS: [char; 4] = ['-', '=', '_', '~'];

impl Bucket {
    /// The character sequence this entry tokenizes.
    pub(crate) fn sequence(&self) -> String {
        match self {
            Bucket::Indent(IndentBucket::TwoSpaces) => String::from("  "),
            Bucket::Indent(IndentBucket::FourSpaces) => String::from("    "),
            Bucket::Markdown(MarkdownBucket::Bold) => String::from("**"),
            Bucket::Markdown(MarkdownBucket::HeadingDepth(depth)) => {
                "#".repeat(*depth as usize)
            }
            Bucket::Ascii(AsciiBucket::LinePair(c)) => c.to_string().repeat(2),
            Bucket::Ascii(AsciiBucket::LineQuad(c)) => c.to_string().repeat(4),
            Bucket::Comment(CommentBucket::LineComment) => String::from("//"),
            Bucket::Comment(CommentBucket::BlockOpen) => String::from("/*"),
            Bucket::Comment(CommentBucket::DocBlockOpen) => String::from("/**"),
            Bucket::Comment(CommentBucket::BlockClose) => String::from("*/"),
        }
    }

    /// The category label, for display surfaces.
    pub(crate) fn category(&self) -> &'static str {
        match self {
            Bucket::Indent(_) => "indent",
            Bucket::Markdown(_) => "markdown",
            Bucket::Ascii(_) => "ascii",
            Bucket::Comment(_) => "comment",
        }
    }

    /// The canonical enumeration: entry position IS the in-layer id,
    /// append-only forever.
    pub(crate) fn all() -> Vec<Bucket> {
        let mut entries = Vec::new();
        entries.push(Bucket::Indent(IndentBucket::TwoSpaces));
        entries.push(Bucket::Indent(IndentBucket::FourSpaces));
        entries.push(Bucket::Markdown(MarkdownBucket::Bold));
        for depth in 2..=6 {
            entries.push(Bucket::Markdown(MarkdownBucket::HeadingDepth(depth)));
        }
        for c in LINE_CHARS {
            entries.push(Bucket::Ascii(AsciiBucket::LinePair(c)));
            entries.push(Bucket::Ascii(AsciiBucket::LineQuad(c)));
        }
        entries.push(Bucket::Comment(CommentBucket::LineComment));
        entries.push(Bucket::Comment(CommentBucket::BlockOpen));
        entries.push(Bucket::Comment(CommentBucket::DocBlockOpen));
        entries.push(Bucket::Comment(CommentBucket::BlockClose));
        entries
    }
}

/// The bucket layer's match table: canonical entries plus a
/// first-character index, longest sequence first.
pub(crate) struct BucketTable {
    entries: Vec<(String, Bucket)>,
    by_first: HashMap<char, Vec<usize>>,
}

impl BucketTable {
    pub(crate) fn new() -> Self {
        let entries: Vec<(String, Bucket)> = Bucket::all()
            .into_iter()
            .map(|bucket| (bucket.sequence(), bucket))
            .collect();
        debug_assert!(
            {
                let unique: HashSet<&String> =
                    entries.iter().map(|(sequence, _)| sequence).collect();
                unique.len() == entries.len()
            },
            "bucket sequences must be unique"
        );
        let mut by_first: HashMap<char, Vec<usize>> = HashMap::new();
        for (index, (sequence, _)) in entries.iter().enumerate() {
            let first = sequence.chars().next().expect("non-empty sequence");
            by_first.entry(first).or_default().push(index);
        }
        for indices in by_first.values_mut() {
            indices.sort_by_key(|&index| std::cmp::Reverse(entries[index].0.len()));
        }
        Self { entries, by_first }
    }

    pub(crate) fn count(&self) -> usize {
        self.entries.len()
    }

    /// The entry at an in-layer index.
    pub(crate) fn entry(&self, index: usize) -> (&str, Bucket) {
        let (sequence, bucket) = &self.entries[index];
        (sequence, *bucket)
    }

    /// The longest entry matching at the start of `rest`, as its
    /// in-layer index and byte length.
    pub(crate) fn match_at(&self, rest: &str) -> Option<(usize, usize)> {
        let first = rest.chars().next()?;
        for &index in self.by_first.get(&first)? {
            let sequence = &self.entries[index].0;
            if rest.starts_with(sequence.as_str()) {
                return Some((index, sequence.len()));
            }
        }
        None
    }
}
