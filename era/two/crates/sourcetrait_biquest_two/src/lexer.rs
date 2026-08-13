//! The Quill content lexer: dictionary hit or character split, nothing
//! between; designed bucket sequences match ahead of single characters.
use crate::*;

use crate::bucket::BucketTable;
use crate::census::read_admitted;
use crate::ucd::CharClass;
use crate::ucd::CharacterTable;

/// The keyword page: 256 ids, 0x00-0xFF, reserved whole forever.
pub(crate) const KEYWORD_PAGE_SIZE: u32 = 256;
/// The character layer's id offset, directly above the keyword page.
pub(crate) const CHARACTER_OFFSET: u32 = KEYWORD_PAGE_SIZE;
/// The REPEAT operator: hardcoded keyword ids allocate from the
/// page's end downward, user bindings from 0x00 upward.
pub(crate) const KEYWORD_REPEAT: u32 = 0xFF;
/// REPEAT's hardcoded authoring alias.
pub(crate) const REPEAT_ALIAS: &str = "<|repeat|>";

/// What one lexed piece is, before id resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PieceKind {
    /// A maximal run of word-constituent characters (W+).
    Word,
    /// One Unicode-class character; they never merge.
    Unicode,
    /// A designed bucket sequence, by its in-layer index.
    Bucket(usize),
}

/// One lexed piece: a slice of the input plus its kind.
#[derive(Debug)]
pub(crate) struct Piece<'a> {
    pub(crate) text: &'a str,
    pub(crate) kind: PieceKind,
}

/// Split text into W+ word runs, bucket sequences, and
/// single-character Unicode pieces. A bucket match is tried first at
/// every Unicode-class position (all bucket sequences start with
/// Unicode-class characters, so a word run is never broken). An
/// unassigned code point refuses; every assigned character lexes.
pub(crate) fn boundary_pieces<'a>(
    table: &CharacterTable,
    buckets: &BucketTable,
    text: &'a str,
) -> BiquestResult<Vec<Piece<'a>>> {
    let mut pieces = Vec::new();
    let mut word_start: Option<usize> = None;
    let mut offset = 0usize;
    while offset < text.len() {
        let c = text[offset..].chars().next().expect("char at boundary");
        match table.class_of(c) {
            CharClass::Word => {
                if word_start.is_none() {
                    word_start = Some(offset);
                }
                offset += c.len_utf8();
            }
            CharClass::Unicode => {
                if let Some(start) = word_start.take() {
                    pieces.push(Piece { text: &text[start..offset], kind: PieceKind::Word });
                }
                match buckets.match_at(&text[offset..]) {
                    Some((index, length)) => {
                        pieces.push(Piece {
                            text: &text[offset..offset + length],
                            kind: PieceKind::Bucket(index),
                        });
                        offset += length;
                    }
                    None => {
                        pieces.push(Piece {
                            text: &text[offset..offset + c.len_utf8()],
                            kind: PieceKind::Unicode,
                        });
                        offset += c.len_utf8();
                    }
                }
            }
            CharClass::Other => snafu::whatever!(
                "ingestion refusal: unassigned code point U+{:04X} at byte {offset}",
                c as u32
            ),
        }
    }
    if let Some(start) = word_start {
        pieces.push(Piece { text: &text[start..], kind: PieceKind::Word });
    }
    Ok(pieces)
}

/// Which layer resolved a token; the id spells it too, this names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layer {
    /// A hardcoded keyword-page operator the encoder emitted.
    Keyword,
    Character,
    Bucket,
    Dictionary,
}

/// One resolved token.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Token {
    pub(crate) id: u32,
    pub(crate) layer: Layer,
}

/// The layered segmenter: keyword page, characters, buckets, words.
pub(crate) struct Segmenter<'a> {
    table: &'a CharacterTable,
    buckets: &'a BucketTable,
    bucket_offset: u32,
    /// Case-folded admitted word to its dictionary-layer id.
    word_ids: HashMap<String, u32>,
}

impl<'a> Segmenter<'a> {
    /// Buckets take ids above the character layer; admitted words
    /// above the buckets, in list order.
    pub(crate) fn new(
        table: &'a CharacterTable,
        buckets: &'a BucketTable,
        admitted: &[String],
    ) -> Self {
        let bucket_offset = CHARACTER_OFFSET + table.assigned_count() as u32;
        let dictionary_offset = bucket_offset + buckets.count() as u32;
        let word_ids = admitted
            .iter()
            .enumerate()
            .map(|(index, word)| (word.clone(), dictionary_offset + index as u32))
            .collect();
        Self { table, buckets, bucket_offset, word_ids }
    }

    fn character_token(&self, c: char) -> BiquestResult<Token> {
        match self.table.index_of(c as u32) {
            Some(index) => Ok(Token {
                id: CHARACTER_OFFSET + index,
                layer: Layer::Character,
            }),
            None => snafu::whatever!(
                "ingestion refusal: unassigned code point U+{:04X}",
                c as u32
            ),
        }
    }

    /// Collapse runs of an identical non-word token into REPEAT
    /// groups: unit, REPEAT, one count digit. Canonical form, ruled:
    /// a run of three or more collapses (ties prefer REPEAT), groups
    /// carry at most nine, a leftover of one or two stays plain. The
    /// single count digit is what keeps the wire unambiguous against
    /// literal digits that follow a run.
    fn collapse_runs(
        &self,
        raw: Vec<(Token, String)>,
    ) -> BiquestResult<Vec<(Token, String)>> {
        let mut out: Vec<(Token, String)> = Vec::with_capacity(raw.len());
        let mut index = 0usize;
        while index < raw.len() {
            let (token, text) = &raw[index];
            if token.layer == Layer::Dictionary {
                out.push((*token, text.clone()));
                index += 1;
                continue;
            }
            let mut run = 1usize;
            while index + run < raw.len()
                && raw[index + run].0.layer == token.layer
                && raw[index + run].0.id == token.id
            {
                run += 1;
            }
            let mut remaining = run;
            while remaining >= 3 {
                let taken = remaining.min(9);
                out.push((*token, text.clone()));
                out.push((
                    Token { id: KEYWORD_REPEAT, layer: Layer::Keyword },
                    String::from(REPEAT_ALIAS),
                ));
                let digit = char::from(b'0' + taken as u8);
                out.push((self.character_token(digit)?, digit.to_string()));
                remaining -= taken;
            }
            for _ in 0..remaining {
                out.push((*token, text.clone()));
            }
            index += run;
        }
        Ok(out)
    }

    /// The raw piece resolution, before the REPEAT collapse.
    fn raw_tokens(&self, text: &str, keep_text: bool) -> BiquestResult<Vec<(Token, String)>> {
        let mut tokens = Vec::new();
        for piece in boundary_pieces(self.table, self.buckets, text)? {
            match piece.kind {
                PieceKind::Bucket(index) => {
                    let token = Token {
                        id: self.bucket_offset + index as u32,
                        layer: Layer::Bucket,
                    };
                    let kept = if keep_text { piece.text.to_string() } else { String::new() };
                    tokens.push((token, kept));
                    continue;
                }
                PieceKind::Word => {
                    let folded = match self.table.fold_str(piece.text) {
                        Ok(folded) => folded,
                        Err(c) => snafu::whatever!(
                            "ingestion refusal: unassigned code point U+{:04X}",
                            c as u32
                        ),
                    };
                    if let Some(&id) = self.word_ids.get(&folded) {
                        let kept = if keep_text { folded } else { String::new() };
                        tokens.push((Token { id, layer: Layer::Dictionary }, kept));
                        continue;
                    }
                }
                PieceKind::Unicode => {}
            }
            for c in piece.text.chars() {
                let kept = if keep_text { c.to_string() } else { String::new() };
                tokens.push((self.character_token(c)?, kept));
            }
        }
        Ok(tokens)
    }

    /// Segment with each token's text kept, for display surfaces.
    pub(crate) fn segment_pieces(
        &self,
        text: &str,
    ) -> BiquestResult<Vec<(Token, String)>> {
        let raw = self.raw_tokens(text, true)?;
        self.collapse_runs(raw)
    }

    /// Segment content text: a word piece is a case-folded dictionary
    /// hit or its character split; a bucket piece is its designed id;
    /// a Unicode piece is its character; identical-token runs collapse
    /// into REPEAT groups.
    pub(crate) fn segment(&self, text: &str) -> BiquestResult<Vec<Token>> {
        let raw = self.raw_tokens(text, false)?;
        Ok(self.collapse_runs(raw)?.into_iter().map(|(token, _)| token).collect())
    }
}

/// One stretch of the test string: a keyword-page marker or content.
enum TestPart<'a> {
    /// A `<|XX|>` spelling: the keyword id and the spelling itself.
    Marker(u8, &'a str),
    Text(&'a str),
}

/// Split a test string at keyword-page spellings (`<|00|>`-`<|FF|>`).
///
/// Test-surface rendering only: corpus ingestion never reads a marker
/// out of content - markers enter through deliberate rendering, and
/// this is that path for the CLI.
fn split_markers(text: &str) -> Vec<TestPart<'_>> {
    let bytes = text.as_bytes();
    let mut parts = Vec::new();
    let mut rest_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        // The hardcoded aliases spell like markers and mean their id.
        if text[i..].starts_with(REPEAT_ALIAS) {
            if rest_start < i {
                parts.push(TestPart::Text(&text[rest_start..i]));
            }
            parts.push(TestPart::Marker(
                KEYWORD_REPEAT as u8,
                &text[i..i + REPEAT_ALIAS.len()],
            ));
            i += REPEAT_ALIAS.len();
            rest_start = i;
            continue;
        }
        let is_marker = i + 6 <= bytes.len()
            && bytes[i] == b'<'
            && bytes[i + 1] == b'|'
            && bytes[i + 2].is_ascii_hexdigit()
            && bytes[i + 3].is_ascii_hexdigit()
            && bytes[i + 4] == b'|'
            && bytes[i + 5] == b'>';
        if is_marker {
            if rest_start < i {
                parts.push(TestPart::Text(&text[rest_start..i]));
            }
            let hex = &text[i + 2..i + 4];
            let id = u8::from_str_radix(hex, 16).expect("two hex digits");
            parts.push(TestPart::Marker(id, &text[i..i + 6]));
            i += 6;
            rest_start = i;
        } else {
            i += 1;
        }
    }
    if rest_start < text.len() {
        parts.push(TestPart::Text(&text[rest_start..]));
    }
    parts
}

/// `biquest tokenize`: a string's tokens as a nu table on stdout.
///
/// Columns: token (the id), unicode (`U+XXXX` for a character-layer
/// token, null for a keyword or dictionary word), value (the token's
/// text). `<|XX|>` spellings render as keyword-page tokens. Without
/// --admitted the embedded full English set is the dictionary; its
/// ids are the test surface's, not a trained vocabulary's.
pub(crate) fn tokenize_text(args: &TokenizeArgs) -> BiquestResult<()> {
    let table = CharacterTable::embedded()?;
    let buckets = BucketTable::new();
    let admitted = match &args.admitted {
        Some(path) => read_admitted(path)?,
        None => crate::dictionary::embedded_words(),
    };
    let segmenter = Segmenter::new(&table, &buckets, &admitted);
    let mut rows: Vec<harness::nu::Value> = Vec::new();
    for part in split_markers(&args.text) {
        match part {
            TestPart::Marker(id, spelling) => {
                rows.push(harness::nu::Value::record(
                    harness::nu::record! {
                        "token" => v_int(id as i64),
                        "unicode" => harness::nu::Value::nothing(span()),
                        "value" => v_str(spelling),
                    },
                    span(),
                ));
            }
            TestPart::Text(text) => {
                for (token, text) in segmenter.segment_pieces(text)? {
                    let unicode = match token.layer {
                        Layer::Character => {
                            let c = text.chars().next().unwrap_or('\u{FFFD}');
                            v_str(&format!("U+{:04X}", c as u32))
                        }
                        Layer::Keyword | Layer::Bucket | Layer::Dictionary => {
                            harness::nu::Value::nothing(span())
                        }
                    };
                    rows.push(harness::nu::Value::record(
                        harness::nu::record! {
                            "token" => v_int(token.id as i64),
                            "unicode" => unicode,
                            "value" => v_str(&text),
                        },
                        span(),
                    ));
                }
            }
        }
    }
    let rendered = harness::nu::to_nuon_pretty(&harness::nu::Value::list(rows, span()))?;
    println!("{rendered}");
    Ok(())
}
