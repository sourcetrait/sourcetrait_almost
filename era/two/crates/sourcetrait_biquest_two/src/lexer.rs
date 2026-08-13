//! The Quill content lexer: dictionary hit or character split, nothing
//! between.
use crate::*;

use crate::census::read_admitted;
use crate::ucd::CharClass;
use crate::ucd::CharacterTable;

/// The keyword page: 256 ids, 0x00-0xFF, reserved whole forever.
pub(crate) const KEYWORD_PAGE_SIZE: u32 = 256;
/// The character layer's id offset, directly above the keyword page.
pub(crate) const CHARACTER_OFFSET: u32 = KEYWORD_PAGE_SIZE;

/// What one lexed piece is, before id resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PieceKind {
    /// A maximal run of word-constituent characters (W+).
    Word,
    /// One Unicode-class character; they never merge.
    Unicode,
}

/// One lexed piece: a slice of the input plus its kind.
#[derive(Debug)]
pub(crate) struct Piece<'a> {
    pub(crate) text: &'a str,
    pub(crate) kind: PieceKind,
}

/// Split text into W+ word runs and single-character Unicode pieces.
/// An unassigned code point refuses; every assigned character lexes.
pub(crate) fn boundary_pieces<'a>(
    table: &CharacterTable,
    text: &'a str,
) -> BiquestResult<Vec<Piece<'a>>> {
    let mut pieces = Vec::new();
    let mut word_start: Option<usize> = None;
    for (offset, c) in text.char_indices() {
        match table.class_of(c) {
            CharClass::Word => {
                if word_start.is_none() {
                    word_start = Some(offset);
                }
            }
            CharClass::Unicode => {
                if let Some(start) = word_start.take() {
                    pieces.push(Piece { text: &text[start..offset], kind: PieceKind::Word });
                }
                pieces.push(Piece {
                    text: &text[offset..offset + c.len_utf8()],
                    kind: PieceKind::Unicode,
                });
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
    Character,
    Dictionary,
}

/// One resolved token.
pub(crate) struct Token {
    pub(crate) id: u32,
    pub(crate) layer: Layer,
}

/// The layered segmenter: keyword page below, characters, then words.
pub(crate) struct Segmenter<'a> {
    table: &'a CharacterTable,
    /// Case-folded admitted word to its dictionary-layer id.
    word_ids: HashMap<String, u32>,
}

impl<'a> Segmenter<'a> {
    /// Admitted words take ids above the character layer, in list order.
    pub(crate) fn new(table: &'a CharacterTable, admitted: &[String]) -> Self {
        let dictionary_offset = CHARACTER_OFFSET + table.assigned_count() as u32;
        let word_ids = admitted
            .iter()
            .enumerate()
            .map(|(index, word)| (word.clone(), dictionary_offset + index as u32))
            .collect();
        Self { table, word_ids }
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

    /// Segment with each token's text kept, for display surfaces.
    pub(crate) fn segment_pieces(
        &self,
        text: &str,
    ) -> BiquestResult<Vec<(Token, String)>> {
        let mut tokens = Vec::new();
        for piece in boundary_pieces(self.table, text)? {
            if piece.kind == PieceKind::Word {
                let folded = match self.table.fold_str(piece.text) {
                    Ok(folded) => folded,
                    Err(c) => snafu::whatever!(
                        "ingestion refusal: unassigned code point U+{:04X}",
                        c as u32
                    ),
                };
                if let Some(&id) = self.word_ids.get(&folded) {
                    tokens.push((Token { id, layer: Layer::Dictionary }, folded));
                    continue;
                }
            }
            for c in piece.text.chars() {
                tokens.push((self.character_token(c)?, c.to_string()));
            }
        }
        Ok(tokens)
    }

    /// Segment content text: a word piece is a case-folded dictionary
    /// hit or its character split; a Unicode piece is its character.
    pub(crate) fn segment(&self, text: &str) -> BiquestResult<Vec<Token>> {
        let mut tokens = Vec::new();
        for piece in boundary_pieces(self.table, text)? {
            if piece.kind == PieceKind::Word {
                let folded = match self.table.fold_str(piece.text) {
                    Ok(folded) => folded,
                    Err(c) => snafu::whatever!(
                        "ingestion refusal: unassigned code point U+{:04X}",
                        c as u32
                    ),
                };
                if let Some(&id) = self.word_ids.get(&folded) {
                    tokens.push(Token { id, layer: Layer::Dictionary });
                    continue;
                }
            }
            for c in piece.text.chars() {
                tokens.push(self.character_token(c)?);
            }
        }
        Ok(tokens)
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
    while i + 6 <= bytes.len() {
        let is_marker = bytes[i] == b'<'
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
    let admitted = match &args.admitted {
        Some(path) => read_admitted(path)?,
        None => crate::dictionary::embedded_words(),
    };
    let segmenter = Segmenter::new(&table, &admitted);
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
                        Layer::Dictionary => harness::nu::Value::nothing(span()),
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
