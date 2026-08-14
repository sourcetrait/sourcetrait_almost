//! The Quill content lexer: dictionary hit or character split, nothing
//! between; repetition rides banded operators and keyboard rows.
use crate::*;

use crate::bucket::BucketTable;
use crate::census::read_admitted;
use crate::ucd::CharClass;
use crate::ucd::CharacterTable;

/// The keyword page: 256 ids, 0x00-0xFF, reserved whole forever.
pub(crate) const KEYWORD_PAGE_SIZE: u32 = 256;
/// The character layer's id offset, directly above the keyword page.
pub(crate) const CHARACTER_OFFSET: u32 = KEYWORD_PAGE_SIZE;

/// The hardcoded operator block allocates from the page's end
/// downward, user bindings from 0x00 upward. begin/end bracket a
/// multi-digit count for long runs; REPEAT carries a one-digit
/// count; REPETITION is the AbstractConceptMarker backing the
/// repetition AbstractConcept - illegal in wire, never emitted,
/// never parsed (TheUser: the other three may speak back).
pub(crate) const KEYWORD_BEGIN_REPEAT: u32 = 0xFC;
pub(crate) const KEYWORD_END_REPEAT: u32 = 0xFD;
pub(crate) const KEYWORD_REPEAT: u32 = 0xFE;
pub(crate) const KEYWORD_REPETITION: u32 = 0xFF;

pub(crate) const BEGIN_REPEAT_ALIAS: &str = "<|begin_repeat|>";
pub(crate) const END_REPEAT_ALIAS: &str = "<|end_repeat|>";
pub(crate) const REPEAT_ALIAS: &str = "<|repeat|>";
pub(crate) const REPETITION_ALIAS: &str = "<|repetition|>";

/// The hardcoded alias table, id beside spelling.
pub(crate) const HARDCODED_ALIASES: [(u32, &str); 4] = [
    (KEYWORD_BEGIN_REPEAT, BEGIN_REPEAT_ALIAS),
    (KEYWORD_END_REPEAT, END_REPEAT_ALIAS),
    (KEYWORD_REPEAT, REPEAT_ALIAS),
    (KEYWORD_REPETITION, REPETITION_ALIAS),
];

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
/// One rule for every character, no designed sequences (TheUser:
/// consistency over special cases - irregularity bites in training).
/// An unassigned code point refuses; every assigned character lexes.
pub(crate) fn boundary_pieces<'a>(
    table: &CharacterTable,
    text: &'a str,
) -> BiquestResult<Vec<Piece<'a>>> {
    let mut pieces = Vec::new();
    let mut word_start: Option<usize> = None;
    for (offset, c) in text.char_indices() {
        match table.class_of(c) {
            CharClass::Word | CharClass::Digit => {
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
    /// A hardcoded keyword-page operator the encoder emitted.
    Keyword,
    Character,
    /// A keyboard-symbol double or triple row.
    Bucket,
    Dictionary,
}

/// One resolved token.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Token {
    pub(crate) id: u32,
    pub(crate) layer: Layer,
}

/// The layered segmenter: keyword page, characters, keyboard rows,
/// then words.
pub(crate) struct Segmenter<'a> {
    table: &'a CharacterTable,
    buckets: &'a BucketTable,
    bucket_offset: u32,
    /// Case-folded admitted word to its dictionary-layer id.
    word_ids: HashMap<String, u32>,
    /// The ten Digit-class character ids, exempt from run encoding.
    digit_ids: HashSet<u32>,
}

impl<'a> Segmenter<'a> {
    /// Keyboard rows take ids above the character layer; admitted
    /// words above the rows, in list order.
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
        let digit_ids = ('0'..='9')
            .filter_map(|digit| table.index_of(digit as u32))
            .map(|index| CHARACTER_OFFSET + index)
            .collect();
        Self { table, buckets, bucket_offset, word_ids, digit_ids }
    }

    /// The character behind a character-layer token id.
    fn char_of(&self, id: u32) -> Option<char> {
        let index = id.checked_sub(CHARACTER_OFFSET)? as usize;
        let row = self.table.rows.get(index)?;
        char::from_u32(row.code_point)
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

    /// Encode runs of an identical non-word, non-digit token through
    /// the magnitude bands (canonical, deterministic):
    ///
    /// - 1: the token itself.
    /// - 2 or 3, keyboard symbol: the double or triple row.
    /// - 2, otherwise: plain (a group would cost three).
    /// - 3..=9 (from 3 where no row exists): unit, REPEAT, one count
    ///   digit - the one-digit bound keeps literal digits after a
    ///   run unambiguous.
    /// - 10 and up: unit, BEGIN_REPEAT, the count's digits,
    ///   END_REPEAT - bracketed, so any magnitude costs a bounded
    ///   handful of tokens.
    ///
    /// The rows never compose with the operators or each other
    /// (TheUser: repeat runs early and greedy, without the doubles
    /// and triples). Digits never encode as runs: a number is
    /// place-value content.
    fn encode_runs(
        &self,
        raw: Vec<(Token, String)>,
        keep_text: bool,
    ) -> BiquestResult<Vec<(Token, String)>> {
        let mut out: Vec<(Token, String)> = Vec::with_capacity(raw.len());
        let mut index = 0usize;
        while index < raw.len() {
            let (token, text) = &raw[index];
            let exempt = token.layer != Layer::Character
                || self.digit_ids.contains(&token.id);
            if exempt {
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
            let symbol = self.char_of(token.id);
            let row_index = symbol.and_then(|c| {
                if run == 2 || run == 3 {
                    self.buckets.index_for(c, run as u8)
                } else {
                    None
                }
            });
            let kept = |value: String| if keep_text { value } else { String::new() };
            match (run, row_index) {
                (1, _) => out.push((*token, text.clone())),
                (2..=3, Some(row)) => out.push((
                    Token {
                        id: self.bucket_offset + row as u32,
                        layer: Layer::Bucket,
                    },
                    kept(self.buckets.entry(row).sequence()),
                )),
                (2, None) => {
                    out.push((*token, text.clone()));
                    out.push((*token, text.clone()));
                }
                (3..=9, None) => {
                    out.push((*token, text.clone()));
                    out.push((
                        Token { id: KEYWORD_REPEAT, layer: Layer::Keyword },
                        kept(String::from(REPEAT_ALIAS)),
                    ));
                    let digit = char::from(b'0' + run as u8);
                    out.push((self.character_token(digit)?, kept(digit.to_string())));
                }
                _ => {
                    out.push((*token, text.clone()));
                    out.push((
                        Token { id: KEYWORD_BEGIN_REPEAT, layer: Layer::Keyword },
                        kept(String::from(BEGIN_REPEAT_ALIAS)),
                    ));
                    for digit in run.to_string().chars() {
                        out.push((self.character_token(digit)?, kept(digit.to_string())));
                    }
                    out.push((
                        Token { id: KEYWORD_END_REPEAT, layer: Layer::Keyword },
                        kept(String::from(END_REPEAT_ALIAS)),
                    ));
                }
            }
            index += run;
        }
        Ok(out)
    }

    /// The raw piece resolution, before the REPEAT collapse.
    fn raw_tokens(&self, text: &str, keep_text: bool) -> BiquestResult<Vec<(Token, String)>> {
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
                    let kept = if keep_text { folded } else { String::new() };
                    tokens.push((Token { id, layer: Layer::Dictionary }, kept));
                    continue;
                }
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
        self.encode_runs(raw, true)
    }

    /// Segment content text: a word piece is a case-folded dictionary
    /// hit or its character split; a Unicode piece is its character;
    /// identical-token runs encode through the magnitude bands.
    pub(crate) fn segment(&self, text: &str) -> BiquestResult<Vec<Token>> {
        let raw = self.raw_tokens(text, false)?;
        Ok(self
            .encode_runs(raw, false)?
            .into_iter()
            .map(|(token, _)| token)
            .collect())
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
    'scan: while i < bytes.len() {
        // The hardcoded aliases spell like markers and mean their id.
        for (id, alias) in HARDCODED_ALIASES {
            if text[i..].starts_with(alias) {
                if rest_start < i {
                    parts.push(TestPart::Text(&text[rest_start..i]));
                }
                parts.push(TestPart::Marker(id as u8, &text[i..i + alias.len()]));
                i += alias.len();
                rest_start = i;
                continue 'scan;
            }
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
