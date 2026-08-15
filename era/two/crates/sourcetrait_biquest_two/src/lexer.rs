//! The Quill content lexer: connected candidates resolved by the
//! dictionary - full, then core, then parts, then the longest-match
//! decomposition - with repetition riding banded operators and
//! keyboard rows.
use crate::*;

use crate::bucket::BucketTable;
use crate::dictionary::read_words_ordered;
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

/// The case family (TheUser's design), uniformly POSTFIX: the match
/// first, then the tokenizer token. This = this-row CAPITALIZED;
/// THIS = this-row UPPERCASED; QuILL = quill-row CASED then exactly
/// the row's length in character tokens carrying the surface,
/// length-bounded by the row so no terminator exists. CASE is the
/// AbstractConceptMarker backing the case AbstractConcept - illegal
/// in wire - and the three operators associate to it in the store.
pub(crate) const KEYWORD_CASED: u32 = 0xF8;
pub(crate) const KEYWORD_CASE: u32 = 0xF9;
pub(crate) const KEYWORD_CAPITALIZED: u32 = 0xFA;
pub(crate) const KEYWORD_UPPERCASED: u32 = 0xFB;

/// The forced per-character span operators (TheUser: a tokenizer
/// thing, not portable language syntax, so hardcoded rather than
/// table-bound). UNICODE opens, UNICODED closes; both ride the wire
/// as encapsulation markers bracketing an exact per-character
/// surface, so the model sees the tokenization change in-band.
pub(crate) const KEYWORD_UNICODE: u32 = 0xF6;
pub(crate) const KEYWORD_UNICODED: u32 = 0xF7;

/// The lowest hardcoded id: user bindings stop below it.
pub(crate) const HARDCODED_BLOCK_FLOOR: u32 = KEYWORD_UNICODE;

pub(crate) const BEGIN_REPEAT_ALIAS: &str = "<|begin_repeat|>";
pub(crate) const END_REPEAT_ALIAS: &str = "<|end_repeat|>";
pub(crate) const REPEAT_ALIAS: &str = "<|repeat|>";
pub(crate) const REPETITION_ALIAS: &str = "<|repetition|>";
pub(crate) const CASE_ALIAS: &str = "<|case|>";
pub(crate) const CASED_ALIAS: &str = "<|cased|>";
pub(crate) const CAPITALIZED_ALIAS: &str = "<|capitalized|>";
pub(crate) const UPPERCASED_ALIAS: &str = "<|uppercased|>";
pub(crate) const UNICODE_ALIAS: &str = "<|unicode|>";
pub(crate) const UNICODED_ALIAS: &str = "<|unicoded|>";

/// The hardcoded alias table, id beside spelling.
pub(crate) const HARDCODED_ALIASES: [(u32, &str); 10] = [
    (KEYWORD_BEGIN_REPEAT, BEGIN_REPEAT_ALIAS),
    (KEYWORD_END_REPEAT, END_REPEAT_ALIAS),
    (KEYWORD_REPEAT, REPEAT_ALIAS),
    (KEYWORD_REPETITION, REPETITION_ALIAS),
    (KEYWORD_CASE, CASE_ALIAS),
    (KEYWORD_CASED, CASED_ALIAS),
    (KEYWORD_CAPITALIZED, CAPITALIZED_ALIAS),
    (KEYWORD_UPPERCASED, UPPERCASED_ALIAS),
    (KEYWORD_UNICODE, UNICODE_ALIAS),
    (KEYWORD_UNICODED, UNICODED_ALIAS),
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

/// A connector's normalized form when word characters flank it: the
/// hyphen, and the apostrophe with U+2019 folding to ASCII.
fn connector_normalized(c: char) -> Option<char> {
    match c {
        '-' => Some('-'),
        '\'' | '\u{2019}' => Some('\''),
        _ => None,
    }
}

/// An apostrophe surface, edge-extension eligible (U+2019 included).
fn is_apostrophe(c: char) -> bool {
    matches!(c, '\'' | '\u{2019}')
}

/// One core part of a candidate: a word run (folded lookup form
/// beside its surface) or an internal connector (normalized beside
/// its surface).
pub(crate) enum CorePart<'a> {
    Word { folded: String, surface: &'a str },
    Connector { normalized: char, surface: char },
}

/// One connected dictionary-lookup candidate (TheUser's design):
/// word parts joined by internal connectors, with at most one edge
/// apostrophe per side. `full` and `core` carry the folded,
/// apostrophe-normalized lookup forms; surfaces survive for the
/// miss fallbacks.
pub(crate) struct ConnectedCandidate<'a> {
    pub(crate) full: String,
    pub(crate) core: String,
    /// The case-preserving twins of full and core: surfaces kept,
    /// connectors and edge apostrophes normalized - what the case
    /// classification compares against the folded identity.
    pub(crate) cased_full: String,
    pub(crate) cased_core: String,
    pub(crate) leading: Option<char>,
    pub(crate) trailing: Option<char>,
    pub(crate) parts: Vec<CorePart<'a>>,
}

/// One item of the candidate walk: a candidate, or one plain
/// Unicode-class character.
pub(crate) enum CandidateItem<'a> {
    Candidate(ConnectedCandidate<'a>),
    Plain(&'a str),
}

fn fold_word_piece(table: &CharacterTable, piece: &Piece<'_>) -> BiquestResult<String> {
    match table.fold_str(piece.text) {
        Ok(folded) => Ok(folded),
        Err(c) => snafu::whatever!(
            "ingestion refusal: unassigned code point U+{:04X}",
            c as u32
        ),
    }
}

/// Assemble one candidate from a piece slice; returns it and how
/// many pieces it consumed.
fn assemble_candidate<'a>(
    table: &CharacterTable,
    pieces: &[Piece<'a>],
) -> BiquestResult<(ConnectedCandidate<'a>, usize)> {
    let mut cursor = 0usize;
    let mut leading = None;
    if pieces[0].kind == PieceKind::Unicode {
        leading = pieces[0].text.chars().next();
        cursor = 1;
    }
    let mut parts: Vec<CorePart<'a>> = vec![CorePart::Word {
        folded: fold_word_piece(table, &pieces[cursor])?,
        surface: pieces[cursor].text,
    }];
    cursor += 1;
    loop {
        let connector = pieces.get(cursor).and_then(|piece| {
            if piece.kind != PieceKind::Unicode {
                return None;
            }
            let c = piece.text.chars().next()?;
            connector_normalized(c).map(|normalized| (c, normalized))
        });
        let Some((surface, normalized)) = connector else { break };
        let Some(next) = pieces.get(cursor + 1) else { break };
        if next.kind != PieceKind::Word {
            break;
        }
        parts.push(CorePart::Connector { normalized, surface });
        parts.push(CorePart::Word {
            folded: fold_word_piece(table, next)?,
            surface: next.text,
        });
        cursor += 2;
    }
    let mut trailing = None;
    if let Some(piece) = pieces.get(cursor)
        && piece.kind == PieceKind::Unicode
        && piece.text.chars().next().is_some_and(is_apostrophe)
    {
        // The internal loop consumed every word-flanked apostrophe,
        // so this one has no word after it: a trailing edge.
        trailing = piece.text.chars().next();
        cursor += 1;
    }
    let core: String = parts
        .iter()
        .map(|part| match part {
            CorePart::Word { folded, .. } => folded.clone(),
            CorePart::Connector { normalized, .. } => normalized.to_string(),
        })
        .collect();
    let cased_core: String = parts
        .iter()
        .map(|part| match part {
            CorePart::Word { surface, .. } => (*surface).to_string(),
            CorePart::Connector { normalized, .. } => normalized.to_string(),
        })
        .collect();
    let mut full = String::new();
    let mut cased_full = String::new();
    if leading.is_some() {
        full.push('\'');
        cased_full.push('\'');
    }
    full.push_str(&core);
    cased_full.push_str(&cased_core);
    if trailing.is_some() {
        full.push('\'');
        cased_full.push('\'');
    }
    Ok((
        ConnectedCandidate { full, core, cased_full, cased_core, leading, trailing, parts },
        cursor,
    ))
}

/// The candidate walk over boundary pieces: word runs extend across
/// internal connectors and take a single edge apostrophe; everything
/// else passes as plain characters. The dictionary decides what
/// stays whole - this walk only proposes.
pub(crate) fn candidate_items<'a>(
    table: &CharacterTable,
    text: &'a str,
) -> BiquestResult<Vec<CandidateItem<'a>>> {
    let pieces = boundary_pieces(table, text)?;
    let mut items = Vec::new();
    let mut index = 0usize;
    while index < pieces.len() {
        let piece = &pieces[index];
        let leading_here = piece.kind == PieceKind::Unicode
            && piece.text.chars().next().is_some_and(is_apostrophe)
            && pieces
                .get(index + 1)
                .is_some_and(|next| next.kind == PieceKind::Word);
        if piece.kind == PieceKind::Word || leading_here {
            let (candidate, consumed) = assemble_candidate(table, &pieces[index..])?;
            items.push(CandidateItem::Candidate(candidate));
            index += consumed;
        } else {
            items.push(CandidateItem::Plain(piece.text));
            index += 1;
        }
    }
    Ok(items)
}

/// An entry admissible as ONE dictionary row: its text is exactly
/// one connected candidate of more than one code point. The only
/// removal beyond reachability is the single-code-point rule
/// (TheUser); the returned form is the folded, normalized full text.
pub(crate) fn whole_candidate_folded(
    table: &CharacterTable,
    candidate: &str,
) -> Option<String> {
    let items = candidate_items(table, candidate).ok()?;
    let mut iterator = items.into_iter();
    let (Some(CandidateItem::Candidate(connected)), None) =
        (iterator.next(), iterator.next())
    else {
        return None;
    };
    if connected.full.chars().count() == 1 {
        return None;
    }
    Some(connected.full)
}

/// One piece of a decomposition cover: a dictionary span over the
/// folded text, or a single uncovered character.
#[derive(Debug, Clone, Copy)]
struct CoverPiece {
    start: usize,
    end: usize,
    word: bool,
}

/// One camel-cover piece: a case-classified dictionary span.
#[derive(Debug, Clone, Copy)]
struct CamelPiece {
    start: usize,
    end: usize,
    id: u32,
    operator: Option<u32>,
}

/// One piece of a CASED overlay cover: a case-classified dictionary
/// span (its id and operator), or one surface character.
#[derive(Debug, Clone, Copy)]
struct OverlayPiece {
    start: usize,
    end: usize,
    /// The span's dictionary id and case operator; None is a
    /// character piece.
    word: Option<(u32, Option<u32>)>,
}

impl OverlayPiece {
    /// The wire tokens this piece costs: a bare word 1, an operated
    /// word 2, a character 1.
    fn cost(&self) -> usize {
        match self.word {
            Some((_, Some(_))) => 2,
            _ => 1,
        }
    }
}

/// Whether overlay cover `a` beats `b`: fewer wire tokens, then the
/// longer piece toward the end - the decomposition criteria priced
/// in tokens, because an operated span costs two.
fn overlay_beats(a: &[OverlayPiece], b: &[OverlayPiece]) -> bool {
    let cost_a: usize = a.iter().map(OverlayPiece::cost).sum();
    let cost_b: usize = b.iter().map(OverlayPiece::cost).sum();
    if cost_a != cost_b {
        return cost_a < cost_b;
    }
    for (piece_a, piece_b) in a.iter().rev().zip(b.iter().rev()) {
        let length_a = piece_a.end - piece_a.start;
        let length_b = piece_b.end - piece_b.start;
        if length_a != length_b {
            return length_a > length_b;
        }
    }
    false
}

/// Whether cover `a` beats cover `b` (TheUser's criteria): fewer
/// tokens first, then the longer match toward the END - English is
/// prefix-heavy, so the tail carries the root.
fn cover_beats(a: &[CoverPiece], b: &[CoverPiece]) -> bool {
    if a.len() != b.len() {
        return a.len() < b.len();
    }
    for (piece_a, piece_b) in a.iter().rev().zip(b.iter().rev()) {
        let length_a = piece_a.end - piece_a.start;
        let length_b = piece_b.end - piece_b.start;
        if length_a != length_b {
            return length_a > length_b;
        }
    }
    false
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
    /// Per-row character counts, in id order - the CASED overlay's
    /// length bound.
    row_lengths: Vec<usize>,
    /// The ten Digit-class character ids, exempt from run encoding.
    digit_ids: HashSet<u32>,
}

impl<'a> Segmenter<'a> {
    /// Keyboard rows take ids above the character layer; dictionary
    /// words above the rows, in file order (the whole dictionary is
    /// the vocabulary).
    pub(crate) fn new(
        table: &'a CharacterTable,
        buckets: &'a BucketTable,
        words: &[String],
    ) -> Self {
        let bucket_offset = CHARACTER_OFFSET + table.assigned_count() as u32;
        let dictionary_offset = bucket_offset + buckets.count() as u32;
        let word_ids = words
            .iter()
            .enumerate()
            .map(|(index, word)| (word.clone(), dictionary_offset + index as u32))
            .collect();
        let row_lengths = words.iter().map(|word| word.chars().count()).collect();
        let digit_ids = ('0'..='9')
            .filter_map(|digit| table.index_of(digit as u32))
            .map(|index| CHARACTER_OFFSET + index)
            .collect();
        Self { table, buckets, bucket_offset, word_ids, row_lengths, digit_ids }
    }

    /// A dictionary id's character count, the CASED overlay bound.
    fn row_length(&self, id: u32) -> Option<usize> {
        let dictionary_offset = self.bucket_offset + self.buckets.count() as u32;
        self.row_lengths
            .get(id.checked_sub(dictionary_offset)? as usize)
            .copied()
    }

    /// The character behind a character-layer token id.
    fn char_of(&self, id: u32) -> Option<char> {
        let index = id.checked_sub(CHARACTER_OFFSET)? as usize;
        let row = self.table.rows.get(index)?;
        char::from_u32(row.code_point)
    }

    /// A character's UCD simple uppercase, itself when unmapped.
    fn upper_char(&self, c: char) -> char {
        self.table
            .simple_uppercase
            .get(&(c as u32))
            .and_then(|&target| char::from_u32(target))
            .unwrap_or(c)
    }

    /// The case operator a cased surface needs over its folded row
    /// (TheUser's design): Some(None) bare, Some(op) capitalize or
    /// uppercase, None for any other mix - the candidate character
    /// splits instead, keeping totality for the McDonald class.
    fn case_operator(&self, cased: &str, folded: &str) -> Option<Option<u32>> {
        if cased == folded {
            return Some(None);
        }
        let mut folded_chars = folded.chars();
        let capitalized: String = match folded_chars.next() {
            Some(first) => {
                let mut out = String::with_capacity(folded.len());
                out.push(self.upper_char(first));
                out.push_str(folded_chars.as_str());
                out
            }
            None => String::new(),
        };
        if cased == capitalized {
            return Some(Some(KEYWORD_CAPITALIZED));
        }
        let uppercased: String = folded.chars().map(|c| self.upper_char(c)).collect();
        if cased == uppercased {
            return Some(Some(KEYWORD_UPPERCASED));
        }
        None
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

    /// The CASED overlay's word-combination cover (TheUser's design):
    /// spend dictionary tokens with case operators inside the overlay
    /// wherever that ties or beats the character run - MicroSoft
    /// reads microsoft CASED micro CAPITALIZED soft CAPITALIZED, six
    /// tokens against eleven. A span is admissible when its folded
    /// slice is a row whose surface slice classifies bare,
    /// capitalized, or uppercased; None means no combination ties the
    /// character overlay.
    fn cased_overlay(
        &self,
        folded: &[char],
        surface: &[char],
    ) -> Option<Vec<OverlayPiece>> {
        if folded.len() != surface.len() {
            return None;
        }
        let length = folded.len();
        let mut best: Vec<Option<Vec<OverlayPiece>>> = vec![None; length + 1];
        best[length] = Some(Vec::new());
        for start in (0..length).rev() {
            let mut chosen: Option<Vec<OverlayPiece>> = None;
            let consider = |piece: OverlayPiece,
                            best: &[Option<Vec<OverlayPiece>>],
                            chosen: &mut Option<Vec<OverlayPiece>>| {
                let Some(tail) = &best[piece.end] else { return };
                let mut candidate = Vec::with_capacity(tail.len() + 1);
                candidate.push(piece);
                candidate.extend(tail.iter().copied());
                let better = match chosen {
                    None => true,
                    Some(current) => overlay_beats(&candidate, current),
                };
                if better {
                    *chosen = Some(candidate);
                }
            };
            consider(
                OverlayPiece { start, end: start + 1, word: None },
                &best,
                &mut chosen,
            );
            for end in (start + 2)..=length {
                let text: String = folded[start..end].iter().collect();
                let Some(&id) = self.word_ids.get(&text) else { continue };
                let cased: String = surface[start..end].iter().collect();
                let Some(operator) = self.case_operator(&cased, &text) else {
                    continue;
                };
                consider(
                    OverlayPiece { start, end, word: Some((id, operator)) },
                    &best,
                    &mut chosen,
                );
            }
            best[start] = chosen;
        }
        let cover = best[0].take()?;
        let cost: usize = cover.iter().map(OverlayPiece::cost).sum();
        // The character overlay costs one token per character; a
        // combination must tie or reduce (TheUser), and a tie
        // prefers the words.
        if cost <= length && cover.iter().any(|piece| piece.word.is_some()) {
            Some(cover)
        } else {
            None
        }
    }

    /// Camel-case boundaries in a missed surface mark the intended
    /// word boundaries (TheUser): when every camel segment resolves
    /// as a case-classified row, that combination IS the cover,
    /// preferred over the token-count criteria - AntOne reads
    /// ant + one even where an + tone would tie.
    fn camel_cover(&self, folded: &[char], surface: &[char]) -> Option<Vec<CamelPiece>> {
        if folded.len() != surface.len() {
            return None;
        }
        let length = surface.len();
        let mut boundaries: Vec<usize> = vec![0];
        for index in 1..length {
            let previous_upper = surface[index - 1].is_uppercase();
            // A lower-to-upper transition starts a segment; inside an
            // uppercase run, the last capital before a lowercase does
            // (HTTPServer reads HTTP + Server).
            let boundary = surface[index].is_uppercase()
                && (!previous_upper
                    || surface.get(index + 1).is_some_and(|c| c.is_lowercase()));
            if boundary {
                boundaries.push(index);
            }
        }
        if boundaries.len() < 2 {
            return None;
        }
        boundaries.push(length);
        let mut pieces = Vec::with_capacity(boundaries.len() - 1);
        for pair in boundaries.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            if end - start < 2 {
                return None;
            }
            let text: String = folded[start..end].iter().collect();
            let &id = self.word_ids.get(&text)?;
            let cased: String = surface[start..end].iter().collect();
            let operator = self.case_operator(&cased, &text)?;
            pieces.push(CamelPiece { start, end, id, operator });
        }
        Some(pieces)
    }

    /// The best decomposition cover of a folded dictionary miss
    /// (TheUser's design): fewest tokens, ties preferring the longer
    /// match at the end - nushell covers as nu + shell, never
    /// nus + hell (shorter tail) or nu + sh + ell (more tokens). A
    /// position no row covers falls back to its character. Dynamic
    /// programming over suffixes; the end-anchored tie-break makes
    /// the suffix-optimal choice compose.
    fn decompose_cover(&self, folded_chars: &[char]) -> Vec<CoverPiece> {
        let length = folded_chars.len();
        let mut best: Vec<Option<Vec<CoverPiece>>> = vec![None; length + 1];
        best[length] = Some(Vec::new());
        for start in (0..length).rev() {
            let mut chosen: Option<Vec<CoverPiece>> = None;
            let consider = |piece: CoverPiece,
                                best: &[Option<Vec<CoverPiece>>],
                                chosen: &mut Option<Vec<CoverPiece>>| {
                let Some(tail) = &best[piece.end] else { return };
                let mut candidate = Vec::with_capacity(tail.len() + 1);
                candidate.push(piece);
                candidate.extend(tail.iter().copied());
                let better = match chosen {
                    None => true,
                    Some(current) => cover_beats(&candidate, current),
                };
                if better {
                    *chosen = Some(candidate);
                }
            };
            consider(
                CoverPiece { start, end: start + 1, word: false },
                &best,
                &mut chosen,
            );
            for end in (start + 2)..=length {
                let candidate: String =
                    folded_chars[start..end].iter().collect();
                if self.word_ids.contains_key(&candidate) {
                    consider(
                        CoverPiece { start, end, word: true },
                        &best,
                        &mut chosen,
                    );
                }
            }
            best[start] = chosen;
        }
        best[0].take().unwrap_or_default()
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
            // A CASED overlay is bounded by the row's length in
            // DECODED characters and never run-encodes: a character
            // piece counts one, a word piece its own row length, and
            // a case operator nothing. The length bound IS the
            // decode contract.
            if token.layer == Layer::Keyword && token.id == KEYWORD_CASED {
                let overlay_chars = out
                    .last()
                    .and_then(|(row, _)| self.row_length(row.id))
                    .unwrap_or(0);
                out.push((*token, text.clone()));
                index += 1;
                let mut consumed = 0usize;
                while consumed < overlay_chars && index < raw.len() {
                    let (overlay, overlay_text) = &raw[index];
                    match overlay.layer {
                        Layer::Character => consumed += 1,
                        Layer::Dictionary => {
                            consumed += self.row_length(overlay.id).unwrap_or(1);
                        }
                        _ => {}
                    }
                    out.push((*overlay, overlay_text.clone()));
                    index += 1;
                }
                continue;
            }
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

    /// The raw resolution, before the REPEAT collapse: the candidate
    /// ladder - full, then core with edge apostrophes as characters,
    /// then parts - with cheap dictionary lookups deciding each rung.
    fn raw_tokens(&self, text: &str, keep_text: bool) -> BiquestResult<Vec<(Token, String)>> {
        let mut tokens: Vec<(Token, String)> = Vec::new();
        let push_char = |tokens: &mut Vec<(Token, String)>, c: char| -> BiquestResult<()> {
            let kept = if keep_text { c.to_string() } else { String::new() };
            tokens.push((self.character_token(c)?, kept));
            Ok(())
        };
        // POSTFIX, uniformly: the match first, then the tokenizer
        // token (TheUser's ruling).
        let push_word = |tokens: &mut Vec<(Token, String)>,
                         id: u32,
                         operator: Option<u32>,
                         folded: &str| {
            let kept = if keep_text { folded.to_string() } else { String::new() };
            tokens.push((Token { id, layer: Layer::Dictionary }, kept));
            if let Some(operator) = operator {
                let alias = HARDCODED_ALIASES
                    .iter()
                    .find(|(op, _)| *op == operator)
                    .map(|(_, alias)| *alias)
                    .unwrap_or("");
                let kept = if keep_text { String::from(alias) } else { String::new() };
                tokens.push((Token { id: operator, layer: Layer::Keyword }, kept));
            }
        };
        let push_cased = |tokens: &mut Vec<(Token, String)>,
                          id: u32,
                          folded: &str,
                          cased: &str|
         -> BiquestResult<()> {
            let kept = if keep_text { folded.to_string() } else { String::new() };
            tokens.push((Token { id, layer: Layer::Dictionary }, kept));
            let kept = if keep_text { String::from(CASED_ALIAS) } else { String::new() };
            tokens.push((Token { id: KEYWORD_CASED, layer: Layer::Keyword }, kept));
            // The overlay spends word tokens where a combination ties
            // or beats the character run (TheUser's design).
            let folded_chars: Vec<char> = folded.chars().collect();
            let surface_chars: Vec<char> = cased.chars().collect();
            if let Some(pieces) = self.cased_overlay(&folded_chars, &surface_chars) {
                for piece in pieces {
                    match piece.word {
                        Some((word_id, operator)) => {
                            let text: String = folded_chars[piece.start..piece.end]
                                .iter()
                                .collect();
                            push_word(tokens, word_id, operator, &text);
                        }
                        None => push_char(tokens, surface_chars[piece.start])?,
                    }
                }
                return Ok(());
            }
            for c in cased.chars() {
                push_char(tokens, c)?;
            }
            Ok(())
        };
        for item in candidate_items(self.table, text)? {
            let candidate = match item {
                CandidateItem::Plain(piece_text) => {
                    for c in piece_text.chars() {
                        push_char(&mut tokens, c)?;
                    }
                    continue;
                }
                CandidateItem::Candidate(candidate) => candidate,
            };
            if let Some(&id) = self.word_ids.get(&candidate.full) {
                match self.case_operator(&candidate.cased_full, &candidate.full) {
                    Some(operator) => {
                        push_word(&mut tokens, id, operator, &candidate.full);
                    }
                    None => push_cased(
                        &mut tokens,
                        id,
                        &candidate.full,
                        &candidate.cased_full,
                    )?,
                }
                continue;
            }
            let edged = candidate.leading.is_some() || candidate.trailing.is_some();
            if edged && let Some(&id) = self.word_ids.get(&candidate.core) {
                if let Some(c) = candidate.leading {
                    push_char(&mut tokens, c)?;
                }
                match self.case_operator(&candidate.cased_core, &candidate.core) {
                    Some(operator) => {
                        push_word(&mut tokens, id, operator, &candidate.core);
                    }
                    None => push_cased(
                        &mut tokens,
                        id,
                        &candidate.core,
                        &candidate.cased_core,
                    )?,
                }
                if let Some(c) = candidate.trailing {
                    push_char(&mut tokens, c)?;
                }
                continue;
            }
            if let Some(c) = candidate.leading {
                push_char(&mut tokens, c)?;
            }
            for part in &candidate.parts {
                match part {
                    CorePart::Word { folded, surface } => {
                        match self.word_ids.get(folded) {
                            Some(&id) => match self.case_operator(surface, folded) {
                                Some(operator) => {
                                    push_word(&mut tokens, id, operator, folded);
                                }
                                None => {
                                    push_cased(&mut tokens, id, folded, surface)?;
                                }
                            },
                            None => {
                                // The decomposition cover: dictionary
                                // subwords where they exist, characters
                                // where they do not. The simple fold is
                                // one-to-one per character, so spans map
                                // onto the cased surface. Camel casing
                                // marks the intended boundaries and wins
                                // outright when its segments all resolve.
                                let folded_chars: Vec<char> = folded.chars().collect();
                                let surface_chars: Vec<char> = surface.chars().collect();
                                let aligned =
                                    folded_chars.len() == surface_chars.len();
                                if aligned
                                    && let Some(pieces) = self
                                        .camel_cover(&folded_chars, &surface_chars)
                                {
                                    for piece in pieces {
                                        let text: String = folded_chars
                                            [piece.start..piece.end]
                                            .iter()
                                            .collect();
                                        push_word(
                                            &mut tokens,
                                            piece.id,
                                            piece.operator,
                                            &text,
                                        );
                                    }
                                    continue;
                                }
                                for piece in self.decompose_cover(&folded_chars) {
                                    if !piece.word {
                                        let c = if aligned {
                                            surface_chars[piece.start]
                                        } else {
                                            folded_chars[piece.start]
                                        };
                                        push_char(&mut tokens, c)?;
                                        continue;
                                    }
                                    let text: String = folded_chars
                                        [piece.start..piece.end]
                                        .iter()
                                        .collect();
                                    let id = self.word_ids[&text];
                                    if !aligned {
                                        push_word(&mut tokens, id, None, &text);
                                        continue;
                                    }
                                    let cased: String = surface_chars
                                        [piece.start..piece.end]
                                        .iter()
                                        .collect();
                                    match self.case_operator(&cased, &text) {
                                        Some(operator) => {
                                            push_word(&mut tokens, id, operator, &text);
                                        }
                                        None => {
                                            push_cased(&mut tokens, id, &text, &cased)?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    CorePart::Connector { surface, .. } => {
                        push_char(&mut tokens, *surface)?;
                    }
                }
            }
            if let Some(c) = candidate.trailing {
                push_char(&mut tokens, c)?;
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
pub(crate) enum TestPart<'a> {
    /// A `<|XX|>` spelling: the keyword id and the spelling itself.
    Marker(u8, &'a str),
    Text(&'a str),
}

/// Split a test string at keyword-page spellings (`<|00|>`-`<|FF|>`).
///
/// Test-surface rendering only: corpus ingestion never reads a marker
/// out of content - markers enter through deliberate rendering, and
/// this is that path for the CLI. The scan is byte-wise: the cursor
/// walks byte offsets, so every comparison is on bytes - a str slice
/// at the cursor would panic mid-multibyte-character.
pub(crate) fn split_markers(text: &str) -> Vec<TestPart<'_>> {
    let bytes = text.as_bytes();
    let mut parts = Vec::new();
    let mut rest_start = 0usize;
    let mut i = 0usize;
    'scan: while i < bytes.len() {
        // The hardcoded aliases spell like markers and mean their id.
        for (id, alias) in HARDCODED_ALIASES {
            if bytes[i..].starts_with(alias.as_bytes()) {
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
/// text). `<|XX|>` spellings render as keyword-page tokens, and a
/// `<|unicode|>`..`<|unicoded|>` span applies the tokenizer's span
/// semantics: one token per character, the exact surface, any other
/// spelling inside char-splitting as raw content. Without --words
/// the embedded full English set is the dictionary - which IS the
/// vocabulary (TheUser: the whole dictionary, file order as id
/// order).
pub(crate) fn tokenize_text(args: &TokenizeArgs) -> BiquestResult<()> {
    let table = CharacterTable::embedded()?;
    let buckets = BucketTable::new();
    let words = match &args.words {
        Some(path) => read_words_ordered(path)?,
        None => crate::dictionary::embedded_words(),
    };
    let segmenter = Segmenter::new(&table, &buckets, &words);
    let mut rows: Vec<harness::nu::Value> = Vec::new();
    let mut in_span = false;
    let push_chars = |rows: &mut Vec<harness::nu::Value>,
                          text: &str|
     -> BiquestResult<()> {
        for c in text.chars() {
            let token = segmenter.character_token(c)?;
            rows.push(harness::nu::Value::record(
                harness::nu::record! {
                    "token" => v_int(token.id as i64),
                    "unicode" => v_str(&format!("U+{:04X}", c as u32)),
                    "value" => v_str(&c.to_string()),
                },
                span(),
            ));
        }
        Ok(())
    };
    for part in split_markers(&args.text) {
        match part {
            TestPart::Marker(id, spelling) => {
                if in_span && id as u32 != KEYWORD_UNICODED {
                    // Inside a span every other spelling is raw
                    // content: the exact surface, char-split.
                    push_chars(&mut rows, spelling)?;
                    continue;
                }
                rows.push(harness::nu::Value::record(
                    harness::nu::record! {
                        "token" => v_int(id as i64),
                        "unicode" => harness::nu::Value::nothing(span()),
                        "value" => v_str(spelling),
                    },
                    span(),
                ));
                match id as u32 {
                    KEYWORD_UNICODE => in_span = true,
                    KEYWORD_UNICODED => in_span = false,
                    _ => {}
                }
            }
            TestPart::Text(text) => {
                if in_span {
                    push_chars(&mut rows, text)?;
                    continue;
                }
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
    snafu::ensure_whatever!(
        !in_span,
        "a unicode span never closes (missing <|unicoded|>)"
    );
    let rendered = harness::nu::to_nuon_pretty(&harness::nu::Value::list(rows, span()))?;
    println!("{rendered}");
    Ok(())
}
