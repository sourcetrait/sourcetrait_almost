//! The Unicode character table: every assigned code point, from the UCD.
use crate::*;

/// The vendored Unicode 17.0.0 UCD, embedded so the binary needs no
/// external data; re-vendor `data/ucd/` on a Unicode version bump.
const UNICODE_DATA: &str = include_str!("../data/ucd/UnicodeData.txt");
const PROP_LIST: &str = include_str!("../data/ucd/PropList.txt");
const CASE_FOLDING: &str = include_str!("../data/ucd/CaseFolding.txt");
const SCRIPTS: &str = include_str!("../data/ucd/Scripts.txt");

/// The lexer's classification of an assigned code point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CharClass {
    /// Letters, marks, and numbers: word-constituent.
    Word,
    /// Whitespace, punctuation, and symbols: always a boundary, one
    /// token per character.
    Symbol,
    /// Assigned but outside the classes above; a lex-time refusal.
    Other,
}

/// One assigned code point's row.
pub(crate) struct CharRow {
    pub(crate) code_point: u32,
    /// Index into `CharacterTable::general_categories`.
    pub(crate) category: u8,
    /// Index into `CharacterTable::scripts`; u16::MAX when unlisted.
    pub(crate) script: u16,
    pub(crate) white_space: bool,
    /// Simple case fold target; the code point itself when unmapped.
    pub(crate) fold: u32,
}

/// Every Unicode-assigned code point, ordered, surrogates and PUA excluded.
pub(crate) struct CharacterTable {
    pub(crate) rows: Vec<CharRow>,
    index_of: HashMap<u32, u32>,
    pub(crate) general_categories: Vec<String>,
    pub(crate) scripts: Vec<String>,
    /// Canonical decompositions, sparse (compatibility ones are skipped).
    pub(crate) decompositions: HashMap<u32, Vec<u32>>,
    /// Numeric values (UnicodeData field 8), sparse, verbatim.
    pub(crate) numeric_values: HashMap<u32, String>,
    /// Simple uppercase/lowercase pairs (fields 12/13), sparse.
    pub(crate) simple_uppercase: HashMap<u32, u32>,
    pub(crate) simple_lowercase: HashMap<u32, u32>,
}

fn parse_hex(text: &str) -> BiquestResult<u32> {
    match u32::from_str_radix(text.trim(), 16) {
        Ok(value) => Ok(value),
        Err(e) => snafu::whatever!("bad code point {text:?}: {e}"),
    }
}

/// A `XXXX` or `XXXX..YYYY` range head from a UCD data line.
fn parse_range(text: &str) -> BiquestResult<(u32, u32)> {
    match text.split_once("..") {
        Some((first, last)) => Ok((parse_hex(first)?, parse_hex(last)?)),
        None => {
            let single = parse_hex(text)?;
            Ok((single, single))
        }
    }
}

/// Data lines of a UCD file: comments and blanks dropped, fields kept raw.
fn data_lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(|line| match line.split_once('#') {
            Some((data, _)) => data.trim_end(),
            None => line.trim_end(),
        })
        .filter(|line| !line.trim().is_empty())
        .collect()
}

impl CharacterTable {
    /// The embedded Unicode 17 table; the only constructor in use.
    pub(crate) fn embedded() -> BiquestResult<Self> {
        let mut table = Self::from_unicode_data(UNICODE_DATA)?;
        table.apply_white_space(PROP_LIST)?;
        table.apply_case_folding(CASE_FOLDING)?;
        table.apply_scripts(SCRIPTS)?;
        Ok(table)
    }

    fn from_unicode_data(text: &str) -> BiquestResult<Self> {
        let mut rows: Vec<CharRow> = Vec::with_capacity(200_000);
        let mut general_categories: Vec<String> = Vec::new();
        let mut decompositions = HashMap::new();
        let mut numeric_values = HashMap::new();
        let mut simple_uppercase = HashMap::new();
        let mut simple_lowercase = HashMap::new();
        let mut range_first: Option<(u32, u8)> = None;

        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(';').collect();
            snafu::ensure_whatever!(
                fields.len() >= 15,
                "UnicodeData line with {} fields: {line:?}",
                fields.len()
            );
            let code_point = parse_hex(fields[0])?;
            let name = fields[1];
            let category = fields[2];
            // Surrogates and private use get no rows: UTF-16 machinery
            // and private meaning are excluded by design.
            if category == "Cs" || category == "Co" {
                continue;
            }
            let category_index = match general_categories
                .iter()
                .position(|known| known == category)
            {
                Some(index) => index as u8,
                None => {
                    general_categories.push(category.to_string());
                    (general_categories.len() - 1) as u8
                }
            };

            if name.ends_with(", First>") {
                range_first = Some((code_point, category_index));
                continue;
            }
            if name.ends_with(", Last>") {
                let Some((first, first_category)) = range_first.take() else {
                    snafu::whatever!("range Last without First: {line:?}");
                };
                snafu::ensure_whatever!(
                    first_category == category_index,
                    "range category mismatch at {line:?}"
                );
                for point in first..=code_point {
                    rows.push(CharRow {
                        code_point: point,
                        category: category_index,
                        script: u16::MAX,
                        white_space: false,
                        fold: point,
                    });
                }
                continue;
            }

            // Canonical decomposition only: a <tag> marks compatibility.
            let decomposition = fields[5].trim();
            if !decomposition.is_empty() && !decomposition.starts_with('<') {
                let mut points = Vec::new();
                for part in decomposition.split_whitespace() {
                    points.push(parse_hex(part)?);
                }
                decompositions.insert(code_point, points);
            }
            let numeric = fields[8].trim();
            if !numeric.is_empty() {
                numeric_values.insert(code_point, numeric.to_string());
            }
            let upper = fields[12].trim();
            if !upper.is_empty() {
                simple_uppercase.insert(code_point, parse_hex(upper)?);
            }
            let lower = fields[13].trim();
            if !lower.is_empty() {
                simple_lowercase.insert(code_point, parse_hex(lower)?);
            }

            rows.push(CharRow {
                code_point,
                category: category_index,
                script: u16::MAX,
                white_space: false,
                fold: code_point,
            });
        }
        snafu::ensure_whatever!(
            range_first.is_none(),
            "UnicodeData ended inside a First/Last range"
        );

        let index_of: HashMap<u32, u32> = rows
            .iter()
            .enumerate()
            .map(|(index, row)| (row.code_point, index as u32))
            .collect();
        Ok(Self {
            rows,
            index_of,
            general_categories,
            scripts: Vec::new(),
            decompositions,
            numeric_values,
            simple_uppercase,
            simple_lowercase,
        })
    }

    fn apply_white_space(&mut self, text: &str) -> BiquestResult<()> {
        for line in data_lines(text) {
            let Some((range, property)) = line.split_once(';') else {
                continue;
            };
            if property.trim() != "White_Space" {
                continue;
            }
            let (first, last) = parse_range(range.trim())?;
            for point in first..=last {
                if let Some(&index) = self.index_of.get(&point) {
                    self.rows[index as usize].white_space = true;
                }
            }
        }
        Ok(())
    }

    fn apply_case_folding(&mut self, text: &str) -> BiquestResult<()> {
        for line in data_lines(text) {
            let fields: Vec<&str> = line.split(';').map(str::trim).collect();
            if fields.len() < 3 {
                continue;
            }
            // C and S give the simple, one-to-one fold the lexer uses.
            if fields[1] != "C" && fields[1] != "S" {
                continue;
            }
            let from = parse_hex(fields[0])?;
            let to = parse_hex(fields[2])?;
            if let Some(&index) = self.index_of.get(&from) {
                self.rows[index as usize].fold = to;
            }
        }
        Ok(())
    }

    fn apply_scripts(&mut self, text: &str) -> BiquestResult<()> {
        for line in data_lines(text) {
            let Some((range, script)) = line.split_once(';') else {
                continue;
            };
            let script = script.trim();
            let script_index = match self.scripts.iter().position(|known| known == script) {
                Some(index) => index as u16,
                None => {
                    snafu::ensure_whatever!(
                        self.scripts.len() < u16::MAX as usize,
                        "script vocabulary exceeds u16"
                    );
                    self.scripts.push(script.to_string());
                    (self.scripts.len() - 1) as u16
                }
            };
            let (first, last) = parse_range(range.trim())?;
            for point in first..=last {
                if let Some(&index) = self.index_of.get(&point) {
                    self.rows[index as usize].script = script_index;
                }
            }
        }
        Ok(())
    }

    /// A hand-built table for unit tests; rows must be code-point sorted.
    #[cfg(test)]
    pub(crate) fn from_rows(rows: Vec<CharRow>, general_categories: Vec<String>) -> Self {
        let index_of = rows
            .iter()
            .enumerate()
            .map(|(index, row)| (row.code_point, index as u32))
            .collect();
        Self {
            rows,
            index_of,
            general_categories,
            scripts: Vec::new(),
            decompositions: HashMap::new(),
            numeric_values: HashMap::new(),
            simple_uppercase: HashMap::new(),
            simple_lowercase: HashMap::new(),
        }
    }

    pub(crate) fn assigned_count(&self) -> usize {
        self.rows.len()
    }

    /// The dense character-layer index of an assigned code point.
    pub(crate) fn index_of(&self, code_point: u32) -> Option<u32> {
        self.index_of.get(&code_point).copied()
    }

    pub(crate) fn class_of_row(&self, row: &CharRow) -> CharClass {
        if row.white_space {
            return CharClass::Symbol;
        }
        let category = &self.general_categories[row.category as usize];
        match category.as_bytes().first() {
            Some(b'L') | Some(b'M') | Some(b'N') => CharClass::Word,
            Some(b'P') | Some(b'S') => CharClass::Symbol,
            _ => CharClass::Other,
        }
    }

    /// Class of a character, or None when the code point is unassigned.
    pub(crate) fn class_of(&self, c: char) -> Option<CharClass> {
        let index = self.index_of(c as u32)?;
        Some(self.class_of_row(&self.rows[index as usize]))
    }

    /// Simple case fold, or None when the code point is unassigned.
    pub(crate) fn fold(&self, c: char) -> Option<char> {
        let index = self.index_of(c as u32)?;
        let target = self.rows[index as usize].fold;
        char::from_u32(target)
    }

    /// Fold a string character-wise; None names the first refused char.
    pub(crate) fn fold_str(&self, text: &str) -> Result<String, char> {
        let mut folded = String::with_capacity(text.len());
        for c in text.chars() {
            match self.fold(c) {
                Some(target) => folded.push(target),
                None => return Err(c),
            }
        }
        Ok(folded)
    }
}
