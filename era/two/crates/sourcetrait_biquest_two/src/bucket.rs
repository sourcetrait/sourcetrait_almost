//! The keyboard-symbol bucket: doubles and triples of every keyboard
//! symbol, one uniform rule, association-hinted to the repetition
//! concept.
use crate::*;

/// The keyboard symbol characters, ascending: tab, space, and the 32
/// printable ASCII symbols. Repetition happens on the keyboard most,
/// so every one of these gets a double and a triple row - a closed
/// rule, never a frequency pick.
pub(crate) const KEYBOARD_CHARS: [char; 34] = [
    '\t', ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.',
    '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|',
    '}', '~',
];

/// One keyboard-symbol entry: the character and its run length (2 or
/// 3). The pair IS the parameter association: at matrix time each row
/// associates to its character row, the `<|repetition|>` concept
/// marker, and its count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KeyboardSymbolBucket {
    pub(crate) symbol: char,
    pub(crate) count: u8,
}

impl KeyboardSymbolBucket {
    /// The character sequence this entry tokenizes.
    pub(crate) fn sequence(&self) -> String {
        self.symbol.to_string().repeat(self.count as usize)
    }
}

/// The bucket layer's table: canonical entries plus the (char, count)
/// index the banded encoder resolves runs through.
pub(crate) struct BucketTable {
    entries: Vec<KeyboardSymbolBucket>,
    index_of: HashMap<(char, u8), usize>,
}

impl BucketTable {
    /// The canonical enumeration: per character ascending, double
    /// then triple. Entry position IS the in-layer id, append-only
    /// forever.
    pub(crate) fn new() -> Self {
        let mut entries = Vec::with_capacity(KEYBOARD_CHARS.len() * 2);
        for symbol in KEYBOARD_CHARS {
            entries.push(KeyboardSymbolBucket { symbol, count: 2 });
            entries.push(KeyboardSymbolBucket { symbol, count: 3 });
        }
        let index_of = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| ((entry.symbol, entry.count), index))
            .collect();
        Self { entries, index_of }
    }

    pub(crate) fn count(&self) -> usize {
        self.entries.len()
    }

    /// The entry at an in-layer index.
    pub(crate) fn entry(&self, index: usize) -> KeyboardSymbolBucket {
        self.entries[index]
    }

    /// The in-layer index for a character's double (2) or triple (3).
    pub(crate) fn index_for(&self, symbol: char, count: u8) -> Option<usize> {
        self.index_of.get(&(symbol, count)).copied()
    }
}
