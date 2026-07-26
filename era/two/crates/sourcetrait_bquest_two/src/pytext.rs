//! Python string semantics for the reference-faithful scorer ports.

use unicode_properties::GeneralCategory;
use unicode_properties::UnicodeGeneralCategory;

/// Python's string.punctuation (32 ASCII characters).
pub(crate) const PY_PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// Python's string.whitespace, ASCII only; not str.isspace's set.
pub(crate) const PY_ASCII_WHITESPACE: &str = " \t\n\r\x0b\x0c";

/// str.isspace / re \s: Unicode whitespace plus the ASCII separators.
pub(crate) fn py_is_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\x0b' | '\x0c' | '\r' | '\x1c' | '\x1d' | '\x1e' | '\x1f' | '\u{85}'
    ) || matches!(
        c.general_category(),
        GeneralCategory::SpaceSeparator
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator
    )
}

fn is_letter(c: char) -> bool {
    matches!(
        c.general_category(),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

fn is_number_category(c: char) -> bool {
    matches!(
        c.general_category(),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

/// re \w: letters, numbers (Nd/Nl/No), underscore.
pub(crate) fn py_is_word(c: char) -> bool {
    c == '_' || is_letter(c) || is_number_category(c)
}

/// str.isalnum (per character): letters and numbers.
pub(crate) fn py_is_alnum(c: char) -> bool {
    is_letter(c) || is_number_category(c)
}

/// re \d / str.isdecimal: decimal digits (Nd).
pub(crate) fn py_is_decimal(c: char) -> bool {
    c.general_category() == GeneralCategory::DecimalNumber
}

/// re [^\W\d] - what punkt calls "alpha".
pub(crate) fn py_is_word_nondigit(c: char) -> bool {
    py_is_word(c) && !py_is_decimal(c)
}

fn is_cased(c: char) -> bool {
    c.is_uppercase()
        || c.is_lowercase()
        || c.general_category() == GeneralCategory::TitlecaseLetter
}

/// str.isupper on a string.
pub(crate) fn py_str_isupper(s: &str) -> bool {
    let mut any_cased = false;
    for c in s.chars() {
        if is_cased(c) {
            any_cased = true;
            if !c.is_uppercase() {
                return false;
            }
        }
    }
    any_cased
}

/// str.islower on a string.
pub(crate) fn py_str_islower(s: &str) -> bool {
    let mut any_cased = false;
    for c in s.chars() {
        if is_cased(c) {
            any_cased = true;
            if !c.is_lowercase() {
                return false;
            }
        }
    }
    any_cased
}

/// str.isdigit on a string, to Nd precision.
pub(crate) fn py_str_isdigit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(py_is_decimal)
}

/// str.strip(chars): remove any of `set`'s characters from both ends.
pub(crate) fn py_strip_chars<'a>(s: &'a str, set: &str) -> &'a str {
    py_rstrip_chars(py_lstrip_chars(s, set), set)
}

pub(crate) fn py_lstrip_chars<'a>(s: &'a str, set: &str) -> &'a str {
    s.trim_start_matches(|c| set.contains(c))
}

pub(crate) fn py_rstrip_chars<'a>(s: &'a str, set: &str) -> &'a str {
    s.trim_end_matches(|c| set.contains(c))
}

/// str.strip() with no arguments: strip str.isspace characters.
pub(crate) fn py_strip_ws(s: &str) -> &str {
    s.trim_matches(py_is_space)
}

pub(crate) fn py_lstrip_ws(s: &str) -> &str {
    s.trim_start_matches(py_is_space)
}

/// str.split() with no arguments: runs of whitespace, empties dropped.
pub(crate) fn py_split_ws(s: &str) -> Vec<&str> {
    s.split(py_is_space).filter(|part| !part.is_empty()).collect()
}

/// str.translate deleting every character of `set`.
pub(crate) fn py_delete_chars(s: &str, set: &str) -> String {
    s.chars().filter(|c| !set.contains(*c)).collect()
}

/// Non-overlapping substring count (str.count).
pub(crate) fn py_count(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return haystack.chars().count() + 1;
    }
    haystack.matches(needle).count()
}

/// re.search(r"\b<literal>\b"), optionally ASCII-case-insensitive.
pub(crate) fn py_boundary_search(haystack: &str, needle: &str, ignore_case: bool) -> bool {
    if needle.is_empty() {
        return false;
    }
    let chars: Vec<char> = haystack.chars().collect();
    let needle_chars: Vec<char> = needle.chars().collect();
    let n = needle_chars.len();
    if chars.len() < n {
        return false;
    }
    let eq = |a: char, b: char| {
        if ignore_case {
            a.eq_ignore_ascii_case(&b)
        } else {
            a == b
        }
    };
    for start in 0..=(chars.len() - n) {
        if !needle_chars.iter().enumerate().all(|(k, &c)| eq(chars[start + k], c)) {
            continue;
        }
        let left_ok = start == 0 || !py_is_word(chars[start - 1]) || !py_is_word(chars[start]);
        let end = start + n;
        let right_ok =
            end == chars.len() || !py_is_word(chars[end]) || !py_is_word(chars[end - 1]);
        if left_ok && right_ok {
            return true;
        }
    }
    false
}

/// NFKD-decompose then keep only ASCII - the checkers' fold.
pub(crate) fn nfkd_ascii(s: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    s.nfkd().filter(|c| c.is_ascii()).collect()
}

/// RegexpTokenizer(r"\w+") token count (instructions_util.count_words).
pub(crate) fn py_word_run_count(s: &str) -> usize {
    let mut count = 0usize;
    let mut in_run = false;
    for c in s.chars() {
        if py_is_word(c) {
            if !in_run {
                count += 1;
                in_run = true;
            }
        } else {
            in_run = false;
        }
    }
    count
}

/// RegexpTokenizer(r"\w+") tokens themselves.
pub(crate) fn py_word_runs(s: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    for c in s.chars() {
        if py_is_word(c) {
            current.push(c);
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs
}
