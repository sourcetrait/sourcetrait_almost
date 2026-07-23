//! nltk's NLTKWordTokenizer (destructive.py, nltk 3.10.0): the exact
//! regex-substitution battery, with Python's \s spelled out and the
//! two lookahead-bearing rules (the clitic quote split and the
//! contraction list) hand-scanned. nltk.word_tokenize = punkt
//! sentence split, then this per sentence.
use crate::*;

use std::sync::LazyLock;

use crate::punkt::PunktParams;
use crate::punkt::sent_tokenize;

macro_rules! rules {
    ($($name:ident = $pattern:expr;)*) => {
        $(static $name: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new($pattern).expect("rule compiles"));)*
    };
}

const PYS: &str = r"\t\n\x0B\x0C\r\x1C-\x1F\u{85}\p{Zs}\u{2028}\u{2029}";

rules! {
    STARTING_1 = "([«“‘„]|`+)";
    STARTING_2 = "^\"";
    STARTING_3 = "(``)";
    STARTING_4 = r#"([ \(\[{<])("|'{2})"#;
    PUNCT_1 = r#"([^\.])(\.)([\]\)}>"'»”’ ]*)[\t\n\x0B\x0C\r\x1C-\x1F\u{85}\p{Zs}\u{2028}\u{2029}]*$"#;
    PUNCT_2 = r"([:,])([^\p{Nd}])";
    PUNCT_3 = "([:,])$";
    PUNCT_4 = r"\.{2,}";
    PUNCT_5 = "[;@#$%&]";
    PUNCT_6 = r"[\u{2012}-\u{2015}]";
    PUNCT_7 = r#"([^\.])(\.)([\]\)}>"']*)[\t\n\x0B\x0C\r\x1C-\x1F\u{85}\p{Zs}\u{2028}\u{2029}]*$"#;
    PUNCT_8 = "[?!]";
    PUNCT_9 = "([^'])' ";
    PUNCT_10 = r"[*]";
    PARENS = r"[\]\[\(\)\{\}<>]";
    DASHES = "--";
    ENDING_1 = "([»”’])";
    ENDING_2 = "''";
    ENDING_3 = "\"";
    ENDING_5 = "([^' ])('[sS]|'[mM]|'[dD]|') ";
    ENDING_6 = "([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ";
}

static ENDING_4: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(&format!("[{PYS}]+")).expect("rule compiles"));

/// STARTING_QUOTES rule 5: (?i)(')(?!re|ve|ll|m|t|s|d|n)(\w)\b - a
/// quote before a single-letter word, negative lookahead hand-checked.
fn split_clitic_quotes(text: &str) -> String {
    const BLOCKED: [&str; 8] = ["re", "ve", "ll", "m", "t", "s", "d", "n"];
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' && i + 1 < chars.len() && py_is_word(chars[i + 1]) {
            let boundary_ok =
                i + 2 >= chars.len() || !py_is_word(chars[i + 2]);
            let blocked = BLOCKED.iter().any(|alt| {
                chars[i + 1..]
                    .iter()
                    .take(alt.len())
                    .map(|c| c.to_ascii_lowercase())
                    .eq(alt.chars())
            });
            if boundary_ok && !blocked {
                out.push('\'');
                out.push(' ');
                out.push(chars[i + 1]);
                i += 2;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

enum Tail {
    WordBoundary,
    SpaceLookahead,
}

/// One MacIntyre contraction rule: an optional literal leading space,
/// two case-preserved parts split by a space in the output, \b (or a
/// \s lookahead) on the right, \b on the left when no leading space.
fn contraction_replace(
    text: &str,
    leading_space: bool,
    part1: &str,
    part2: &str,
    tail: Tail,
) -> String {
    let chars: Vec<char> = text.chars().collect();
    let needle: Vec<char> = format!("{part1}{part2}").chars().collect();
    let part1_len = part1.chars().count();
    let mut out = String::new();
    let mut i = 0usize;
    while i < chars.len() {
        let mut start = i;
        let matched = 'm: {
            if leading_space {
                if chars[i] != ' ' {
                    break 'm false;
                }
                start = i + 1;
            } else {
                let left_word = i > 0 && py_is_word(chars[i - 1]);
                if left_word && py_is_word(chars[i]) {
                    break 'm false;
                }
            }
            if start + needle.len() > chars.len() {
                break 'm false;
            }
            if !needle
                .iter()
                .enumerate()
                .all(|(k, &n)| chars[start + k].to_ascii_lowercase() == n)
            {
                break 'm false;
            }
            let end = start + needle.len();
            match tail {
                Tail::WordBoundary => end >= chars.len() || !py_is_word(chars[end]),
                Tail::SpaceLookahead => end < chars.len() && py_is_space(chars[end]),
            }
        };
        if matched {
            // The replacement " \1 \2 " carries exactly one leading
            // space whether or not the pattern consumed one.
            out.push(' ');
            for &c in &chars[start..start + part1_len] {
                out.push(c);
            }
            out.push(' ');
            for &c in &chars[start + part1_len..start + needle.len()] {
                out.push(c);
            }
            out.push(' ');
            i = start + needle.len();
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// NLTKWordTokenizer.tokenize (convert_parentheses = False).
pub(crate) fn treebank_tokenize(text: &str) -> Vec<String> {
    let mut text = text.to_string();

    text = STARTING_1.replace_all(&text, " $1 ").into_owned();
    text = STARTING_2.replace_all(&text, "``").into_owned();
    text = STARTING_3.replace_all(&text, " $1 ").into_owned();
    text = STARTING_4.replace_all(&text, "$1 `` ").into_owned();
    text = split_clitic_quotes(&text);

    text = PUNCT_1.replace_all(&text, "$1 $2 $3 ").into_owned();
    text = PUNCT_2.replace_all(&text, " $1 $2").into_owned();
    text = PUNCT_3.replace_all(&text, " $1 ").into_owned();
    text = PUNCT_4.replace_all(&text, " $0 ").into_owned();
    text = PUNCT_5.replace_all(&text, " $0 ").into_owned();
    text = PUNCT_6.replace_all(&text, " $0 ").into_owned();
    text = PUNCT_7.replace_all(&text, "$1 $2$3 ").into_owned();
    text = PUNCT_8.replace_all(&text, " $0 ").into_owned();
    text = PUNCT_9.replace_all(&text, "$1 ' ").into_owned();
    text = PUNCT_10.replace_all(&text, " $0 ").into_owned();

    text = PARENS.replace_all(&text, " $0 ").into_owned();
    text = DASHES.replace_all(&text, " -- ").into_owned();

    text = format!(" {text} ");

    text = ENDING_1.replace_all(&text, " $1 ").into_owned();
    text = ENDING_2.replace_all(&text, " '' ").into_owned();
    text = ENDING_3.replace_all(&text, " '' ").into_owned();
    text = ENDING_4.replace_all(&text, " ").into_owned();
    text = ENDING_5.replace_all(&text, "$1 $2 ").into_owned();
    text = ENDING_6.replace_all(&text, "$1 $2 ").into_owned();

    text = contraction_replace(&text, false, "can", "not", Tail::WordBoundary);
    text = contraction_replace(&text, false, "d", "'ye", Tail::WordBoundary);
    text = contraction_replace(&text, false, "gim", "me", Tail::WordBoundary);
    text = contraction_replace(&text, false, "gon", "na", Tail::WordBoundary);
    text = contraction_replace(&text, false, "got", "ta", Tail::WordBoundary);
    text = contraction_replace(&text, false, "lem", "me", Tail::WordBoundary);
    text = contraction_replace(&text, false, "more", "'n", Tail::WordBoundary);
    text = contraction_replace(&text, false, "wan", "na", Tail::SpaceLookahead);
    text = contraction_replace(&text, true, "'t", "is", Tail::WordBoundary);
    text = contraction_replace(&text, true, "'t", "was", Tail::WordBoundary);

    py_split_ws(&text).into_iter().map(str::to_string).collect()
}

/// nltk.word_tokenize: punkt sentence split then the treebank battery.
pub(crate) fn nltk_word_tokenize(params: &PunktParams, text: &str) -> Vec<String> {
    sent_tokenize(params, text)
        .iter()
        .flat_map(|sentence| treebank_tokenize(sentence))
        .collect()
}
