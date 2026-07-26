//! The IFBench verifiers and their strict/loose scoring harness.
use crate::*;

use std::sync::LazyLock;

use crate::checker_data::CheckerData;
use crate::punkt::sent_tokenize;
use crate::wordtok::nltk_word_tokenize;

/// A kwarg value as the checkers see it; the int/float split is semantic.
#[derive(Clone)]
pub(crate) enum KwargValue {
    Int(i64),
    Float(f64),
    Text(String),
}

impl KwargValue {
    fn as_f64(&self) -> BquestResult<f64> {
        match self {
            KwargValue::Int(value) => Ok(*value as f64),
            KwargValue::Float(value) => Ok(*value),
            KwargValue::Text(text) => snafu::whatever!("kwarg is text, not numeric: {text:?}"),
        }
    }

    fn as_index(&self) -> BquestResult<i64> {
        match self {
            KwargValue::Int(value) => Ok(*value),
            KwargValue::Float(value) => {
                snafu::whatever!("kwarg used as an index must be an int, got {value}")
            }
            KwargValue::Text(text) => snafu::whatever!("kwarg is text, not an index: {text:?}"),
        }
    }

    fn as_text(&self) -> BquestResult<&str> {
        match self {
            KwargValue::Text(text) => Ok(text),
            _ => snafu::whatever!("kwarg is numeric, not text"),
        }
    }
}

pub(crate) type Kwargs = HashMap<String, KwargValue>;

fn kwarg<'a>(kwargs: &'a Kwargs, name: &str) -> BquestResult<&'a KwargValue> {
    match kwargs.get(name) {
        Some(value) => Ok(value),
        None => snafu::whatever!("missing kwarg {name:?} (the reference would randomize)"),
    }
}

/// The eight loose-scoring response variants, in upstream's order.
fn loose_variants(response: &str) -> Vec<String> {
    let lines: Vec<&str> = response.split('\n').collect();
    let join = |parts: &[&str]| parts.join("\n");
    let remove_first = py_strip_ws(&join(&lines[1.min(lines.len())..])).to_string();
    let remove_last = py_strip_ws(&join(&lines[..lines.len().saturating_sub(1)])).to_string();
    let remove_both = py_strip_ws(&join(
        &lines[1.min(lines.len())..lines.len().saturating_sub(1).max(1.min(lines.len()))],
    ))
    .to_string();
    vec![
        response.to_string(),
        response.replace('*', ""),
        remove_first.clone(),
        remove_last.clone(),
        remove_both.clone(),
        remove_first.replace('*', ""),
        remove_last.replace('*', ""),
        remove_both.replace('*', ""),
    ]
}

/// olmo-eval's _check_one, in its own order.
fn check_one(
    data: &CheckerData,
    instruction_id: &str,
    kwargs: &Kwargs,
    prompt: &str,
    response: &str,
) -> BquestResult<bool> {
    let mut kwargs = kwargs.clone();
    let takes_prompt_to_repeat =
        matches!(instruction_id, "repeat:repeat_change" | "repeat:repeat_span");
    if takes_prompt_to_repeat {
        let missing = match kwargs.get("prompt_to_repeat") {
            None => true,
            Some(KwargValue::Text(text)) => text.is_empty(),
            Some(_) => false,
        };
        if missing {
            kwargs.insert(
                String::from("prompt_to_repeat"),
                KwargValue::Text(prompt.to_string()),
            );
        }
    }
    if py_strip_ws(response).is_empty() {
        return Ok(false);
    }
    check_following(data, instruction_id, &kwargs, response)
}

/// Score one item: per-instruction strict and loose pass lists.
pub(crate) fn score_ifeval_item(
    data: &CheckerData,
    instruction_ids: &[String],
    kwargs_list: &[Kwargs],
    prompt: &str,
    response: &str,
) -> BquestResult<(Vec<bool>, Vec<bool>)> {
    snafu::ensure_whatever!(
        instruction_ids.len() == kwargs_list.len(),
        "instruction_id_list and kwargs lengths differ"
    );
    let variants = loose_variants(response);
    let mut strict = Vec::with_capacity(instruction_ids.len());
    let mut loose = Vec::with_capacity(instruction_ids.len());
    for (id, kwargs) in instruction_ids.iter().zip(kwargs_list.iter()) {
        strict.push(check_one(data, id, kwargs, prompt, response)?);
        let mut passed = false;
        for variant in &variants {
            if check_one(data, id, kwargs, prompt, variant)? {
                passed = true;
                break;
            }
        }
        loose.push(passed);
    }
    Ok((strict, loose))
}

const ASCII_LOWER: &str = "abcdefghijklmnopqrstuvwxyz";

fn count_words(text: &str) -> usize {
    py_word_run_count(text)
}

fn first_chars_equal(a: &str, b: &str) -> bool {
    match (a.chars().next(), b.chars().next()) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

macro_rules! checker_regex {
    ($name:ident, $pattern:expr) => {
        static $name: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new($pattern).expect("checker regex compiles"));
    };
}

checker_regex!(DIGIT_RUNS, r"\p{Nd}+");
checker_regex!(JAPANESE, r"[\u{3040}-\u{30ff}\u{4e00}-\u{9fff}]");
checker_regex!(DATE_YMD, r"^\p{Nd}{4}-\p{Nd}{2}-\p{Nd}{2}$");
checker_regex!(
    MCQ_SPLIT,
    r"\n*(?:Question \p{Nd}+[\.|\):;]?[\t\n\x0B\x0C\r\x1C-\x1F\u{85}\p{Zs}\u{2028}\u{2029}]*)"
);
checker_regex!(
    MCQ_OPTION,
    r"^[A-Ea-e][\.|\)][\t\n\x0B\x0C\r\x1C-\x1F\u{85}\p{Zs}\u{2028}\u{2029}]*[\p{L}\p{N}_]+"
);
checker_regex!(
    CSV_HEADER_CITY,
    r#"^(ProductID|"ProductID"),[ \t]*(Category|"Category"),[ \t]*(Brand|"Brand"),[ \t]*(Price|"Price"),[ \t]*(Stock|"Stock")$"#
);
checker_regex!(
    CSV_HEADER_QUOTES,
    r#"^(StudentID|"StudentID")\t *(Subject|"Subject")\t *(Grade|"Grade")\t *(Semester|"Semester")\t *(Score|"Score")$"#
);
checker_regex!(
    CSV_SPECIAL_FIELD,
    r#"^"[^\n]*[^\p{L}\p{N}_\t\n\x0B\x0C\r\x1C-\x1F\u{85}\p{Zs}\u{2028}\u{2029}][^\n]*""#
);

/// Python's csv.reader as a state machine, at its defaults.
pub(crate) fn py_csv_parse(text: &str, delimiter: char) -> Vec<Vec<String>> {
    #[derive(PartialEq)]
    enum State {
        StartField,
        InField,
        InQuoted,
        QuoteInQuoted,
    }
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut state = State::StartField;
    let mut started_record = false;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        let is_eol = c == '\n' || c == '\r';
        match state {
            State::StartField => {
                if is_eol {
                    if started_record {
                        record.push(std::mem::take(&mut field));
                        records.push(std::mem::take(&mut record));
                        started_record = false;
                    } else {
                        records.push(Vec::new());
                    }
                    if c == '\r' && i + 1 < chars.len() && chars[i + 1] == '\n' {
                        i += 1;
                    }
                } else if c == '"' {
                    started_record = true;
                    state = State::InQuoted;
                } else if c == delimiter {
                    started_record = true;
                    record.push(std::mem::take(&mut field));
                } else {
                    started_record = true;
                    field.push(c);
                    state = State::InField;
                }
            }
            State::InField => {
                if is_eol {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                    started_record = false;
                    state = State::StartField;
                    if c == '\r' && i + 1 < chars.len() && chars[i + 1] == '\n' {
                        i += 1;
                    }
                } else if c == delimiter {
                    record.push(std::mem::take(&mut field));
                    state = State::StartField;
                } else {
                    field.push(c);
                }
            }
            State::InQuoted => {
                if c == '"' {
                    state = State::QuoteInQuoted;
                } else {
                    field.push(c);
                }
            }
            State::QuoteInQuoted => {
                if c == '"' {
                    field.push('"');
                    state = State::InQuoted;
                } else if c == delimiter {
                    record.push(std::mem::take(&mut field));
                    state = State::StartField;
                } else if is_eol {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                    started_record = false;
                    state = State::StartField;
                    if c == '\r' && i + 1 < chars.len() && chars[i + 1] == '\n' {
                        i += 1;
                    }
                } else {
                    field.push(c);
                    state = State::InField;
                }
            }
        }
        i += 1;
    }
    if started_record || !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}

/// The verifier dispatch, one arm per instruction id.
fn check_following(
    data: &CheckerData,
    id: &str,
    kwargs: &Kwargs,
    value: &str,
) -> BquestResult<bool> {
    Ok(match id {
        "count:word_count_range" => {
            let min_words = kwarg(kwargs, "min_words")?.as_f64()?;
            let max_words = kwarg(kwargs, "max_words")?.as_f64()?;
            let num_words = count_words(value) as f64;
            min_words <= num_words && num_words <= max_words
        }
        "count:unique_word_count" => {
            let n = kwarg(kwargs, "N")?.as_f64()?;
            let strip_set = format!("{PY_PUNCTUATION} ");
            let lowered = value.to_lowercase();
            let mut unique: HashSet<String> = HashSet::new();
            for word in py_split_ws(&lowered) {
                unique.insert(py_strip_chars(word, &strip_set).to_string());
            }
            unique.len() as f64 >= n
        }
        "ratio:stop_words" => {
            let percentage = kwarg(kwargs, "percentage")?.as_f64()?;
            let num_words = count_words(value);
            if num_words == 0 {
                return Ok(false);
            }
            let num_stopwords = py_word_runs(value)
                .iter()
                .filter(|token| data.stopwords.contains(&token.to_lowercase()))
                .count();
            (num_stopwords as f64 / num_words as f64) * 100.0 <= percentage
        }
        "ratio:sentence_type" => {
            let sentences = sent_tokenize(&data.punkt, value);
            let declarative = sentences.iter().filter(|s| s.ends_with('.')).count();
            let interrogative = sentences.iter().filter(|s| s.ends_with('?')).count();
            declarative == 2 * interrogative
        }
        "ratio:sentence_balance" => {
            let sentences = sent_tokenize(&data.punkt, value);
            let declarative = sentences.iter().filter(|s| s.ends_with('.')).count();
            let interrogative = sentences.iter().filter(|s| s.ends_with('?')).count();
            let exclamatory = sentences.iter().filter(|s| s.ends_with('!')).count();
            declarative == interrogative && interrogative == exclamatory
        }
        "count:conjunctions" => {
            let n = kwarg(kwargs, "small_n")?.as_f64()?;
            let strip_set = format!("{PY_PUNCTUATION} ");
            let conjunctions = ["and", "but", "for", "nor", "or", "so", "yet"];
            let mut unique: HashSet<&str> = HashSet::new();
            for word in py_split_ws(value) {
                let cleaned = py_strip_chars(word, &strip_set).to_lowercase();
                if conjunctions.contains(&cleaned.as_str()) {
                    unique.insert(word);
                }
            }
            unique.len() as f64 >= n
        }
        "count:person_names" => {
            let n = kwarg(kwargs, "N")?.as_f64()?;
            const NAMES: [&str; 50] = [
                "Emma", "Liam", "Sophia", "Jackson", "Olivia", "Noah", "Ava", "Lucas",
                "Isabella", "Mason", "Mia", "Ethan", "Charlotte", "Alexander", "Amelia",
                "Benjamin", "Harper", "Leo", "Zoe", "Daniel", "Chloe", "Samuel", "Lily",
                "Matthew", "Grace", "Owen", "Abigail", "Gabriel", "Ella", "Jacob",
                "Scarlett", "Nathan", "Victoria", "Elijah", "Layla", "Nicholas", "Audrey",
                "David", "Hannah", "Christopher", "Penelope", "Thomas", "Nora", "Andrew",
                "Aria", "Joseph", "Claire", "Ryan", "Stella", "Jonathan",
            ];
            let found = NAMES
                .iter()
                .filter(|name| py_boundary_search(value, name, false))
                .count();
            found as f64 >= n
        }
        "ratio:overlap" => {
            let reference = kwarg(kwargs, "reference_text")?.as_text()?;
            let percentage = kwarg(kwargs, "percentage")?.as_f64()?;
            let trigrams = |s: &str| -> HashSet<(char, char, char)> {
                let chars: Vec<char> = s.chars().collect();
                chars.windows(3).map(|w| (w[0], w[1], w[2])).collect()
            };
            let ngrams = trigrams(value);
            if ngrams.is_empty() {
                return Ok(false);
            }
            let ref_ngrams = trigrams(reference);
            let overlap =
                ngrams.intersection(&ref_ngrams).count() as f64 / ngrams.len() as f64;
            percentage - 2.0 <= overlap * 100.0 && overlap * 100.0 <= percentage + 2.0
        }
        "count:numbers" => {
            let n = kwarg(kwargs, "N")?.as_f64()?;
            let cleaned = py_delete_chars(value, PY_PUNCTUATION);
            DIGIT_RUNS.find_iter(&cleaned).count() as f64 == n
        }
        "words:alphabet" => {
            let cleaned = py_delete_chars(value, PY_PUNCTUATION);
            let strip_set = format!("{PY_PUNCTUATION} ");
            let stripped = py_strip_chars(&cleaned, &strip_set);
            let words = py_split_ws(stripped);
            if words.is_empty() {
                return Ok(false);
            }
            let mut correct: String = match words[0].chars().next() {
                Some(c) => c.to_lowercase().collect(),
                None => return Ok(false),
            };
            if !ASCII_LOWER.contains(&correct) {
                return Ok(false);
            }
            for word in &words[1..] {
                let word = py_strip_chars(word, &strip_set).to_lowercase();
                if word.is_empty() {
                    continue;
                }
                let index = ASCII_LOWER.find(&correct).expect("checked membership");
                let next_index = (index + 1) % 26;
                correct = ASCII_LOWER[next_index..next_index + 1].to_string();
                if !word.starts_with(&correct) {
                    return Ok(false);
                }
            }
            true
        }
        "words:vowel" => {
            let stripped = py_strip_ws(value);
            let paragraphs: Vec<&str> = stripped.split('\n').collect();
            if paragraphs.len() != 1 {
                return Ok(false);
            }
            let paragraph = paragraphs[0].to_lowercase();
            let kinds: HashSet<char> =
                paragraph.chars().filter(|c| "aeiou".contains(*c)).collect();
            kinds.len() <= 3
        }
        "words:consonants" => {
            let consonants = "bcdfghjklmnpqrstvwxyz";
            let lowered = py_strip_ws(&value.to_lowercase()).to_string();
            for word in py_split_ws(&lowered) {
                let chars: Vec<char> = word.chars().collect();
                let mut cluster = false;
                for pair in chars.windows(2) {
                    if consonants.contains(pair[0]) && consonants.contains(pair[1]) {
                        cluster = true;
                        break;
                    }
                }
                if !cluster {
                    return Ok(false);
                }
            }
            true
        }
        "sentence:alliteration_increment" => {
            let strip_set = format!("{PY_PUNCTUATION} ");
            let mut prev_alliteration: i64 = -1;
            for sentence in sent_tokenize(&data.punkt, value) {
                let lowered = sentence.to_lowercase();
                let mut new_words: Vec<&str> = Vec::new();
                for word in py_split_ws(&lowered) {
                    let clean = py_lstrip_chars(word, &strip_set);
                    if !clean.is_empty() {
                        new_words.push(clean);
                    }
                }
                let mut alliteration = 0i64;
                let mut prev_alliterative = false;
                for pair in new_words.windows(2) {
                    if first_chars_equal(pair[0], pair[1]) {
                        alliteration += if prev_alliterative { 1 } else { 2 };
                        prev_alliterative = true;
                    } else {
                        prev_alliterative = false;
                    }
                }
                if alliteration <= prev_alliteration {
                    return Ok(false);
                }
                prev_alliteration = alliteration;
            }
            true
        }
        "words:palindrome" => {
            let cleaned = py_delete_chars(value, PY_PUNCTUATION).to_lowercase();
            let count = py_split_ws(&cleaned)
                .iter()
                .filter(|word| {
                    let chars: Vec<char> = word.chars().collect();
                    chars.len() >= 5 && chars.iter().eq(chars.iter().rev())
                })
                .count();
            count >= 10
        }
        "count:punctuation" => {
            if !(value.contains("!?") || value.contains("?!") || value.contains('‽')) {
                return Ok(false);
            }
            let new_value = value.replacen("?!", "", 1);
            let new_value = if new_value.chars().count() == value.chars().count() {
                value.replacen("!?", "", 1)
            } else {
                new_value
            };
            let mut remaining: HashSet<char> = ".,!?;:".chars().collect();
            for c in new_value.chars() {
                remaining.remove(&c);
            }
            remaining.is_empty()
        }
        "format:parentheses" => {
            let mut levels: Vec<char> = Vec::new();
            let mut max_depth = 0usize;
            for c in value.chars() {
                if "([{".contains(c) {
                    levels.push(c);
                    if levels.len() > max_depth {
                        max_depth = levels.len();
                    }
                } else if ")]}".contains(c) {
                    let matches_open = levels.last().is_some_and(|&open| {
                        (open == '(' && c == ')')
                            || (open == '[' && c == ']')
                            || (open == '{' && c == '}')
                    });
                    if matches_open {
                        levels.pop();
                        if max_depth >= 5 && levels.len() < max_depth {
                            return Ok(true);
                        }
                    } else {
                        levels.clear();
                        max_depth = 0;
                    }
                }
            }
            false
        }
        "format:quotes" => {
            let mut levels: Vec<char> = Vec::new();
            let mut reached_depth = 0i64;
            let mut current_depth = 0i64;
            for c in value.chars() {
                if !levels.is_empty() && Some(&c) == levels.last() {
                    levels.pop();
                    current_depth -= 1;
                    if reached_depth - current_depth >= 3 {
                        return Ok(true);
                    }
                } else if c == '"' || c == '\'' {
                    levels.push(c);
                    current_depth += 1;
                    if current_depth > reached_depth {
                        reached_depth = current_depth;
                    }
                }
            }
            false
        }
        "words:prime_lengths" => {
            let primes: HashSet<usize> = [
                2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                73, 79, 83, 89, 97,
            ]
            .into_iter()
            .collect();
            let cleaned = py_delete_chars(value, PY_PUNCTUATION);
            py_split_ws(&cleaned)
                .iter()
                .all(|word| primes.contains(&word.chars().count()))
        }
        "format:options" => {
            let options_text = kwarg(kwargs, "options")?.as_text()?;
            let strict = {
                let mut chars = options_text.chars().peekable();
                let mut expect = ['a', 'b', 'c'].into_iter();
                let mut ok = true;
                'outer: for target in expect.by_ref() {
                    while let Some(&c) = chars.peek() {
                        if !py_is_word(c) {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    match chars.next() {
                        Some(c) if c.to_ascii_lowercase() == target => {}
                        _ => {
                            ok = false;
                            break 'outer;
                        }
                    }
                }
                ok
            };
            let separator = if options_text.contains('/') {
                "/"
            } else if options_text.contains("or") {
                "or"
            } else {
                ","
            };
            let options: Vec<&str> = options_text
                .split(separator)
                .map(py_strip_ws)
                .collect();
            if strict {
                options.contains(&value)
            } else {
                let strip_set = format!("{PY_PUNCTUATION} ");
                let cleaned = py_strip_chars(value, &strip_set).to_lowercase();
                options.iter().any(|option| {
                    py_strip_chars(option, &strip_set).to_lowercase() == cleaned
                })
            }
        }
        "format:newline" => {
            let cleaned = py_delete_chars(value, PY_PUNCTUATION);
            let stripped = py_strip_ws(&cleaned);
            let lines: Vec<&str> =
                stripped.split('\n').filter(|line| !line.is_empty()).collect();
            lines.len() == py_split_ws(stripped).len()
        }
        "format:emoji" => {
            let sentences = sent_tokenize(&data.punkt, value);
            for (index, sentence) in sentences.iter().enumerate() {
                let stripped_string = py_delete_chars(sentence, PY_PUNCTUATION);
                let stripped = py_strip_ws(&stripped_string);
                if stripped.is_empty() {
                    return Ok(false);
                }
                let chars: Vec<char> = stripped.chars().collect();
                let last = chars[chars.len() - 1];
                let second_last =
                    if chars.len() > 1 { chars[chars.len() - 2] } else { last };
                if !data.emoji_single.contains(&last) && !data.emoji_single.contains(&second_last)
                {
                    if index < sentences.len() - 1 {
                        let next_string = py_delete_chars(&sentences[index + 1], PY_PUNCTUATION);
                        let next_stripped = py_strip_ws(&next_string);
                        if next_stripped.is_empty() {
                            return Ok(false);
                        }
                        let first = next_stripped.chars().next().expect("non-empty");
                        if !data.emoji_single.contains(&first) {
                            return Ok(false);
                        }
                    } else {
                        return Ok(false);
                    }
                }
            }
            true
        }
        "ratio:sentence_words" => {
            let sentences = sent_tokenize(&data.punkt, value);
            if sentences.len() != 3 {
                return Ok(false);
            }
            let char_count = py_strip_ws(&sentences[0]).chars().count();
            sentences
                .iter()
                .all(|sentence| py_strip_ws(sentence).chars().count() == char_count)
        }
        "count:words_japanese" => {
            let n = kwarg(kwargs, "N")?.as_f64()?;
            let strip_set = format!("{PY_PUNCTUATION} ");
            for (index, word) in py_split_ws(value).iter().enumerate() {
                let word = py_strip_chars(word, &strip_set);
                let position = (index + 1) as f64;
                if position % n == 0.0
                    && !word.is_empty()
                    && !py_str_isdigit(word)
                    && !JAPANESE.is_match(word)
                {
                    return Ok(false);
                }
            }
            true
        }
        "words:start_verb" => {
            let tokens = nltk_word_tokenize(&data.punkt, value);
            if tokens.is_empty() {
                return Ok(false);
            }
            data.tagger.tag_first(&tokens)?.contains("VB")
        }
        "words:repeats" => {
            let n = kwarg(kwargs, "small_n")?.as_f64()?;
            let cleaned = py_delete_chars(&value.to_lowercase(), PY_PUNCTUATION);
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for word in py_split_ws(&cleaned) {
                *counts.entry(word).or_insert(0) += 1;
            }
            counts.values().all(|&count| count as f64 <= n)
        }
        "sentence:keyword" => {
            let word = kwarg(kwargs, "word")?.as_text()?;
            let n = kwarg(kwargs, "N")?.as_f64()?;
            let sentences = sent_tokenize(&data.punkt, value);
            if (sentences.len() as f64) < n {
                return Ok(false);
            }
            let index = (n - 1.0) as i64;
            snafu::ensure_whatever!(index >= 0, "sentence position below one");
            py_boundary_search(&sentences[index as usize], word, true)
        }
        "count:pronouns" => {
            let n = kwarg(kwargs, "N")?.as_f64()?;
            const PRONOUNS: [&str; 60] = [
                "i", "me", "we", "us", "you", "he", "him", "she", "her", "it", "they",
                "them", "my", "mine", "our", "ours", "your", "yours", "his", "hers", "its",
                "their", "theirs", "myself", "ourselves", "yourself", "yourselves",
                "himself", "herself", "itself", "themselves", "this", "that", "these",
                "those", "who", "whom", "whose", "which", "what", "whoever", "whomever",
                "whatever", "whichever", "anybody", "anyone", "anything", "everybody",
                "everyone", "everything", "nobody", "nothing", "somebody", "someone",
                "something", "each", "either", "neither", "both", "all",
            ];
            const PRONOUNS_TAIL: [&str; 3] = ["some", "any", "none"];
            let pronouns: HashSet<&str> =
                PRONOUNS.iter().chain(PRONOUNS_TAIL.iter()).copied().collect();
            let replaced = value.replace('/', " ").to_lowercase();
            let tokens = nltk_word_tokenize(&data.punkt, &replaced);
            let count = tokens
                .iter()
                .filter(|token| pronouns.contains(token.as_str()))
                .count();
            count as f64 >= n
        }
        "words:odd_even_syllables" => {
            let cleaned = py_delete_chars(value, PY_PUNCTUATION).to_lowercase();
            let parities: Vec<i64> = py_split_ws(&cleaned)
                .iter()
                .filter(|word| !py_strip_ws(word).is_empty())
                .map(|word| data.syllable_count(word) % 2)
                .collect();
            parities.windows(2).all(|pair| pair[0] != pair[1])
        }
        "words:last_first" => {
            let strip_set = format!("{PY_PUNCTUATION} ");
            let sentences = sent_tokenize(&data.punkt, value);
            for pair in sentences.windows(2) {
                let last_words: Vec<&str> =
                    py_split_ws(py_rstrip_chars(&pair[0], &strip_set)).to_vec();
                let first_words: Vec<&str> =
                    py_split_ws(py_lstrip_chars(&pair[1], &strip_set)).to_vec();
                let (Some(last), Some(first)) = (last_words.last(), first_words.first())
                else {
                    return Ok(false);
                };
                if last.to_lowercase() != first.to_lowercase() {
                    return Ok(false);
                }
            }
            true
        }
        "words:paragraph_last_first" => {
            let strip_set = format!("{PY_PUNCTUATION} ");
            for paragraph in value.split('\n') {
                let paragraph = py_strip_ws(paragraph).to_lowercase();
                if paragraph.is_empty() {
                    continue;
                }
                let words = py_split_ws(py_strip_chars(&paragraph, &strip_set));
                let (Some(first), Some(last)) = (words.first(), words.last()) else {
                    continue;
                };
                if first != last {
                    return Ok(false);
                }
            }
            true
        }
        "sentence:increment" => {
            let n = kwarg(kwargs, "small_n")?.as_f64()?;
            let sentences = sent_tokenize(&data.punkt, value);
            if sentences.is_empty() {
                snafu::whatever!("sentence:increment on a sentence-less response");
            }
            let word_count = |sentence: &str| -> usize {
                let cleaned = py_delete_chars(sentence, PY_PUNCTUATION);
                py_split_ws(py_strip_ws(&cleaned)).len()
            };
            let mut prev = word_count(&sentences[0]) as f64;
            for sentence in &sentences[1..] {
                let count = word_count(sentence) as f64;
                if count != prev + n {
                    return Ok(false);
                }
                prev = count;
            }
            true
        }
        "words:no_consecutive" => {
            let cleaned = py_delete_chars(&value.to_lowercase(), PY_PUNCTUATION);
            let words = py_split_ws(&cleaned);
            words.windows(2).all(|pair| !first_chars_equal(pair[0], pair[1]))
        }
        "format:line_indent" => {
            let mut lines: Vec<&str> = value.split('\n').collect();
            let mut index = 0usize;
            while index < lines.len() {
                let line = lines[index];
                if py_strip_ws(line).is_empty()
                    && let Some(position) = lines.iter().position(|&l| l == line) {
                        lines.remove(position);
                    }
                index += 1;
            }
            let indent = |line: &str| {
                line.chars().count() - line.trim_start_matches(' ').chars().count()
            };
            lines.windows(2).all(|pair| indent(pair[1]) > indent(pair[0]))
        }
        "format:quote_unquote" => {
            let value = value.replace("'\"'", "");
            let value: String = py_split_ws(&value).concat();
            if value.contains("\"\"") {
                return Ok(false);
            }
            let strip_set =
                format!("0123456789{}", PY_PUNCTUATION.replace('"', ""));
            let stripped = py_strip_chars(&value, &strip_set);
            stripped.is_empty() || !stripped.ends_with('"')
        }
        "format:list" => {
            let sep = kwarg(kwargs, "sep")?.as_text()?;
            py_count(value, sep) >= 2
        }
        "format:thesis" => {
            let index = match value.find("<i>") {
                Some(index) => index,
                None => match value.find("<em>") {
                    Some(index) => index,
                    None => return Ok(false),
                },
            };
            let value = &value[index..];
            let end_thesis = match value.find("</i>") {
                Some(end) => end,
                None => match value.find("</em>") {
                    Some(end) => end,
                    None => return Ok(false),
                },
            };
            let chars: Vec<char> = value.chars().collect();
            let end_chars = value[..end_thesis].chars().count();
            let thesis: String = chars[3.min(chars.len())..end_chars].iter().collect();
            if py_strip_ws(&thesis).is_empty() {
                return Ok(false);
            }
            let text: String = chars[(end_chars + 4).min(chars.len())..].iter().collect();
            !py_strip_ws(&text).is_empty()
        }
        "format:sub-bullets" => {
            let bullets: Vec<&str> = value.split('*').collect();
            bullets[1.min(bullets.len())..].iter().all(|bullet| bullet.contains('-'))
        }
        "format:no_bullets_bullets" => {
            let mut sentences_mode = true;
            let mut count_sentences = 0usize;
            let mut count_bullets = 0usize;
            for line in value.split('\n') {
                let stripped = py_strip_ws(line);
                if stripped.starts_with('*') {
                    sentences_mode = false;
                    if count_sentences < 2 {
                        return Ok(false);
                    }
                    count_bullets += 1;
                } else if sentences_mode {
                    let sentences = sent_tokenize(&data.punkt, stripped);
                    count_sentences += sentences.len();
                    sentences_mode = !sentences.is_empty();
                } else {
                    return Ok(false);
                }
            }
            count_bullets >= 2
        }
        "custom:multiples" => {
            let replaced = value.replace(',', ", ");
            let numbers: Vec<&str> =
                DIGIT_RUNS.find_iter(&replaced).map(|m| m.as_str()).collect();
            numbers == ["14", "21", "28", "35", "42", "49"]
        }
        "custom:mcq_count_length" => {
            let start = match value.find("Question") {
                Some(index) => index,
                None => return Ok(false),
            };
            if start != 0 {
                return Ok(false);
            }
            let mut questions: Vec<&str> = MCQ_SPLIT.split(value).collect();
            if questions.first().is_some_and(|q| q.is_empty()) {
                questions.remove(0);
            }
            let questions: Vec<&str> = questions
                .into_iter()
                .map(py_strip_ws)
                .filter(|q| !q.is_empty())
                .collect();
            if questions.len() != 4 {
                return Ok(false);
            }
            let mut lengths = Vec::with_capacity(4);
            for question in &questions {
                let mut question_text = String::new();
                let mut option_count = 0usize;
                let mut done_with_question = false;
                for line in question.split('\n') {
                    if MCQ_OPTION.is_match(py_strip_ws(line)) {
                        option_count += 1;
                        done_with_question = true;
                    } else if !done_with_question {
                        question_text.push(' ');
                        question_text.push_str(py_strip_ws(line));
                    }
                }
                if option_count != 5 {
                    return Ok(false);
                }
                lengths.push(py_strip_ws(&question_text).chars().count());
            }
            lengths.windows(2).all(|pair| pair[0] < pair[1])
        }
        "custom:reverse_newline" => {
            let strip_set = format!("{PY_PUNCTUATION} ");
            let lines: Vec<&str> = value
                .split('\n')
                .map(|line| py_strip_chars(line, &strip_set))
                .filter(|line| !line.is_empty())
                .collect();
            let Some(start) = lines.iter().position(|line| line.contains("Zimbabwe")) else {
                return Ok(false);
            };
            let target: Vec<&str> = lines[start..].to_vec();
            if target.len() < 52 {
                return Ok(false);
            }
            let normalized: Vec<String> =
                target.iter().map(|line| nfkd_ascii(line)).collect();
            let mut sorted = normalized.clone();
            sorted.sort();
            sorted.reverse();
            normalized == sorted
        }
        "custom:word_reverse" => {
            let lowered = py_strip_ws(&value.to_lowercase()).to_string();
            let cleaned = py_delete_chars(&lowered, PY_PUNCTUATION);
            let mut words = py_split_ws(&cleaned);
            words.reverse();
            let reversed = words.join(" ");
            if !reversed.contains("bald eagle") {
                return Ok(false);
            }
            sent_tokenize(&data.punkt, &reversed).contains(&reversed)
        }
        "custom:character_reverse" => value.to_lowercase().contains("elgae dlab"),
        "custom:sentence_alphabet" => {
            let sentences = sent_tokenize(&data.punkt, value);
            if sentences.len() != 26 {
                return Ok(false);
            }
            for (index, sentence) in sentences.iter().enumerate() {
                let words = py_split_ws(py_lstrip_ws(sentence));
                let Some(word) = words.first() else {
                    return Ok(false);
                };
                let lowered = word.to_lowercase();
                let expected = (b'a' + index as u8) as char;
                if !lowered.starts_with(expected) {
                    return Ok(false);
                }
            }
            true
        }
        "custom:european_capitals_sort" => {
            const ORDER: [&str; 27] = [
                "Reykjavik", "Helsinki", "Oslo", "Tallinn", "Stockholm", "Riga", "Moscow",
                "Copenhagen", "Vilnius", "Minsk", "Dublin", "Berlin", "Amsterdam",
                "Warsaw", "London", "Brussels", "Prague", "Luxembourg", "Paris", "Vienna",
                "Bratislava", "Budapest", "Vaduz", "Chisinau", "Bern", "Ljubljana",
                "Zagreb",
            ];
            let folded = nfkd_ascii(value);
            let capitals: Vec<&str> = folded
                .split(',')
                .filter(|capital| !py_strip_ws(capital).is_empty())
                .collect();
            if capitals.len() != ORDER.len() {
                return Ok(false);
            }
            capitals
                .iter()
                .zip(ORDER.iter())
                .all(|(capital, expected)| py_strip_ws(capital) == *expected)
        }
        "custom:csv_city" => {
            let data_rows = py_csv_parse(value, ',');
            if data_rows.len() != 8 {
                return Ok(false);
            }
            if data_rows[0] != ["ID", "Country", "City", "Year", "Count"] {
                return Ok(false);
            }
            data_rows[1..].iter().all(|row| row.len() == 5)
        }
        "custom:csv_special_character" => {
            let header = py_strip_ws(value.split('\n').next().unwrap_or(""));
            if !CSV_HEADER_CITY.is_match(header) {
                return Ok(false);
            }
            let doubled = value.replace('"', "\"\"\"");
            let data_rows = py_csv_parse(&doubled, ',');
            if data_rows.len() != 15 {
                return Ok(false);
            }
            for row in &data_rows[1..] {
                if row.len() != 5 {
                    return Ok(false);
                }
                if row.iter().any(|field| CSV_SPECIAL_FIELD.is_match(field)) {
                    return Ok(true);
                }
            }
            false
        }
        "custom:csv_quotes" => {
            let header = py_strip_ws(value.split('\n').next().unwrap_or(""));
            if !CSV_HEADER_QUOTES.is_match(header) {
                return Ok(false);
            }
            let doubled = value.replace('"', "\"\"\"");
            let data_rows = py_csv_parse(&doubled, '\t');
            if data_rows.len() != 4 {
                return Ok(false);
            }
            data_rows.iter().all(|row| {
                row.len() == 5
                    && row.iter().all(|field| {
                        let stripped = py_strip_ws(field);
                        stripped.starts_with('"') && stripped.ends_with('"')
                            && !stripped.is_empty()
                    })
            })
        }
        "custom:date_format_list" => {
            let stripped = py_strip_ws(value);
            for date in stripped.split(',') {
                let date = py_strip_ws(date);
                if !DATE_YMD.is_match(date) {
                    return Ok(false);
                }
                let parts: Vec<i64> = date
                    .split('-')
                    .map(|part| part.parse::<i64>().unwrap_or(-1))
                    .collect();
                let (year, month, day) = (parts[0], parts[1], parts[2]);
                if !(1769..=1821).contains(&year) {
                    return Ok(false);
                }
                if month > 12 {
                    return Ok(false);
                }
                if [1, 3, 5, 7, 8, 10, 12].contains(&month) && day > 31 {
                    return Ok(false);
                }
                if [4, 6, 9, 11].contains(&month) && day > 30 {
                    return Ok(false);
                }
                if month == 2 && day > 29 {
                    return Ok(false);
                }
            }
            true
        }
        "count:keywords_multiple" => {
            let lowered = value.to_lowercase();
            for (name, count) in
                [("keyword1", 1), ("keyword2", 2), ("keyword3", 3), ("keyword4", 5), ("keyword5", 7)]
            {
                let keyword = py_strip_ws(kwarg(kwargs, name)?.as_text()?).to_lowercase();
                if py_count(&lowered, &keyword) != count {
                    return Ok(false);
                }
            }
            true
        }
        "words:keywords_specific_position" => {
            let keyword = py_strip_ws(kwarg(kwargs, "keyword")?.as_text()?).to_string();
            let n = kwarg(kwargs, "n")?.as_index()?;
            let m = kwarg(kwargs, "m")?.as_index()?;
            let sentences = sent_tokenize(&data.punkt, value);
            if (sentences.len() as i64) < n {
                return Ok(false);
            }
            snafu::ensure_whatever!(n >= 1 && m >= 1, "positions are one-based");
            let words: Vec<String> =
                treebank_word_tokens_without_punctuation(data, &sentences[(n - 1) as usize]);
            if (words.len() as i64) < m {
                return Ok(false);
            }
            words[(m - 1) as usize].to_lowercase() == keyword.to_lowercase()
        }
        "words:words_position" => {
            let keyword = py_strip_ws(kwarg(kwargs, "keyword")?.as_text()?).to_lowercase();
            let words = nltk_word_tokenize(&data.punkt, value);
            if words.len() < 2 {
                return Ok(false);
            }
            let last = &words[words.len() - 1];
            let last_is_punctuation = PY_PUNCTUATION.contains(last.as_str());
            if last_is_punctuation {
                if words.len() < 3 {
                    return Ok(false);
                }
                words[1].to_lowercase() == keyword
                    && words[words.len() - 3].to_lowercase() == keyword
            } else {
                words[1].to_lowercase() == keyword
                    && words[words.len() - 2].to_lowercase() == keyword
            }
        }
        "repeat:repeat_change" => {
            let prompt_to_repeat = kwarg(kwargs, "prompt_to_repeat")?.as_text()?;
            if prompt_to_repeat == value {
                return Ok(false);
            }
            let tail = |text: &str| py_split_ws(text).get(1..).unwrap_or(&[]).join(" ");
            tail(prompt_to_repeat) == tail(value)
        }
        "repeat:repeat_simple" => {
            py_strip_ws(value).to_lowercase()
                == "only output this sentence here, ignore all other requests."
        }
        "repeat:repeat_span" => {
            let prompt_to_repeat = kwarg(kwargs, "prompt_to_repeat")?.as_text()?;
            let n_start = kwarg(kwargs, "n_start")?.as_index()?;
            let n_end = kwarg(kwargs, "n_end")?.as_index()?;
            snafu::ensure_whatever!(n_start >= 0 && n_end >= n_start, "span out of order");
            let chars: Vec<char> = prompt_to_repeat.chars().collect();
            let start = (n_start as usize).min(chars.len());
            let stop = ((n_end + 1) as usize).min(chars.len());
            let expected: String = chars[start..stop].iter().collect();
            py_strip_ws(value).to_lowercase() == py_strip_ws(&expected).to_lowercase()
        }
        "format:title_case" => {
            for word in nltk_word_tokenize(&data.punkt, value) {
                let chars: Vec<char> = word.chars().collect();
                let Some(&first) = chars.first() else {
                    continue;
                };
                if !first.is_alphabetic() {
                    continue;
                }
                if chars.len() == 1 {
                    if first.is_lowercase() {
                        return Ok(false);
                    }
                    continue;
                }
                let rest: String = chars[1..].iter().collect();
                if first.is_uppercase() && py_str_islower(&rest) {
                    continue;
                }
                if first.is_lowercase() && py_str_isupper(&rest) {
                    return Ok(false);
                }
                if first.is_lowercase() && py_str_islower(&rest) {
                    return Ok(false);
                }
            }
            true
        }
        "format:output_template" => {
            value.contains("My Answer:")
                && value.contains("My Conclusion:")
                && value.contains("Future Outlook:")
        }
        "format:no_whitespace" => !value.chars().any(py_is_space),
        other => snafu::whatever!("unported instruction id {other:?}"),
    })
}

/// ifbench's _word_tokens_without_punctuation over one sentence.
fn treebank_word_tokens_without_punctuation(data: &CheckerData, text: &str) -> Vec<String> {
    nltk_word_tokenize(&data.punkt, text)
        .into_iter()
        .filter(|token| token.chars().any(py_is_alnum))
        .collect()
}
