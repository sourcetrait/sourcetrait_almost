//! Byte copies of the evaluation environment's own checker data files.
use crate::*;

use crate::punkt::PunktParams;
use crate::tagger::PerceptronTagger;

pub(crate) struct CheckerData {
    pub(crate) punkt: PunktParams,
    pub(crate) tagger: PerceptronTagger,
    pub(crate) stopwords: HashSet<String>,
    pub(crate) syllapy: HashMap<String, i64>,
    pub(crate) emoji_single: HashSet<char>,
}

impl CheckerData {
    pub(crate) fn load(dir: &Path) -> BquestResult<Self> {
        let punkt = PunktParams::load(&dir.join("punkt_tab_english"))?;
        let tagger = PerceptronTagger::load(&dir.join("tagger"))?;

        let stopwords: HashSet<String> = fs::read_to_string(dir.join("stopwords_english.txt"))?
            .lines()
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect();

        let mut syllapy = HashMap::new();
        for line in fs::read_to_string(dir.join("syllapy_data.csv"))?.lines() {
            if line.is_empty() {
                continue;
            }
            let Some((word, count)) = line.split_once(',') else {
                snafu::whatever!("syllapy_data.csv line without a comma: {line:?}");
            };
            let Ok(count) = count.trim().parse::<i64>() else {
                snafu::whatever!("syllapy_data.csv count not an int: {line:?}");
            };
            syllapy.insert(word.to_string(), count);
        }

        let emoji_text = fs::read_to_string(dir.join("emoji.json"))?;
        let emoji_json: serde_json::Value = serde_json::from_str(&emoji_text)?;
        let Some(emoji_map) = emoji_json.as_object() else {
            snafu::whatever!("emoji.json is not an object");
        };
        let emoji_single: HashSet<char> = emoji_map
            .keys()
            .filter_map(|key| {
                let mut chars = key.chars();
                match (chars.next(), chars.next()) {
                    (Some(c), None) => Some(c),
                    _ => None,
                }
            })
            .collect();

        Ok(Self { punkt, tagger, stopwords, syllapy, emoji_single })
    }

    /// syllapy.count.
    pub(crate) fn syllable_count(&self, word: &str) -> i64 {
        let word = py_strip_ws(word);
        let word = word.to_lowercase();
        let word = py_strip_chars(&word, PY_PUNCTUATION);
        if word.is_empty() {
            return 0;
        }
        if word.chars().any(py_is_decimal) {
            return 0;
        }
        if let Some(&count) = self.syllapy.get(word) {
            return count;
        }
        if let Some((head, tail)) = word.split_once('-')
            && !head.is_empty() && !tail.is_empty() {
                let first = self.syllable_count(head);
                let second = self.syllable_count(tail);
                if first == 0 || second == 0 {
                    return 0;
                }
                return first + second;
            }
        syllable_heuristic(word)
    }
}

/// syllapy._syllables.
fn syllable_heuristic(word: &str) -> i64 {
    let vowels = "aeiouy";
    let chars: Vec<char> = word.chars().collect();
    let mut count = 0i64;
    if !chars.is_empty() && vowels.contains(chars[0]) {
        count += 1;
    }
    for index in 1..chars.len() {
        if vowels.contains(chars[index]) && !vowels.contains(chars[index - 1]) {
            count += 1;
        }
    }
    if word.ends_with('e') {
        count -= 1;
    }
    if word.ends_with("le") && chars.len() > 2 && !vowels.contains(chars[chars.len() - 3]) {
        count += 1;
    }
    if count == 0 {
        count = 1;
    }
    count
}
