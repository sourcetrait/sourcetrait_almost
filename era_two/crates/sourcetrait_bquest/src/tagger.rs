//! nltk's averaged perceptron POS tagger (perceptron.py, nltk
//! 3.10.0), inference for the first token only - all the ifeval port
//! needs (StartWithVerbChecker reads pos_tag(tokens)[0]).
//!
//! Faithfulness: the tagdict shortcut keys on the RAW word; features
//! accumulate in insertion order; weight maps iterate in JSON file
//! order (serde_json preserve_order); prediction takes the maximum of
//! (score, label) over the class list exactly as Python's max does.
use crate::*;

pub(crate) struct PerceptronTagger {
    weights: serde_json::Map<String, serde_json::Value>,
    tagdict: HashMap<String, String>,
    classes: Vec<String>,
}

impl PerceptronTagger {
    pub(crate) fn load(dir: &Path) -> BquestResult<Self> {
        let weights_text =
            fs::read_to_string(dir.join("averaged_perceptron_tagger_eng.weights.json"))?;
        let weights: serde_json::Value = serde_json::from_str(&weights_text)?;
        let Some(weights) = weights.as_object().cloned() else {
            snafu::whatever!("tagger weights.json is not an object");
        };
        let tagdict_text =
            fs::read_to_string(dir.join("averaged_perceptron_tagger_eng.tagdict.json"))?;
        let tagdict: HashMap<String, String> = serde_json::from_str(&tagdict_text)?;
        let classes_text =
            fs::read_to_string(dir.join("averaged_perceptron_tagger_eng.classes.json"))?;
        let classes: Vec<String> = serde_json::from_str(&classes_text)?;
        Ok(Self { weights, tagdict, classes })
    }

    /// PerceptronTagger.normalize.
    fn normalize(word: &str) -> String {
        let first = word.chars().next();
        if word.contains('-') && first != Some('-') {
            return String::from("!HYPHEN");
        }
        if py_str_isdigit(word) && word.chars().count() == 4 {
            return String::from("!YEAR");
        }
        if first.is_some_and(py_is_decimal) {
            return String::from("!DIGITS");
        }
        word.to_lowercase()
    }

    /// word[-3:] - the last three codepoints.
    fn suffix3(word: &str) -> String {
        let chars: Vec<char> = word.chars().collect();
        let start = chars.len().saturating_sub(3);
        chars[start..].iter().collect()
    }

    /// The tag of tokens[0] under PerceptronTagger.tag (prev/prev2 are
    /// the START sentinels for the first token, so later predictions
    /// cannot affect it).
    pub(crate) fn tag_first(&self, tokens: &[String]) -> BquestResult<String> {
        let Some(word) = tokens.first() else {
            snafu::whatever!("tag_first on an empty token list");
        };
        if let Some(tag) = self.tagdict.get(word) {
            return Ok(tag.clone());
        }

        let mut context: Vec<String> = vec![String::from("-START-"), String::from("-START2-")];
        context.extend(tokens.iter().map(|token| Self::normalize(token)));
        context.push(String::from("-END-"));
        context.push(String::from("-END2-"));

        // _get_features(i=0, ...) with i offset by len(START) = 2.
        let i = 2usize;
        let first_char = word.chars().next().map(String::from).unwrap_or_default();
        let mut features: Vec<(String, i64)> = Vec::new();
        let add = |features: &mut Vec<(String, i64)>, name: String| {
            if let Some(entry) = features.iter_mut().find(|(key, _)| *key == name) {
                entry.1 += 1;
            } else {
                features.push((name, 1));
            }
        };
        add(&mut features, String::from("bias"));
        add(&mut features, format!("i suffix {}", Self::suffix3(word)));
        add(&mut features, format!("i pref1 {first_char}"));
        add(&mut features, String::from("i-1 tag -START-"));
        add(&mut features, String::from("i-2 tag -START2-"));
        add(&mut features, String::from("i tag+i-2 tag -START- -START2-"));
        add(&mut features, format!("i word {}", context[i]));
        add(&mut features, format!("i-1 tag+i word -START- {}", context[i]));
        add(&mut features, format!("i-1 word {}", context[i - 1]));
        add(&mut features, format!("i-1 suffix {}", Self::suffix3(&context[i - 1])));
        add(&mut features, format!("i-2 word {}", context[i - 2]));
        add(&mut features, format!("i+1 word {}", context[i + 1]));
        add(&mut features, format!("i+1 suffix {}", Self::suffix3(&context[i + 1])));
        add(&mut features, format!("i+2 word {}", context[i + 2]));

        let mut scores: HashMap<&str, f64> = HashMap::new();
        for (feature, value) in &features {
            if *value == 0 {
                continue;
            }
            let Some(label_weights) = self.weights.get(feature).and_then(|w| w.as_object())
            else {
                continue;
            };
            for (label, weight) in label_weights {
                let Some(weight) = weight.as_f64() else {
                    snafu::whatever!("non-numeric weight for feature {feature:?}");
                };
                *scores.entry(label.as_str()).or_insert(0.0) += *value as f64 * weight;
            }
        }

        // max(classes, key=lambda label: (scores[label], label)).
        let mut best: Option<(&str, f64)> = None;
        for label in &self.classes {
            let score = *scores.get(label.as_str()).unwrap_or(&0.0);
            let better = match &best {
                None => true,
                Some((best_label, best_score)) => {
                    score > *best_score
                        || (score == *best_score && label.as_str() > *best_label)
                }
            };
            if better {
                best = Some((label.as_str(), score));
            }
        }
        match best {
            Some((label, _)) => Ok(label.to_string()),
            None => snafu::whatever!("tagger has no classes"),
        }
    }
}
