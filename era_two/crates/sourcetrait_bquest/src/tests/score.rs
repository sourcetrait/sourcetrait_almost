//! ScoreRust locks. Checkpoint-free: the treebank tokenizer doctest,
//! the gsm8k extraction regex semantics, python string helpers.
//! Box-data gates (#[ignore]): punkt against nltk's own doctest, the
//! tagger against the pos_tag doctest, and GATE 3 - rust scores vs
//! the standing rescore outputs, per-item and aggregate, exact f64
//! equality, swept over every standing run.
use crate::*;

use crate::checker_data::CheckerData;
use crate::punkt::PunktParams;
use crate::punkt::sent_tokenize;
use crate::score::clean_short_answer;
use crate::score::extract_last_number;
use crate::score::find_fixture_file;
use crate::score::load_sorted_rows;
use crate::score::resolve_task_kind;
use crate::score::score_task;
use crate::wordtok::treebank_tokenize;

#[test]
fn treebank_matches_the_nltk_doctest() {
    let text = "Good muffins cost $3.88 (roughly 3,36 euros)\nin New York.  Please buy me\ntwo of them.\nThanks.";
    let tokens = treebank_tokenize(text);
    let expected = [
        "Good", "muffins", "cost", "$", "3.88", "(", "roughly", "3,36", "euros", ")",
        "in", "New", "York.", "Please", "buy", "me", "two", "of", "them.", "Thanks", ".",
    ];
    assert_eq!(tokens, expected);
}

#[test]
fn gsm8k_extraction_matches_the_reference_regex() {
    assert_eq!(extract_last_number("So the answer is 2280."), Some(String::from("2280")));
    assert_eq!(
        extract_last_number("costs $1,234,567 total"),
        Some(String::from("1234567"))
    );
    assert_eq!(extract_last_number("pi is 3.14 roughly"), Some(String::from("3.14")));
    assert_eq!(extract_last_number("minus -5 wins"), Some(String::from("-5")));
    assert_eq!(extract_last_number("no digits here"), None);
    assert_eq!(clean_short_answer("2,280"), "2280");
    assert_eq!(clean_short_answer("none"), "none");
}

#[test]
fn python_string_helpers_hold() {
    assert!(py_str_isupper("ABC1"));
    assert!(!py_str_isupper("123"));
    assert!(py_str_islower("abc-def"));
    assert_eq!(py_split_ws("  a\tb\nc  "), ["a", "b", "c"]);
    assert_eq!(py_strip_chars("..word!!", PY_PUNCTUATION), "word");
    assert_eq!(py_count("ababab", "ab"), 3);
    assert_eq!(py_word_run_count("two words, three"), 3);
    assert!(py_boundary_search("Emma and Liam", "Emma", false));
    assert!(!py_boundary_search("Emmaline", "Emma", false));
    assert_eq!(nfkd_ascii("São Tomé"), "Sao Tome");
}

fn capability_home() -> PathBuf {
    data_home().expect("data home").join(CAPABILITY_HOME_RELATIVE)
}

#[test]
#[ignore]
fn punkt_matches_the_nltk_doctest() {
    let params = PunktParams::load(&capability_home().join("checker_data/punkt_tab_english"))
        .expect("punkt params load");
    let text = "Punkt knows that the periods in Mr. Smith and Johann S. Bach\ndo not mark sentence boundaries.  And sometimes sentences\ncan start with non-capitalized words.  i is a good variable\nname.";
    let sentences = sent_tokenize(&params, text);
    assert_eq!(
        sentences,
        [
            "Punkt knows that the periods in Mr. Smith and Johann S. Bach\ndo not mark sentence boundaries.",
            "And sometimes sentences\ncan start with non-capitalized words.",
            "i is a good variable\nname.",
        ]
    );
    let quoted = "(How does it deal with this parenthesis?)  \"It should be part of the\nprevious sentence.\" \"(And the same with this one.)\" ('And this one!')\n\"('(And (this)) '?)\" [(and this. )]";
    let sentences = sent_tokenize(&params, quoted);
    assert_eq!(
        sentences,
        [
            "(How does it deal with this parenthesis?)",
            "\"It should be part of the\nprevious sentence.\"",
            "\"(And the same with this one.)\"",
            "('And this one!')",
            "\"('(And (this)) '?)\"",
            "[(and this. )]",
        ]
    );
}

#[test]
#[ignore]
fn tagger_matches_the_nltk_doctest_first_tokens() {
    let data = CheckerData::load(&capability_home().join("checker_data"))
        .expect("checker data load");
    let sentence: Vec<String> = "The quick brown fox jumps over the lazy dog"
        .split(' ')
        .map(str::to_string)
        .collect();
    assert_eq!(data.tagger.tag_first(&sentence).expect("tags"), "DT");
    let red: Vec<String> = "The red cat".split(' ').map(str::to_string).collect();
    assert_eq!(data.tagger.tag_first(&red).expect("tags"), "DT");
}

/// GATE 3: per-item and aggregate exact equality against the standing
/// rescore outputs, over every standing run.
#[test]
#[ignore]
fn gate3_rust_scores_match_the_standing_rescore_outputs() {
    let home = capability_home();
    let checker_data = CheckerData::load(&home.join("checker_data")).expect("checker data");
    let fixtures_root = home.join("nuon/fixtures");
    let runs = ["engine_default", "engine_evict_v1", "engine_evict_v1_graph", "controlleg"];

    let mut mismatches: Vec<String> = Vec::new();
    let mut tasks_checked = 0usize;
    let mut items_checked = 0usize;

    for run in runs {
        let nuon_run = home.join("nuon/runs").join(run);
        let reference_run = home.join("runs").join(run);
        let metrics_text = fs::read_to_string(reference_run.join("metrics_rescored.json"))
            .expect("metrics_rescored.json");
        let metrics_json: serde_json::Value =
            serde_json::from_str(&metrics_text).expect("metrics json");

        // Their keys are raw task specs; sanitize(spec) aligns them
        // with the file-derived task tokens.
        let mut reference_metrics: HashMap<String, &serde_json::Value> = HashMap::new();
        for (spec, value) in metrics_json.as_object().expect("metrics object") {
            let sanitized: String = spec
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }
                })
                .collect();
            reference_metrics.insert(sanitized, value);
        }

        let prediction_files =
            collect_suffix_files(&nuon_run.join("predictions"), "-predictions.nuon")
                .expect("prediction files");
        assert!(!prediction_files.is_empty(), "{run}: no converted predictions");

        for file in &prediction_files {
            let file_name = file.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            let token = task_token(file_name, "-predictions.nuon");
            let kind = resolve_task_kind(&token).expect("task kind");
            let fixture_rows = load_sorted_rows(
                &find_fixture_file(&fixtures_root, &token).expect("fixture file"),
            )
            .expect("fixture rows");
            let prediction_rows = load_sorted_rows(file).expect("prediction rows");
            let fixture_records: Vec<&lib::nu::Record> =
                fixture_rows.iter().map(|row| row.as_record().expect("record")).collect();
            let prediction_records: Vec<&lib::nu::Record> = prediction_rows
                .iter()
                .map(|row| row.as_record().expect("record"))
                .collect();
            let scores = match score_task(
                Some(&checker_data),
                kind,
                &fixture_records,
                &prediction_records,
            ) {
                Ok(scores) => scores,
                Err(error) => {
                    mismatches.push(format!("{run}/{token}: scoring failed: {error}"));
                    continue;
                }
            };
            tasks_checked += 1;
            items_checked += scores.num_instances;

            // Per-item comparison against their scores JSONL.
            let reference_path =
                reference_run.join("scores").join(format!("{token}-scores.jsonl"));
            let reference_text =
                fs::read_to_string(&reference_path).expect("reference scores");
            let reference_rows: Vec<serde_json::Value> = reference_text
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| serde_json::from_str(line).expect("score row json"))
                .collect();
            if reference_rows.len() != scores.rows.len() {
                mismatches.push(format!(
                    "{run}/{token}: {} reference rows vs {} rust rows",
                    reference_rows.len(),
                    scores.rows.len()
                ));
                continue;
            }
            for (mine, theirs) in scores.rows.iter().zip(reference_rows.iter()) {
                let their_doc_id = theirs["doc_id"].as_i64().expect("doc_id");
                if their_doc_id != mine.doc_id {
                    mismatches
                        .push(format!("{run}/{token}: doc_id order {their_doc_id}"));
                    continue;
                }
                let their_scores = theirs["scores"].as_object();
                let their_scores_len = their_scores.map(|m| m.len()).unwrap_or(0);
                if their_scores_len != mine.scores.len() {
                    mismatches.push(format!(
                        "{run}/{token} doc {}: scores key sets differ",
                        mine.doc_id
                    ));
                    continue;
                }
                for (name, value) in &mine.scores {
                    let theirs_value = their_scores
                        .and_then(|m| m.get(name))
                        .and_then(|v| v.as_f64());
                    if theirs_value.map(f64::to_bits) != Some(value.to_bits()) {
                        mismatches.push(format!(
                            "{run}/{token} doc {} score {name}: rust {value:?} vs \
                             reference {theirs_value:?}",
                            mine.doc_id
                        ));
                    }
                }
                let their_extracted = theirs["extracted"].as_array().expect("extracted");
                if their_extracted.len() != mine.extracted.len() {
                    mismatches.push(format!(
                        "{run}/{token} doc {}: extracted lengths differ",
                        mine.doc_id
                    ));
                    continue;
                }
                for (index, (mine_entry, their_entry)) in
                    mine.extracted.iter().zip(their_extracted.iter()).enumerate()
                {
                    let matches = match (mine_entry, their_entry) {
                        (None, serde_json::Value::Null) => true,
                        (Some(text), serde_json::Value::String(reference)) => {
                            text == reference
                        }
                        _ => false,
                    };
                    if !matches {
                        mismatches.push(format!(
                            "{run}/{token} doc {} extracted[{index}] differs",
                            mine.doc_id
                        ));
                    }
                }
            }

            // Aggregate comparison against metrics_rescored.json.
            let Some(reference_task) = reference_metrics.get(&token) else {
                mismatches.push(format!("{run}/{token}: absent from metrics_rescored"));
                continue;
            };
            let reference_instances =
                reference_task["num_instances"].as_u64().unwrap_or(0) as usize;
            if reference_instances != scores.num_instances {
                mismatches.push(format!(
                    "{run}/{token}: num_instances {} vs reference {reference_instances}",
                    scores.num_instances
                ));
            }
            let reference_pairs: usize = reference_task["metrics"]
                .as_object()
                .map(|metrics| {
                    metrics
                        .values()
                        .map(|scorers| scorers.as_object().map(|s| s.len()).unwrap_or(0))
                        .sum()
                })
                .unwrap_or(0);
            if reference_pairs != scores.metrics.len() {
                mismatches
                    .push(format!("{run}/{token}: metric pair counts differ"));
            }
            for (metric, scorer, value) in &scores.metrics {
                let reference_value =
                    reference_task["metrics"][metric][scorer].as_f64();
                if reference_value.map(f64::to_bits) != Some(value.to_bits()) {
                    mismatches.push(format!(
                        "{run}/{token} {metric}.{scorer}: rust {value:?} vs reference \
                         {reference_value:?}"
                    ));
                }
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "GATE 3 FAILED with {} mismatches over {tasks_checked} tasks / {items_checked} \
         items:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    assert_eq!(tasks_checked, 240, "expected 60 tasks x 4 runs");
    println!("GATE 3: {tasks_checked} tasks, {items_checked} items, zero mismatches");
}
