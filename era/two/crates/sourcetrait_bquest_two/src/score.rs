//! `bquest capability score`: pure-rust, CPU, nuon-in/nuon-out
//! scoring of a converted run - the faithful port of olmo-eval's
//! scorers for the capability task set (MC logprob accuracy, gsm8k
//! exact-match with last-number extraction, IFBench OOD ifeval).
//!
//! Task routing mirrors the reference registry: mmlu_* and arc_* are
//! logprob MC (mmlu extracts nothing, arc extracts the continuation
//! text), gsm8k is exact-match, ifeval_ood is the instruction
//! verifier suite; any other task errors loudly rather than
//! mis-scoring.
use crate::*;

use std::sync::LazyLock;

use crate::checker_data::CheckerData;
use crate::ifeval::KwargValue;
use crate::ifeval::Kwargs;
use crate::ifeval::score_ifeval_item;

static COMMA_IN_NUMBER: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(\p{Nd}),(\p{Nd})").expect("compiles"));
static NUMBER: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"[-+]?\p{Nd}*\.\p{Nd}+|[-+]?\p{Nd}+").expect("compiles")
});

/// gsm8k _extract_last_number.
pub(crate) fn extract_last_number(text: &str) -> Option<String> {
    let without_commas = COMMA_IN_NUMBER.replace_all(text, "$1$2");
    NUMBER
        .find_iter(&without_commas)
        .last()
        .map(|found| found.as_str().to_string())
}

/// gsm8k _clean_short_answer (falls back to the input text).
pub(crate) fn clean_short_answer(text: &str) -> String {
    extract_last_number(text).unwrap_or_else(|| text.to_string())
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum TaskKind {
    McMmlu,
    McArc,
    Gsm8k,
    IfevalOod,
}

pub(crate) fn resolve_task_kind(token: &str) -> BquestResult<TaskKind> {
    if token.starts_with("mmlu_") {
        return Ok(TaskKind::McMmlu);
    }
    if token.starts_with("arc_") {
        return Ok(TaskKind::McArc);
    }
    match token {
        "gsm8k" => Ok(TaskKind::Gsm8k),
        "ifeval_ood" => Ok(TaskKind::IfevalOod),
        other => snafu::whatever!("no scorer registered for task {other:?}"),
    }
}

/// One instruction's strict and loose verdict, keyed by its IFBench
/// instruction id. The four headline ratios are already built from
/// these; retaining them is what gives a care-weighted re-aggregation
/// per-TYPE results to attach to, where the collapsed per-item score
/// offers nothing.
pub(crate) struct InstructionVerdict {
    pub(crate) id: String,
    pub(crate) strict: bool,
    pub(crate) loose: bool,
}

pub(crate) struct ScoreRow {
    pub(crate) doc_id: i64,
    pub(crate) scores: Vec<(String, f64)>,
    pub(crate) extracted: Vec<Option<String>>,
    /// Per-instruction detail, empty for tasks that have none. STRICTLY
    /// ADDITIVE: no value in `scores` or `extracted` moves, so the
    /// reference-parity gate - which compares those two alone, by key
    /// set and by length - is untouched. The fork constraint holds
    /// because only aggregation may differ from the reference, never
    /// the per-item layer.
    pub(crate) instructions: Vec<InstructionVerdict>,
}

pub(crate) struct TaskScores {
    /// metric name -> (scorer name, value), in the reference's
    /// config.metrics order.
    pub(crate) metrics: Vec<(String, String, f64)>,
    pub(crate) num_instances: usize,
    pub(crate) rows: Vec<ScoreRow>,
}

fn field_f64(record: &lib::nu::Record, name: &str) -> BquestResult<f64> {
    let value = field(record, name)?;
    if let Ok(float) = value.as_float() {
        return Ok(float);
    }
    if let Ok(int) = value.as_int() {
        return Ok(int as f64);
    }
    snafu::whatever!("field {name} is not numeric")
}

fn output_text(output: &lib::nu::Record) -> BquestResult<String> {
    field_str(output, "text")
}

/// Python max(list) / list.index(max): the value and its first index.
fn first_max(values: &[f64]) -> (usize, f64) {
    let mut best_index = 0usize;
    let mut best = values[0];
    for (index, &value) in values.iter().enumerate().skip(1) {
        if value > best {
            best = value;
            best_index = index;
        }
    }
    (best_index, best)
}

fn score_mc(
    kind: TaskKind,
    fixture_rows: &[&lib::nu::Record],
    prediction_rows: &[&lib::nu::Record],
) -> BquestResult<TaskScores> {
    let mut rows = Vec::with_capacity(prediction_rows.len());
    let mut correct = 0usize;
    for (fixture, prediction) in fixture_rows.iter().zip(prediction_rows.iter()) {
        let doc = field(fixture, "doc")?.as_record()?;
        let gold_idx = field_int(doc, "gold_idx")?;
        let outputs = field_rows(prediction, "model_output")?;
        snafu::ensure_whatever!(!outputs.is_empty(), "MC row without outputs");
        let mut sums = Vec::with_capacity(outputs.len());
        for output in &outputs {
            sums.push(field_f64(output, "sum_logits")?);
        }
        let (argmax, max_sum) = first_max(&sums);
        if argmax as i64 == gold_idx {
            correct += 1;
        }
        let extracted = match kind {
            TaskKind::McMmlu => outputs.iter().map(|_| None).collect(),
            _ => outputs
                .iter()
                .map(|output| output_text(output).map(Some))
                .collect::<BquestResult<Vec<_>>>()?,
        };
        rows.push(ScoreRow {
            doc_id: field_int(prediction, "doc_id")?,
            scores: vec![(String::from("logprob"), max_sum)],
            extracted,
            instructions: Vec::new(),
        });
    }
    let accuracy = correct as f64 / prediction_rows.len() as f64;
    Ok(TaskScores {
        metrics: vec![(String::from("accuracy"), String::from("logprob"), accuracy)],
        num_instances: prediction_rows.len(),
        rows,
    })
}

fn score_gsm8k(
    fixture_rows: &[&lib::nu::Record],
    prediction_rows: &[&lib::nu::Record],
) -> BquestResult<TaskScores> {
    let mut rows = Vec::with_capacity(prediction_rows.len());
    let mut total = 0f64;
    for (fixture, prediction) in fixture_rows.iter().zip(prediction_rows.iter()) {
        let doc = field(fixture, "doc")?.as_record()?;
        let gold = clean_short_answer(&field_str(doc, "short_answer")?);
        let outputs = field_rows(prediction, "model_output")?;
        snafu::ensure_whatever!(!outputs.is_empty(), "gsm8k row without outputs");
        let extracted = extract_last_number(&output_text(outputs[0])?);
        let score = match &extracted {
            Some(answer) => {
                let gold_cmp = py_strip_ws(&gold).to_lowercase();
                let answer_cmp = py_strip_ws(answer).to_lowercase();
                if gold_cmp == answer_cmp { 1.0 } else { 0.0 }
            }
            None => 0.0,
        };
        total += score;
        rows.push(ScoreRow {
            doc_id: field_int(prediction, "doc_id")?,
            scores: vec![(String::from("exact_match"), score)],
            extracted: vec![extracted],
            instructions: Vec::new(),
        });
    }
    let accuracy = total / prediction_rows.len() as f64;
    Ok(TaskScores {
        metrics: vec![(String::from("accuracy"), String::from("exact_match"), accuracy)],
        num_instances: prediction_rows.len(),
        rows,
    })
}

fn kwargs_from_record(record: &lib::nu::Record) -> BquestResult<Kwargs> {
    let mut kwargs = Kwargs::new();
    for (name, value) in record.iter() {
        let converted = match value {
            lib::nu::Value::Nothing { .. } => continue,
            lib::nu::Value::Int { val, .. } => KwargValue::Int(*val),
            lib::nu::Value::Float { val, .. } => KwargValue::Float(*val),
            lib::nu::Value::String { val, .. } => KwargValue::Text(val.clone()),
            other => snafu::whatever!(
                "unsupported kwarg type for {name:?}: {}",
                other.get_type()
            ),
        };
        kwargs.insert(name.clone(), converted);
    }
    Ok(kwargs)
}

fn score_ifeval_task(
    data: &CheckerData,
    fixture_rows: &[&lib::nu::Record],
    prediction_rows: &[&lib::nu::Record],
) -> BquestResult<TaskScores> {
    let mut rows = Vec::with_capacity(prediction_rows.len());
    let mut strict_lists: Vec<Vec<bool>> = Vec::with_capacity(prediction_rows.len());
    let mut loose_lists: Vec<Vec<bool>> = Vec::with_capacity(prediction_rows.len());
    for (fixture, prediction) in fixture_rows.iter().zip(prediction_rows.iter()) {
        let doc = field(fixture, "doc")?.as_record()?;
        let prompt = field_str(doc, "prompt")?;
        let instruction_ids = field_strings(doc, "instruction_id_list")?;
        let kwargs_rows = field_rows(doc, "kwargs")?;
        let mut kwargs_list = Vec::with_capacity(kwargs_rows.len());
        for kwargs_row in kwargs_rows {
            kwargs_list.push(kwargs_from_record(kwargs_row)?);
        }
        let outputs = field_rows(prediction, "model_output")?;
        snafu::ensure_whatever!(!outputs.is_empty(), "ifeval row without outputs");
        let response = output_text(outputs[0])?;
        let (strict, loose) =
            score_ifeval_item(data, &instruction_ids, &kwargs_list, &prompt, &response)?;
        let score = if loose.is_empty() {
            0.0
        } else if loose.iter().all(|&passed| passed) {
            1.0
        } else {
            0.0
        };
        let instructions: Vec<InstructionVerdict> = instruction_ids
            .iter()
            .zip(strict.iter())
            .zip(loose.iter())
            .map(|((id, strict), loose)| InstructionVerdict {
                id: id.clone(),
                strict: *strict,
                loose: *loose,
            })
            .collect();
        rows.push(ScoreRow {
            doc_id: field_int(prediction, "doc_id")?,
            scores: vec![(String::from("ifeval"), score)],
            extracted: vec![Some(response)],
            instructions,
        });
        strict_lists.push(strict);
        loose_lists.push(loose);
    }

    let prompt_level = |lists: &[Vec<bool>]| -> f64 {
        if lists.is_empty() {
            return 0.0;
        }
        let correct = lists
            .iter()
            .filter(|list| !list.is_empty() && list.iter().all(|&passed| passed))
            .count();
        correct as f64 / lists.len() as f64
    };
    let instruction_level = |lists: &[Vec<bool>]| -> f64 {
        let total: usize = lists.iter().map(Vec::len).sum();
        if total == 0 {
            return 0.0;
        }
        let correct: usize =
            lists.iter().map(|list| list.iter().filter(|&&passed| passed).count()).sum();
        correct as f64 / total as f64
    };

    Ok(TaskScores {
        metrics: vec![
            (
                String::from("prompt_level_strict_acc"),
                String::from("ifeval"),
                prompt_level(&strict_lists),
            ),
            (
                String::from("prompt_level_loose_acc"),
                String::from("ifeval"),
                prompt_level(&loose_lists),
            ),
            (
                String::from("inst_level_strict_acc"),
                String::from("ifeval"),
                instruction_level(&strict_lists),
            ),
            (
                String::from("inst_level_loose_acc"),
                String::from("ifeval"),
                instruction_level(&loose_lists),
            ),
        ],
        num_instances: prediction_rows.len(),
        rows,
    })
}

/// Score one task's prediction rows against its fixture rows. Rows
/// must arrive doc_id-sorted; doc_id == position is enforced exactly
/// as the reference rescore does.
pub(crate) fn score_task(
    data: Option<&CheckerData>,
    kind: TaskKind,
    fixture_rows: &[&lib::nu::Record],
    prediction_rows: &[&lib::nu::Record],
) -> BquestResult<TaskScores> {
    snafu::ensure_whatever!(
        fixture_rows.len() == prediction_rows.len(),
        "{} fixture rows vs {} prediction rows",
        fixture_rows.len(),
        prediction_rows.len()
    );
    snafu::ensure_whatever!(!prediction_rows.is_empty(), "no rows to score");
    for (position, (fixture, prediction)) in
        fixture_rows.iter().zip(prediction_rows.iter()).enumerate()
    {
        let fixture_doc_id = field_int(fixture, "doc_id")?;
        let prediction_doc_id = field_int(prediction, "doc_id")?;
        snafu::ensure_whatever!(
            fixture_doc_id == position as i64 && prediction_doc_id == position as i64,
            "doc_id misalignment at position {position}: fixture {fixture_doc_id}, \
             prediction {prediction_doc_id}"
        );
    }
    match kind {
        TaskKind::McMmlu | TaskKind::McArc => score_mc(kind, fixture_rows, prediction_rows),
        TaskKind::Gsm8k => score_gsm8k(fixture_rows, prediction_rows),
        TaskKind::IfevalOod => {
            let Some(data) = data else {
                snafu::whatever!("ifeval scoring requires checker data");
            };
            score_ifeval_task(data, fixture_rows, prediction_rows)
        }
    }
}

/// Sorted rows of a whole-value nuon table file, by doc_id.
pub(crate) fn load_sorted_rows(path: &Path) -> BquestResult<Vec<lib::nu::Value>> {
    let value = lib::nu::load_value(path)?;
    let list = match value.as_list() {
        Ok(list) => list.to_vec(),
        Err(e) => snafu::whatever!("{}: not a table: {e}", path.display()),
    };
    let mut rows = list;
    let mut keyed: Vec<(i64, lib::nu::Value)> = Vec::with_capacity(rows.len());
    for row in rows.drain(..) {
        let record = match row.as_record() {
            Ok(record) => record,
            Err(e) => snafu::whatever!("{}: row is not a record: {e}", path.display()),
        };
        let doc_id = field_int(record, "doc_id")?;
        keyed.push((doc_id, row));
    }
    keyed.sort_by_key(|(doc_id, _)| *doc_id);
    Ok(keyed.into_iter().map(|(_, row)| row).collect())
}

fn rows_as_records(rows: &[lib::nu::Value]) -> BquestResult<Vec<&lib::nu::Record>> {
    rows.iter()
        .map(|row| match row.as_record() {
            Ok(record) => Ok(record),
            Err(e) => snafu::whatever!("row is not a record: {e}"),
        })
        .collect()
}

/// Locate the fixtures requests nuon file for a task token.
pub(crate) fn find_fixture_file(
    fixtures_root: &Path,
    token: &str,
) -> BquestResult<PathBuf> {
    let requests_root = fixtures_root.join("requests");
    let candidates: Vec<PathBuf> = collect_suffix_files(&requests_root, "-requests.nuon")?
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| task_token(name, "-requests.nuon") == token)
        })
        .collect();
    match candidates.len() {
        1 => Ok(candidates.into_iter().next().expect("one candidate")),
        0 => snafu::whatever!("no fixtures for task {token:?} under {}", requests_root.display()),
        _ => snafu::whatever!("ambiguous fixtures for task {token:?}"),
    }
}

fn score_row_value(row: &ScoreRow) -> lib::nu::Value {
    let mut scores = lib::nu::Record::new();
    for (name, value) in &row.scores {
        scores.push(name.clone(), v_float(*value));
    }
    let extracted: Vec<lib::nu::Value> = row
        .extracted
        .iter()
        .map(|entry| match entry {
            Some(text) => v_str(text),
            None => lib::nu::Value::nothing(span()),
        })
        .collect();
    let mut record = lib::nu::record! {
        "doc_id" => v_int(row.doc_id),
        "scores" => lib::nu::Value::record(scores, span()),
        "extracted" => lib::nu::Value::list(extracted, span()),
    };
    // Present only where the task carries instructions, so the 1,263
    // multiple-choice and exact-match rows stay exactly as they were.
    // Records are open, so a consumer tests presence.
    if !row.instructions.is_empty() {
        let verdicts: Vec<lib::nu::Value> = row
            .instructions
            .iter()
            .map(|verdict| {
                lib::nu::Value::record(
                    lib::nu::record! {
                        "id" => v_str(&verdict.id),
                        "strict" => v_bool(verdict.strict),
                        "loose" => v_bool(verdict.loose),
                    },
                    span(),
                )
            })
            .collect();
        record.push("instructions", lib::nu::Value::list(verdicts, span()));
    }
    lib::nu::Value::record(record, span())
}

fn aggregate_value(scores: &TaskScores) -> lib::nu::Value {
    let mut metrics = lib::nu::Record::new();
    for (metric, scorer, value) in &scores.metrics {
        let mut inner = lib::nu::Record::new();
        inner.push(scorer.clone(), v_float(*value));
        metrics.push(metric.clone(), lib::nu::Value::record(inner, span()));
    }
    lib::nu::Value::record(
        lib::nu::record! {
            "metrics" => lib::nu::Value::record(metrics, span()),
            "num_instances" => v_int(scores.num_instances as i64),
        },
        span(),
    )
}

pub(crate) fn capability_score(args: &CapabilityScoreArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let capability_home = data_home()?.join(CAPABILITY_HOME_RELATIVE);
    let run_dir = match &args.run {
        Some(dir) => dir.clone(),
        None => capability_home.join("nuon/runs/engine_default"),
    };
    let fixtures_root = match &args.fixtures {
        Some(dir) => dir.clone(),
        None => capability_home.join("nuon/fixtures"),
    };
    let out_root = match &args.out {
        Some(dir) => dir.clone(),
        None => run_dir.clone(),
    };
    let checker_dir = match &args.checker_data {
        Some(dir) => dir.clone(),
        None => capability_home.join("checker_data"),
    };
    let task_filter: Option<Vec<String>> = args
        .tasks
        .as_ref()
        .map(|csv| csv.split(',').map(|t| t.trim().to_string()).collect());

    let predictions_root = run_dir.join("predictions");
    let prediction_files = collect_suffix_files(&predictions_root, "-predictions.nuon")?;
    snafu::ensure_whatever!(
        !prediction_files.is_empty(),
        "no *-predictions.nuon under {}",
        predictions_root.display()
    );

    let mut checker_data: Option<CheckerData> = None;
    let mut aggregates = lib::nu::Record::new();
    let mut tasks_scored = 0usize;
    let mut items_scored = 0usize;
    for file in &prediction_files {
        let file_name = file.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let token = task_token(file_name, "-predictions.nuon");
        if let Some(filter) = &task_filter
            && !filter.contains(&token) {
                continue;
            }
        let kind = resolve_task_kind(&token)?;
        if kind == TaskKind::IfevalOod && checker_data.is_none() {
            checker_data = Some(CheckerData::load(&checker_dir)?);
        }
        let fixture_rows = load_sorted_rows(&find_fixture_file(&fixtures_root, &token)?)?;
        let prediction_rows = load_sorted_rows(file)?;
        let scores = score_task(
            checker_data.as_ref(),
            kind,
            &rows_as_records(&fixture_rows)?,
            &rows_as_records(&prediction_rows)?,
        )?;
        let rows: Vec<lib::nu::Value> = scores.rows.iter().map(score_row_value).collect();
        let scores_path = out_root.join("scores").join(format!("{token}-scores.nuon"));
        lib::nu::save_value(&scores_path, &lib::nu::Value::list(rows, span()))?;
        aggregates.push(token.clone(), aggregate_value(&scores));
        tasks_scored += 1;
        items_scored += scores.num_instances;
        eprintln!("capability score: {token} - {} items", scores.num_instances);
    }
    snafu::ensure_whatever!(tasks_scored > 0, "no tasks matched");

    let aggregates_path = out_root.join("aggregates.nuon");
    lib::nu::save_value(&aggregates_path, &lib::nu::Value::record(aggregates, span()))?;

    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "tasks" => v_int(tasks_scored as i64),
            "items" => v_int(items_scored as i64),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
            "out" => v_str(&out_root.display().to_string()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}
