//! The lossless JSONL and nu-value bridge, and `capability convert`.
use crate::*;

/// Row-kind typedefs, harvested from the standing trees.
const REQUEST_LOGLIKELIHOOD_TYPEDEF: &str = "record<request_type: string, \
     doc: record<query: string, gold_idx: int, choices: list<string>>, \
     request: record<context: string, continuations: list<string>>, \
     idx: int, task_name: string, doc_id: int, \
     native_id: oneof<int, string>, label: int>";

const REQUEST_GENERATE_TYPEDEF: &str = "record<request_type: string, \
     doc: record<query: string>, \
     request: record<context: oneof<string, table<role: string, content: string>>, \
     generation_kwargs: record<max_gen_toks: int>, stop_sequences: list<string>>, \
     idx: int, task_name: string, doc_id: int, \
     native_id: oneof<int, string>, label: oneof<string, nothing>>";

const PREDICTION_TYPEDEF: &str = "record<doc_id: int, \
     native_id: oneof<int, string>, \
     model_output: table<text: string, extracted_answer: oneof<string, nothing>, \
     is_greedy: bool, num_chars: int>, \
     label: oneof<int, string, nothing>, final_output: string>";

pub(crate) fn span() -> lib::nu::Span {
    lib::nu::Span::unknown()
}

pub(crate) fn v_int(value: i64) -> lib::nu::Value {
    lib::nu::Value::int(value, span())
}

pub(crate) fn v_float(value: f64) -> lib::nu::Value {
    lib::nu::Value::float(value, span())
}

pub(crate) fn v_str(value: &str) -> lib::nu::Value {
    lib::nu::Value::string(value, span())
}

pub(crate) fn v_bool(value: bool) -> lib::nu::Value {
    lib::nu::Value::bool(value, span())
}

pub(crate) fn v_int_list(ids: &[u32]) -> lib::nu::Value {
    lib::nu::Value::list(ids.iter().map(|id| v_int(*id as i64)).collect(), span())
}

pub(crate) fn field<'a>(
    record: &'a lib::nu::Record,
    name: &str,
) -> BquestResult<&'a lib::nu::Value> {
    match record.get(name) {
        Some(value) => Ok(value),
        None => snafu::whatever!("missing field {name}"),
    }
}

pub(crate) fn field_str(record: &lib::nu::Record, name: &str) -> BquestResult<String> {
    match field(record, name)?.as_str() {
        Ok(text) => Ok(text.to_string()),
        Err(e) => snafu::whatever!("field {name}: {e}"),
    }
}

pub(crate) fn field_int(record: &lib::nu::Record, name: &str) -> BquestResult<i64> {
    match field(record, name)?.as_int() {
        Ok(value) => Ok(value),
        Err(e) => snafu::whatever!("field {name}: {e}"),
    }
}

pub(crate) fn field_rows<'a>(
    record: &'a lib::nu::Record,
    name: &str,
) -> BquestResult<Vec<&'a lib::nu::Record>> {
    let value = field(record, name)?;
    let list = match value.as_list() {
        Ok(list) => list,
        Err(e) => snafu::whatever!("field {name}: {e}"),
    };
    let mut rows = Vec::with_capacity(list.len());
    for item in list {
        match item.as_record() {
            Ok(row) => rows.push(row),
            Err(e) => snafu::whatever!("field {name} row: {e}"),
        }
    }
    Ok(rows)
}

pub(crate) fn field_ids(record: &lib::nu::Record, name: &str) -> BquestResult<Vec<u32>> {
    let value = field(record, name)?;
    let list = match value.as_list() {
        Ok(list) => list,
        Err(e) => snafu::whatever!("field {name}: {e}"),
    };
    let mut ids = Vec::with_capacity(list.len());
    for item in list {
        match item.as_int() {
            Ok(id) => ids.push(id as u32),
            Err(e) => snafu::whatever!("field {name} id: {e}"),
        }
    }
    Ok(ids)
}

pub(crate) fn field_strings(
    record: &lib::nu::Record,
    name: &str,
) -> BquestResult<Vec<String>> {
    let value = field(record, name)?;
    let list = match value.as_list() {
        Ok(list) => list,
        Err(e) => snafu::whatever!("field {name}: {e}"),
    };
    let mut strings = Vec::with_capacity(list.len());
    for item in list {
        match item.as_str() {
            Ok(text) => strings.push(text.to_string()),
            Err(e) => snafu::whatever!("field {name} entry: {e}"),
        }
    }
    Ok(strings)
}

/// nu Value to JSON, the exact inverse of json_to_value.
pub(crate) fn value_to_json(value: &lib::nu::Value) -> BquestResult<serde_json::Value> {
    Ok(match value {
        lib::nu::Value::Nothing { .. } => serde_json::Value::Null,
        lib::nu::Value::Bool { val, .. } => serde_json::Value::Bool(*val),
        lib::nu::Value::Int { val, .. } => serde_json::Value::Number((*val).into()),
        lib::nu::Value::Float { val, .. } => match serde_json::Number::from_f64(*val) {
            Some(number) => serde_json::Value::Number(number),
            None => snafu::whatever!("non-finite float cannot render to JSON"),
        },
        lib::nu::Value::String { val, .. } => serde_json::Value::String(val.clone()),
        lib::nu::Value::List { vals, .. } => {
            let mut items = Vec::with_capacity(vals.len());
            for item in vals {
                items.push(value_to_json(item)?);
            }
            serde_json::Value::Array(items)
        }
        lib::nu::Value::Record { val, .. } => {
            let mut map = serde_json::Map::new();
            for (key, item) in val.iter() {
                map.insert(key.clone(), value_to_json(item)?);
            }
            serde_json::Value::Object(map)
        }
        other => snafu::whatever!("unsupported value type for JSON: {}", other.get_type()),
    })
}

/// JSON to nu Value, lossless field for field.
pub(crate) fn json_to_value(json: &serde_json::Value) -> BquestResult<lib::nu::Value> {
    Ok(match json {
        serde_json::Value::Null => lib::nu::Value::nothing(span()),
        serde_json::Value::Bool(flag) => v_bool(*flag),
        serde_json::Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                v_int(int)
            } else if number.is_u64() {
                snafu::whatever!("integer {number} exceeds i64; lossless conversion refused")
            } else if let Some(float) = number.as_f64() {
                v_float(float)
            } else {
                snafu::whatever!("unrepresentable JSON number {number}")
            }
        }
        serde_json::Value::String(text) => v_str(text),
        serde_json::Value::Array(items) => {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(json_to_value(item)?);
            }
            lib::nu::Value::list(values, span())
        }
        serde_json::Value::Object(map) => {
            let mut record = lib::nu::Record::new();
            for (key, value) in map {
                record.push(key.clone(), json_to_value(value)?);
            }
            lib::nu::Value::record(record, span())
        }
    })
}

/// Recursively collect files whose names end with `suffix`, sorted.
pub(crate) fn collect_suffix_files(
    dir: &Path,
    suffix: &str,
) -> BquestResult<Vec<PathBuf>> {
    fn walk(dir: &Path, suffix: &str, into: &mut Vec<PathBuf>) -> BquestResult<()> {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(&path, suffix, into)?;
            } else if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(suffix))
            {
                into.push(path);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, suffix, &mut files)?;
    files.sort();
    Ok(files)
}

/// A source file's task token: the stem without suffix or render hash.
pub(crate) fn task_token(file_name: &str, suffix: &str) -> String {
    let stem = file_name.strip_suffix(suffix).unwrap_or(file_name);
    if let Some(position) = stem.rfind('_') {
        let tail = &stem[position + 1..];
        if tail.len() == 6 && tail.chars().all(|c| c.is_ascii_hexdigit()) {
            return stem[..position].to_string();
        }
    }
    stem.to_string()
}

/// Whether a task token passes the --tasks filter.
fn task_selected(filter: &Option<Vec<String>>, token: &str) -> bool {
    match filter {
        Some(tokens) => tokens.iter().any(|t| t == token),
        None => true,
    }
}

fn sha256_hex(path: &Path) -> BquestResult<String> {
    use sha2::Digest;
    let bytes = fs::read(path)?;
    let digest = sha2::Sha256::digest(&bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    Ok(hex)
}

pub(crate) fn epoch_seconds() -> i64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(elapsed) => elapsed.as_secs() as i64,
        Err(_) => 0,
    }
}

/// The typedef a request row must conform to, by its request_type.
fn request_typedef(request_type: &str) -> BquestResult<lib::nu::Type> {
    match request_type {
        "loglikelihood" => Ok(lib::nu::parse_typedef(REQUEST_LOGLIKELIHOOD_TYPEDEF)?),
        "generate_until" => Ok(lib::nu::parse_typedef(REQUEST_GENERATE_TYPEDEF)?),
        other => snafu::whatever!("unsupported request_type {other:?}"),
    }
}

enum SourceKind {
    Requests,
    Predictions,
}

/// Convert one JSONL file, conform-validated and gate-verified.
fn convert_file(
    source: &Path,
    dest: &Path,
    kind: &SourceKind,
    task: &str,
    run: Option<&str>,
) -> BquestResult<usize> {
    let text = fs::read_to_string(source)?;
    let mut rows: Vec<lib::nu::Value> = Vec::new();
    let mut request_type_cache: Option<lib::nu::Type> = None;
    let prediction_typedef = match kind {
        SourceKind::Predictions => Some(lib::nu::parse_typedef(PREDICTION_TYPEDEF)?),
        SourceKind::Requests => None,
    };
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let json: serde_json::Value = match serde_json::from_str(line) {
            Ok(json) => json,
            Err(e) => snafu::whatever!("{}:{}: JSON parse failed: {e}", source.display(), index + 1),
        };
        let value = match json_to_value(&json) {
            Ok(value) => value,
            Err(e) => snafu::whatever!("{}:{}: {e}", source.display(), index + 1),
        };
        let declared = match kind {
            SourceKind::Requests => {
                if request_type_cache.is_none() {
                    let request_type = json["request_type"].as_str().unwrap_or_default();
                    request_type_cache = Some(request_typedef(request_type)?);
                }
                request_type_cache.as_ref().expect("cached")
            }
            SourceKind::Predictions => prediction_typedef.as_ref().expect("built"),
        };
        if let Err(e) = lib::nu::conform(&value, declared) {
            snafu::whatever!("{}:{}: {e}", source.display(), index + 1);
        }
        rows.push(value);
    }
    snafu::ensure_whatever!(!rows.is_empty(), "{}: no rows", source.display());

    lib::nu::save_value(dest, &lib::nu::Value::list(rows.clone(), span()))?;

    let reloaded = lib::nu::load_value(dest)?;
    let reloaded_rows = match reloaded.as_list() {
        Ok(list) => list,
        Err(e) => snafu::whatever!("{}: reloaded value is not a table: {e}", dest.display()),
    };
    snafu::ensure_whatever!(
        reloaded_rows.len() == rows.len(),
        "{}: lossless gate FAILED: {} source rows vs {} nuon rows",
        dest.display(),
        rows.len(),
        reloaded_rows.len()
    );
    let mismatches: Vec<usize> = rows
        .iter()
        .zip(reloaded_rows.iter())
        .enumerate()
        .filter(|(_, (source_row, nuon_row))| source_row != nuon_row)
        .map(|(index, _)| index)
        .collect();
    snafu::ensure_whatever!(
        mismatches.is_empty(),
        "{}: lossless gate FAILED on {} of {} rows (first mismatched row indices: {:?})",
        dest.display(),
        mismatches.len(),
        rows.len(),
        mismatches.iter().take(8).collect::<Vec<_>>()
    );

    let mut provenance = lib::nu::record! {
        "source" => v_str(&source.display().to_string()),
        "task" => v_str(task),
        "items" => v_int(rows.len() as i64),
        "sha256" => v_str(&sha256_hex(source)?),
        "run" => match run {
            Some(name) => v_str(name),
            None => lib::nu::Value::nothing(span()),
        },
    };
    provenance.push("bquest_version", v_str(env!("CARGO_PKG_VERSION")));
    provenance.push("converted_at", v_int(epoch_seconds()));
    let provenance_path = dest.with_extension("provenance.nuon");
    lib::nu::save_value(&provenance_path, &lib::nu::Value::record(provenance, span()))?;

    Ok(rows.len())
}

pub(crate) fn capability_convert(args: &CapabilityConvertArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let capability_home = data_home()?.join(CAPABILITY_HOME_RELATIVE);
    let fixtures = match &args.fixtures {
        Some(dir) => dir.clone(),
        None => capability_home.join("fixtures/full"),
    };
    let runs_root = match &args.runs {
        Some(dir) => dir.clone(),
        None => capability_home.join("runs"),
    };
    let out_root = match &args.out {
        Some(dir) => dir.clone(),
        None => capability_home.join("nuon"),
    };
    let task_filter: Option<Vec<String>> = args
        .tasks
        .as_ref()
        .map(|csv| csv.split(',').map(|t| t.trim().to_string()).collect());

    let requests_root = fixtures.join("requests");
    let request_files = collect_suffix_files(&requests_root, "-requests.jsonl")?;
    snafu::ensure_whatever!(
        !request_files.is_empty(),
        "no *-requests.jsonl found under {}",
        requests_root.display()
    );
    let mut fixture_files = 0usize;
    let mut fixture_items = 0usize;
    for file in &request_files {
        let file_name = file.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let task = task_token(file_name, "-requests.jsonl");
        if !task_selected(&task_filter, &task) {
            continue;
        }
        let Ok(relative) = file.strip_prefix(&requests_root) else {
            snafu::whatever!("request file escapes the requests root: {}", file.display());
        };
        let dest = out_root
            .join("fixtures/requests")
            .join(relative)
            .with_extension("")
            .with_extension("nuon");
        let items = convert_file(file, &dest, &SourceKind::Requests, &task, None)?;
        fixture_files += 1;
        fixture_items += items;
        eprintln!("capability convert: fixtures {task} - {items} items");
    }

    let mut run_names: Vec<String> = Vec::new();
    let mut prediction_files = 0usize;
    let mut prediction_items = 0usize;
    if runs_root.is_dir() {
        let mut run_dirs: Vec<PathBuf> = fs::read_dir(&runs_root)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.is_dir() && path.join("predictions").is_dir())
            .collect();
        run_dirs.sort();
        for run_dir in &run_dirs {
            let run_name = run_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            let predictions_root = run_dir.join("predictions");
            let files = collect_suffix_files(&predictions_root, "-predictions.jsonl")?;
            let mut run_items = 0usize;
            for file in &files {
                let file_name =
                    file.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                let task = task_token(file_name, "-predictions.jsonl");
                if !task_selected(&task_filter, &task) {
                    continue;
                }
                let Ok(relative) = file.strip_prefix(&predictions_root) else {
                    snafu::whatever!(
                        "prediction file escapes the predictions root: {}",
                        file.display()
                    );
                };
                let dest = out_root
                    .join("runs")
                    .join(&run_name)
                    .join("predictions")
                    .join(relative)
                    .with_extension("")
                    .with_extension("nuon");
                let items =
                    convert_file(file, &dest, &SourceKind::Predictions, &task, Some(&run_name))?;
                prediction_files += 1;
                run_items += items;
            }
            prediction_items += run_items;
            if run_items > 0 {
                eprintln!("capability convert: run {run_name} - {run_items} items");
                run_names.push(run_name);
            }
        }
    }

    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "fixture_files" => v_int(fixture_files as i64),
            "fixture_items" => v_int(fixture_items as i64),
            "runs" => lib::nu::Value::list(
                run_names.iter().map(|name| v_str(name)).collect(),
                span(),
            ),
            "prediction_files" => v_int(prediction_files as i64),
            "prediction_items" => v_int(prediction_items as i64),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
            "out" => v_str(&out_root.display().to_string()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}
