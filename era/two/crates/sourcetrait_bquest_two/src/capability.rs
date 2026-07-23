//! The CapabilityRatchet engine runner (EngineRunner): drive the lib
//! engine over rendered olmo-eval fixture requests and emit
//! predictions in their JSONL shape; scoring stays their code (the
//! CPU-only rescore step in the eval env consumes these rows beside
//! the fixture requests).
//!
//! Pinned olmo-eval contracts this file mirrors: the lm_eval-style
//! context/continuation encode (trailing whitespace moves to the
//! continuation; continuation ids are whole_enc[len(ctx_enc)..]; the
//! fed ids are ctx_enc ++ cont_ids - which may differ from whole_enc
//! across a BPE boundary, deliberately), and build_predictions'
//! model_output field set (the logprob-derived keys appear only on
//! loglikelihood rows; generation rows carry text/extracted_answer/
//! num_chars/is_greedy alone).
use crate::*;

/// The prefill chunk for context advances (the lib generate default).
const PREFILL_CHUNK: usize = 512;
/// The capability home relative to the XDG data home.
pub(crate) const CAPABILITY_HOME_RELATIVE: &str = "sourcetrait/quest/capability";

/// A fixture request row (the olmo-eval requests JSONL shape; fields
/// this runner does not consume are ignored by serde).
#[derive(Debug, serde::Deserialize)]
pub(crate) struct RequestRow {
    pub(crate) request_type: String,
    pub(crate) request: RequestBody,
    pub(crate) task_name: String,
    pub(crate) doc_id: i64,
    pub(crate) native_id: serde_json::Value,
    #[serde(default)]
    pub(crate) label: serde_json::Value,
}

#[derive(Debug, serde::Deserialize)]
pub(crate) struct RequestBody {
    /// A string (completion/loglikelihood) or a [{role, content}]
    /// message list (chat).
    #[serde(default)]
    pub(crate) context: serde_json::Value,
    #[serde(default)]
    pub(crate) continuation: Option<String>,
    #[serde(default)]
    pub(crate) continuations: Option<Vec<String>>,
    #[serde(default)]
    pub(crate) generation_kwargs: Option<GenerationKwargs>,
    #[serde(default)]
    pub(crate) stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, serde::Deserialize)]
pub(crate) struct GenerationKwargs {
    #[serde(default)]
    pub(crate) max_gen_toks: Option<usize>,
    #[serde(default)]
    pub(crate) do_sample: Option<bool>,
    #[serde(default)]
    pub(crate) temperature: Option<f64>,
}

/// One model_output entry in the olmo-eval predictions shape; the
/// Option fields serialize only when present (their builder omits the
/// logprob family on generation rows).
#[derive(Debug, serde::Serialize)]
pub(crate) struct ModelOutput {
    pub(crate) text: String,
    pub(crate) extracted_answer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) is_greedy: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sum_logits: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) num_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) num_tokens_all: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logits_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logits_per_char: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bits_per_byte: Option<f64>,
    pub(crate) num_chars: usize,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct PredictionRow {
    pub(crate) doc_id: i64,
    pub(crate) native_id: serde_json::Value,
    pub(crate) model_output: Vec<ModelOutput>,
    pub(crate) label: serde_json::Value,
    pub(crate) final_output: String,
}

/// XDG data home, honoring the spec fallback (~/.local/share).
pub(crate) fn data_home() -> BquestResult<PathBuf> {
    if let Ok(dir) = std::env::var("XDG_DATA_HOME")
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() => Ok(PathBuf::from(home).join(".local/share")),
        _ => snafu::whatever!("neither XDG_DATA_HOME nor HOME is set"),
    }
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> BquestResult<Vec<u32>> {
    match tokenizer.encode(text, false) {
        Ok(encoding) => Ok(encoding.get_ids().to_vec()),
        Err(e) => snafu::whatever!("encode failed: {e}"),
    }
}

fn ids_tensor(ids: &[u32], device: &candle_core::Device) -> BquestResult<candle_core::Tensor> {
    Ok(candle_core::Tensor::from_vec(ids.to_vec(), ids.len(), device)?)
}

/// Log-softmax picks on one host logits row: (logprob of `token`,
/// whether `token` is the row argmax).
fn row_pick(row: &[f32], token: u32) -> (f64, bool) {
    let mut max_value = f32::NEG_INFINITY;
    let mut max_index = 0usize;
    for (index, value) in row.iter().enumerate() {
        if *value > max_value {
            max_value = *value;
            max_index = index;
        }
    }
    let mut exp_sum = 0f64;
    for value in row {
        exp_sum += ((*value - max_value) as f64).exp();
    }
    let lse = max_value as f64 + exp_sum.ln();
    ((row[token as usize] as f64) - lse, max_index == token as usize)
}

pub(crate) fn capability_run(cli: &Cli, args: &CapabilityRunArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let config = lib::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let settings = lib::LibSettings::load_from_dir(cli.dir.as_ref(), cli.settings.as_deref())?;

    let capability_home = data_home()?.join(CAPABILITY_HOME_RELATIVE);
    let fixtures = match &args.fixtures {
        Some(dir) => dir.clone(),
        None => capability_home.join("fixtures/full"),
    };
    // Input-format detection, decided at source: a requests tree
    // carrying *-requests.nuon runs the nuon path; JSONL otherwise.
    let requests_root = fixtures.join("requests");
    snafu::ensure_whatever!(
        requests_root.is_dir(),
        "no requests directory under {}",
        fixtures.display()
    );
    let nuon_request_files = collect_suffix_files(&requests_root, "-requests.nuon")?;
    let nuon_mode = !nuon_request_files.is_empty();
    let settings_token = cli.settings.clone().unwrap_or_else(|| "default".to_string());
    let settings_snake: String = settings_token
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect();
    let out_root = match &args.out {
        Some(dir) => dir.clone(),
        None => {
            if nuon_mode {
                capability_home.join("nuon/runs").join(format!("engine_{settings_snake}"))
            } else {
                capability_home.join("runs").join(format!("engine_{settings_snake}"))
            }
        }
    };
    let task_filter: Option<Vec<String>> = args
        .tasks
        .as_ref()
        .map(|csv| csv.split(',').map(|t| t.trim().to_string()).collect());

    #[cfg(feature = "cuda")]
    let (device, dtype) = (candle_core::Device::new_cuda(0)?, candle_core::DType::BF16);
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let model_dir = config.model_dir();
    let checkpoint = lib::load_config(&model_dir)?;
    let tokenizer = lib::load_tokenizer(&model_dir)?;
    lib::verify_token_map(&tokenizer)?;
    let weights = lib::mmap_weights(&model_dir, dtype, &device)?;
    let mut model = lib::OlmoHybrid::new(&checkpoint, settings, weights)?;
    eprintln!(
        "capability run: model {} loaded ({:.1}s)",
        config.model,
        started.elapsed().as_secs_f64()
    );

    let mut tasks_run = 0usize;
    let mut items_run = 0usize;
    if nuon_mode {
        for file in &nuon_request_files {
            let file_name = file.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            let token = task_token(file_name, "-requests.nuon");
            if let Some(filter) = &task_filter
                && !filter.contains(&token)
            {
                continue;
            }
            let rows_value = lib::nu::load_value(file)?;
            let rows_list = match rows_value.as_list() {
                Ok(list) => list,
                Err(e) => snafu::whatever!("{}: not a table: {e}", file.display()),
            };
            let mut rows: Vec<RequestRow> = Vec::with_capacity(rows_list.len());
            for row in rows_list {
                let json = value_to_json(row)?;
                rows.push(serde_json::from_value(json)?);
            }
            let Some(first) = rows.first() else {
                continue;
            };
            let task_started = std::time::Instant::now();
            let predictions = match first.request_type.as_str() {
                "loglikelihood" => run_loglikelihood_task(&mut model, &tokenizer, &rows)?,
                "generate_until" => run_generation_task(&mut model, &tokenizer, &rows)?,
                other => {
                    snafu::whatever!("unsupported request_type {other:?} in {}", file.display())
                }
            };
            let Ok(relative) = file.strip_prefix(&requests_root) else {
                snafu::whatever!("request file escapes the requests root: {}", file.display());
            };
            let name = relative
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .replace("-requests.nuon", "-predictions.nuon");
            let parent = relative.parent().unwrap_or_else(|| Path::new(""));
            let dest = out_root.join("predictions").join(parent).join(name);
            let mut prediction_values = Vec::with_capacity(predictions.len());
            for row in &predictions {
                prediction_values.push(json_to_value(&serde_json::to_value(row)?)?);
            }
            lib::nu::save_value(&dest, &lib::nu::Value::list(prediction_values, span()))?;
            tasks_run += 1;
            items_run += predictions.len();
            eprintln!(
                "capability run: {token} - {} items in {:.1}s",
                predictions.len(),
                task_started.elapsed().as_secs_f64()
            );
        }
        snafu::ensure_whatever!(tasks_run > 0, "no tasks matched");
        let provenance = lib::nu::Value::record(
            lib::nu::record! {
                "fixtures" => v_str(&fixtures.display().to_string()),
                "model" => v_str(&config.model),
                "settings" => v_str(&settings_token),
                "bquest_version" => v_str(env!("CARGO_PKG_VERSION")),
                "generated_at" => v_int(epoch_seconds()),
                "tasks" => v_int(tasks_run as i64),
                "items" => v_int(items_run as i64),
            },
            span(),
        );
        lib::nu::save_value(&out_root.join("provenance.nuon"), &provenance)?;
        let summary = lib::nu::Value::record(
            lib::nu::record! {
                "tasks" => v_int(tasks_run as i64),
                "items" => v_int(items_run as i64),
                "seconds" => v_float(started.elapsed().as_secs_f64()),
                "out" => v_str(&out_root.display().to_string()),
            },
            span(),
        );
        println!("{}", lib::nu::to_nuon_text(&summary)?);
        return Ok(());
    }

    let mut request_files = Vec::new();
    collect_request_files(&requests_root, &mut request_files)?;
    request_files.sort();
    snafu::ensure_whatever!(
        !request_files.is_empty(),
        "no *-requests.jsonl found under {}",
        requests_root.display()
    );

    for file in &request_files {
        let rows = read_rows(file)?;
        let Some(first) = rows.first() else {
            continue;
        };
        if let Some(filter) = &task_filter
            && !filter.contains(&first.task_name)
        {
            continue;
        }
        let task_started = std::time::Instant::now();
        let predictions = match first.request_type.as_str() {
            "loglikelihood" => run_loglikelihood_task(&mut model, &tokenizer, &rows)?,
            "generate_until" => run_generation_task(&mut model, &tokenizer, &rows)?,
            other => snafu::whatever!("unsupported request_type {other:?} in {}", file.display()),
        };
        let out_path = prediction_path(&requests_root, file, &out_root)?;
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut payload = String::new();
        for row in &predictions {
            payload.push_str(&serde_json::to_string(row)?);
            payload.push('\n');
        }
        fs::write(&out_path, payload)?;
        tasks_run += 1;
        items_run += predictions.len();
        eprintln!(
            "capability run: {} - {} items in {:.1}s",
            first.task_name,
            predictions.len(),
            task_started.elapsed().as_secs_f64()
        );
    }

    println!(
        "{}",
        serde_json::json!({
            "tasks": tasks_run,
            "items": items_run,
            "seconds": started.elapsed().as_secs_f64(),
            "out": out_root,
        })
    );
    Ok(())
}

/// The reverse adapter: a nuon run's predictions rendered back to the
/// reference's JSONL tree (the live bridge for rescore.py
/// comparisons).
pub(crate) fn capability_bridge(args: &CapabilityBridgeArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let out_root = match &args.out {
        Some(dir) => dir.clone(),
        None => args.run.join("jsonl"),
    };
    let predictions_root = args.run.join("predictions");
    let files = collect_suffix_files(&predictions_root, "-predictions.nuon")?;
    snafu::ensure_whatever!(
        !files.is_empty(),
        "no *-predictions.nuon under {}",
        predictions_root.display()
    );
    let mut items = 0usize;
    for file in &files {
        let rows_value = lib::nu::load_value(file)?;
        let rows = match rows_value.as_list() {
            Ok(list) => list,
            Err(e) => snafu::whatever!("{}: not a table: {e}", file.display()),
        };
        let Ok(relative) = file.strip_prefix(&predictions_root) else {
            snafu::whatever!("prediction file escapes the run: {}", file.display());
        };
        let name = relative
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .replace("-predictions.nuon", "-predictions.jsonl");
        let parent = relative.parent().unwrap_or_else(|| Path::new(""));
        let dest = out_root.join("predictions").join(parent).join(name);
        if let Some(dest_parent) = dest.parent() {
            fs::create_dir_all(dest_parent)?;
        }
        let mut payload = String::new();
        for row in rows {
            payload.push_str(&serde_json::to_string(&value_to_json(row)?)?);
            payload.push('\n');
        }
        fs::write(&dest, payload)?;
        items += rows.len();
    }
    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "files" => v_int(files.len() as i64),
            "items" => v_int(items as i64),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
            "out" => v_str(&out_root.display().to_string()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// Recursively collect *-requests.jsonl files.
fn collect_request_files(dir: &Path, into: &mut Vec<PathBuf>) -> BquestResult<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_request_files(&path, into)?;
        } else if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("-requests.jsonl"))
        {
            into.push(path);
        }
    }
    Ok(())
}

fn read_rows(file: &Path) -> BquestResult<Vec<RequestRow>> {
    let text = fs::read_to_string(file)?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(line)?);
    }
    Ok(rows)
}

/// The mirrored output path: requests/<sub>/<stem>-requests.jsonl ->
/// <out>/predictions/<sub>/<stem>-predictions.jsonl.
fn prediction_path(
    requests_root: &Path,
    file: &Path,
    out_root: &Path,
) -> BquestResult<PathBuf> {
    let Ok(relative) = file.strip_prefix(requests_root) else {
        snafu::whatever!("request file escapes the requests root: {}", file.display());
    };
    let name = relative
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .replace("-requests.jsonl", "-predictions.jsonl");
    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    Ok(out_root.join("predictions").join(parent).join(name))
}

/// MC/loglikelihood scoring: the shared context prefills once (all but
/// its final token), then each continuation forwards
/// [last_ctx ++ cont_ids] and rolls back - row j of the block predicts
/// cont_ids[j] (the all-position logits contract).
fn run_loglikelihood_task(
    model: &mut lib::OlmoHybrid,
    tokenizer: &tokenizers::Tokenizer,
    rows: &[RequestRow],
) -> BquestResult<Vec<PredictionRow>> {
    let device = model.device().clone();
    let mut predictions = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(context) = row.request.context.as_str() else {
            snafu::whatever!(
                "loglikelihood context must be a string (task {}, doc {})",
                row.task_name,
                row.doc_id
            );
        };
        let continuations: Vec<String> = match (&row.request.continuations, &row.request.continuation) {
            (Some(list), _) if !list.is_empty() => list.clone(),
            (_, Some(single)) => vec![single.clone()],
            _ => snafu::whatever!(
                "loglikelihood row carries no continuations (task {}, doc {})",
                row.task_name,
                row.doc_id
            ),
        };

        // The lm_eval boundary: trailing whitespace moves from the
        // context to every continuation before tokenization.
        let trimmed = context.trim_end();
        let moved = &context[trimmed.len()..];
        let ctx_enc = encode(tokenizer, trimmed)?;
        snafu::ensure_whatever!(
            !ctx_enc.is_empty(),
            "empty context after encode (task {}, doc {})",
            row.task_name,
            row.doc_id
        );

        model.clear_cache()?;
        let last_ctx = *ctx_enc.last().expect("non-empty");
        let prefix = &ctx_enc[..ctx_enc.len() - 1];
        let mut start = 0usize;
        while start < prefix.len() {
            let len = PREFILL_CHUNK.min(prefix.len() - start);
            model.forward_chunk_carry(&ids_tensor(&prefix[start..start + len], &device)?)?;
            start += len;
        }
        let mark = model.mark_context()?;

        let mut outputs = Vec::with_capacity(continuations.len());
        for (index, continuation) in continuations.iter().enumerate() {
            if index > 0 {
                model.rollback_context(&mark)?;
            }
            let cont_full = format!("{moved}{continuation}");
            let whole_enc = encode(tokenizer, &format!("{trimmed}{cont_full}"))?;
            snafu::ensure_whatever!(
                whole_enc.len() > ctx_enc.len(),
                "continuation encodes to zero tokens (task {}, doc {})",
                row.task_name,
                row.doc_id
            );
            let cont_ids = &whole_enc[ctx_enc.len()..];
            let mut block = Vec::with_capacity(1 + cont_ids.len());
            block.push(last_ctx);
            block.extend_from_slice(cont_ids);
            let logits = model.forward_chunk(&ids_tensor(&block, &device)?)?;
            let host = logits
                .narrow(0, 0, cont_ids.len())?
                .to_dtype(candle_core::DType::F32)?
                .to_device(&candle_core::Device::Cpu)?
                .to_vec2::<f32>()?;
            let mut sum_logits = 0f64;
            let mut is_greedy = true;
            for (position, token) in cont_ids.iter().enumerate() {
                let (logprob, greedy) = row_pick(&host[position], *token);
                sum_logits += logprob;
                is_greedy = is_greedy && greedy;
            }
            let num_chars = continuation.chars().count();
            let num_tokens = cont_ids.len();
            let logits_per_char = if num_chars == 0 { 0.0 } else { sum_logits / num_chars as f64 };
            outputs.push(ModelOutput {
                text: continuation.clone(),
                extracted_answer: continuation.clone(),
                is_greedy: Some(is_greedy),
                sum_logits: Some(sum_logits),
                num_tokens: Some(num_tokens),
                num_tokens_all: Some(num_tokens),
                logits_per_token: Some(sum_logits / num_tokens as f64),
                logits_per_char: Some(logits_per_char),
                bits_per_byte: Some(-logits_per_char / std::f64::consts::LN_2),
                num_chars,
            });
        }
        let final_output = outputs.first().map(|o| o.text.clone()).unwrap_or_default();
        predictions.push(PredictionRow {
            doc_id: row.doc_id,
            native_id: row.native_id.clone(),
            model_output: outputs,
            label: row.label.clone(),
            final_output,
        });
    }
    Ok(predictions)
}

/// generate_until rows: greedy decode through the carried engine at
/// the row's cap, with their client-side stop-sequence truncation
/// (cut at the earliest stop occurrence). Chat contexts render
/// through the byte-identical chat template; string contexts feed
/// verbatim.
fn run_generation_task(
    model: &mut lib::OlmoHybrid,
    tokenizer: &tokenizers::Tokenizer,
    rows: &[RequestRow],
) -> BquestResult<Vec<PredictionRow>> {
    let mut predictions = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(kwargs) = &row.request.generation_kwargs else {
            snafu::whatever!(
                "generate_until row carries no generation_kwargs (task {}, doc {})",
                row.task_name,
                row.doc_id
            );
        };
        let sampled = kwargs.do_sample.unwrap_or(false)
            && kwargs.temperature.is_some_and(|t| t > 0.0);
        snafu::ensure_whatever!(
            !sampled,
            "the capability runner is greedy-only (task {}, doc {})",
            row.task_name,
            row.doc_id
        );
        let Some(max_gen_toks) = kwargs.max_gen_toks else {
            snafu::whatever!(
                "generate_until row carries no max_gen_toks (task {}, doc {})",
                row.task_name,
                row.doc_id
            );
        };

        let rendered = match &row.request.context {
            serde_json::Value::String(text) => text.clone(),
            serde_json::Value::Array(messages) => {
                snafu::ensure_whatever!(
                    messages.len() == 1,
                    "chat contexts are single-user-message in v1 (task {}, doc {}, {} messages)",
                    row.task_name,
                    row.doc_id,
                    messages.len()
                );
                let message = &messages[0];
                let role = message["role"].as_str().unwrap_or_default();
                snafu::ensure_whatever!(
                    role == "user",
                    "chat contexts are single-user-message in v1 (task {}, doc {}, role {role:?})",
                    row.task_name,
                    row.doc_id
                );
                lib::chat_wrap(message["content"].as_str().unwrap_or_default())
            }
            other => snafu::whatever!(
                "unsupported context shape {other:?} (task {}, doc {})",
                row.task_name,
                row.doc_id
            ),
        };

        let options = lib::GenerateOptions {
            temperature: None,
            top_p: None,
            sample_len: max_gen_toks,
            chat: false,
            ignore_stops: false,
            speculate: false,
            ..lib::GenerateOptions::default()
        };
        let mut generation = model.generate(tokenizer, &rendered, &options)?;
        let mut text = String::new();
        for step in generation.by_ref() {
            text.push_str(&step?.chunk);
        }
        let report = generation.finish();
        text.push_str(&report.rest);

        let stops = row.request.stop_sequences.clone().unwrap_or_default();
        let cut = stops.iter().filter_map(|stop| text.find(stop.as_str())).min();
        let kept = match cut {
            Some(position) => text[..position].to_string(),
            None => text,
        };
        let num_chars = kept.chars().count();
        let outputs = vec![ModelOutput {
            text: kept.clone(),
            extracted_answer: kept.clone(),
            is_greedy: Some(true),
            sum_logits: None,
            num_tokens: None,
            num_tokens_all: None,
            logits_per_token: None,
            logits_per_char: None,
            bits_per_byte: None,
            num_chars,
        }];
        predictions.push(PredictionRow {
            doc_id: row.doc_id,
            native_id: row.native_id.clone(),
            model_output: outputs,
            label: row.label.clone(),
            final_output: kept,
        });
    }
    Ok(predictions)
}
