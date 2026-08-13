//! Their-side conversion: pool chat rows into quest-native examples.
use crate::*;

/// The extracted pool rows shape (`--rows`).
pub(crate) const POOL_ROWS_TYPEDEF: &str = "table<id: string, source: string, \
     messages: table<role: string, content: string>>";

/// The non-pipe added-token family; a row carrying one is refused.
const REFUSED_MARKERS: [&str; 4] =
    ["<functions>", "</functions>", "<function_calls>", "</function_calls>"];

/// Common English function words, chosen to miss es/de/fr spellings.
const ENGLISH_HINTS: [&str; 20] = [
    "the", "and", "of", "to", "is", "that", "it", "for", "you", "was", "with", "are",
    "this", "have", "not", "be", "as", "on", "they", "at",
];

/// The function-word share below which a row does not read as English.
const ENGLISH_FLOOR: f64 = 0.12;

/// What one conversion pass counted, refusals split by cause.
#[derive(Debug, Default)]
pub(crate) struct TheirsTally {
    pub(crate) rows_available: usize,
    pub(crate) rows_taken: usize,
    pub(crate) supervised_tokens: usize,
    pub(crate) refused_markers: usize,
    pub(crate) refused_language: usize,
    pub(crate) refused_code: usize,
    pub(crate) refused_no_assistant: usize,
    pub(crate) refused_role: usize,
    pub(crate) refused_empty: usize,
    pub(crate) stripped_system: usize,
    pub(crate) escaped_rows: usize,
    pub(crate) too_long: usize,
}

/// The knobs one conversion pass runs under.
pub(crate) struct TheirsOptions {
    pub(crate) budget_tokens: usize,
    pub(crate) seq_len: usize,
    pub(crate) seed: u64,
    pub(crate) english: bool,
    pub(crate) no_code_fences: bool,
}

/// One pool row, read off its record.
pub(crate) struct PoolRow {
    pub(crate) id: String,
    pub(crate) source: String,
    pub(crate) messages: Vec<(String, String)>,
}

/// Why one row did not convert; the tally splits on it.
enum Refusal {
    Markers,
    Language,
    Code,
    NoAssistant,
    Role,
    Empty,
}

/// One converted row: the authored messages plus its measured encoding.
struct Converted {
    messages: Vec<(String, String)>,
    stripped_system: usize,
    escaped: bool,
}

/// Whether text reads as English by function-word share.
fn reads_english(text: &str) -> bool {
    let lowered = text.to_lowercase();
    let words: Vec<&str> = lowered
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    if words.is_empty() {
        return false;
    }
    let hits = words
        .iter()
        .filter(|word| ENGLISH_HINTS.contains(&**word))
        .count();
    hits as f64 / words.len() as f64 >= ENGLISH_FLOOR
}

/// An assistant turn's prose, wrapped as an authored md output block.
fn wrap_md(content: &str) -> String {
    use harness::channel::Tag;
    format!("{} md\n{content}\n{}", Tag::Output.name(), Tag::Close.name())
}

/// Prove a wrapped turn parses on the wire as one md output block.
///
/// The gate is validate-equals-serve: a failure here is a converter
/// defect, never a row to skip, so it raises rather than counting.
fn codec_gate(id: &str, wrapped: &str) -> BquestResult<()> {
    let wire = harness::channel::authoring_to_wire(wrapped);
    let blocks =
        match harness::channel::parse_blocks(&wire, harness::channel::Aliasing::Strict) {
            Ok(blocks) => blocks,
            Err(error) => snafu::whatever!("row {id}: converted turn does not parse: {error}"),
        };
    snafu::ensure_whatever!(
        blocks.len() == 1 && blocks[0].tag == harness::channel::Tag::Output,
        "row {id}: converted turn parsed to {} blocks, expected one output",
        blocks.len()
    );
    let descriptor = harness::channel::Descriptor::parse(&blocks[0].header)?;
    snafu::ensure_whatever!(
        descriptor == harness::channel::Descriptor::md(),
        "row {id}: converted turn's descriptor is {:?}, expected md",
        blocks[0].header
    );
    Ok(())
}

/// Convert one row's messages, or say why the row is refused.
fn convert_messages(
    id: &str,
    messages: &[(String, String)],
    options: &TheirsOptions,
) -> BquestResult<Result<Converted, Refusal>> {
    if options.english {
        let whole: String = messages
            .iter()
            .map(|(_, content)| content.as_str())
            .collect::<Vec<&str>>()
            .join(" ");
        if !reads_english(&whole) {
            return Ok(Err(Refusal::Language));
        }
    }
    // A fenced block marks a computer-language-subject row - the
    // substance-honesty exclusion's mechanical proxy for mixed classes.
    if options.no_code_fences
        && messages.iter().any(|(_, content)| content.contains("```"))
    {
        return Ok(Err(Refusal::Code));
    }
    let mut converted: Vec<(String, String)> = Vec::with_capacity(messages.len());
    let mut stripped_system = 0usize;
    let mut escaped = false;
    let mut carries_assistant = false;
    for (role, content) in messages {
        // The quest posture owns the system turn: a foreign one is
        // stripped so the encode path's QUEST_SYSTEM stands alone.
        if role == "system" {
            stripped_system += 1;
            continue;
        }
        if role != "user" && role != "assistant" {
            return Ok(Err(Refusal::Role));
        }
        if REFUSED_MARKERS.iter().any(|marker| content.contains(marker)) {
            return Ok(Err(Refusal::Markers));
        }
        let cleaned = harness::channel::escape_payload_markers(content.trim());
        if cleaned != content.trim() {
            escaped = true;
        }
        if cleaned.is_empty() {
            return Ok(Err(Refusal::Empty));
        }
        if role == "assistant" {
            carries_assistant = true;
            let wrapped = wrap_md(&cleaned);
            codec_gate(id, &wrapped)?;
            converted.push((role.clone(), wrapped));
        } else {
            converted.push((role.clone(), cleaned));
        }
    }
    if !carries_assistant {
        return Ok(Err(Refusal::NoAssistant));
    }
    Ok(Ok(Converted {
        messages: converted,
        stripped_system,
        escaped,
    }))
}

/// The authored messages as the chat messages the pack will encode.
fn wire_messages(messages: &[(String, String)]) -> BquestResult<Vec<llm::ChatMessage>> {
    let mut wire = Vec::with_capacity(messages.len());
    for (role, content) in messages {
        let role = llm::ChatRole::parse(role)?;
        wire.push(llm::ChatMessage::new(
            role,
            &harness::channel::authoring_to_wire(content),
        ));
    }
    Ok(wire)
}

pub(crate) fn read_pool_rows(path: &Path) -> BquestResult<Vec<PoolRow>> {
    let value = harness::nu::load_value(path)?;
    harness::nu::conform(&value, &harness::nu::parse_typedef(POOL_ROWS_TYPEDEF)?)?;
    let rows = match value.as_list() {
        Ok(rows) => rows,
        Err(e) => snafu::whatever!("{}: not a table: {e}", path.display()),
    };
    let mut pool = Vec::with_capacity(rows.len());
    for row in rows {
        let record = match row.as_record() {
            Ok(record) => record,
            Err(e) => snafu::whatever!("pool row: {e}"),
        };
        let mut messages = Vec::new();
        for message in field_rows(record, "messages")? {
            messages.push((field_str(message, "role")?, field_str(message, "content")?));
        }
        pool.push(PoolRow {
            id: field_str(record, "id")?,
            source: field_str(record, "source")?,
            messages,
        });
    }
    Ok(pool)
}

/// One converted example row: provenance beside the messages.
fn example_row(row: &PoolRow, converted: &Converted) -> harness::nu::Value {
    let messages: Vec<harness::nu::Value> = converted
        .messages
        .iter()
        .map(|(role, content)| {
            harness::nu::Value::record(
                harness::nu::record! {
                    "role" => v_str(role),
                    "content" => v_str(content),
                },
                span(),
            )
        })
        .collect();
    harness::nu::Value::record(
        harness::nu::record! {
            "id" => v_str(&row.id),
            "source" => v_str(&row.source),
            "messages" => harness::nu::Value::list(messages, span()),
        },
        span(),
    )
}

/// Convert a seeded draw of pool rows up to the supervised budget.
pub(crate) fn convert_rows(
    tokenizer: &tokenizers::Tokenizer,
    pool: Vec<PoolRow>,
    options: &TheirsOptions,
) -> BquestResult<(Vec<harness::nu::Value>, TheirsTally)> {
    let mut tally = TheirsTally {
        rows_available: pool.len(),
        ..TheirsTally::default()
    };
    let mut pool = pool;
    let mut rng = SplitMix64::new(options.seed);
    for index in (1..pool.len()).rev() {
        let swap_with = rng.next_below(index + 1);
        pool.swap(index, swap_with);
    }

    let width = options.seq_len + 1;
    let mut taken: Vec<harness::nu::Value> = Vec::new();
    for row in &pool {
        if tally.supervised_tokens >= options.budget_tokens {
            break;
        }
        let converted = match convert_messages(&row.id, &row.messages, options)? {
            Ok(converted) => converted,
            Err(refusal) => {
                match refusal {
                    Refusal::Markers => tally.refused_markers += 1,
                    Refusal::Language => tally.refused_language += 1,
                    Refusal::Code => tally.refused_code += 1,
                    Refusal::NoAssistant => tally.refused_no_assistant += 1,
                    Refusal::Role => tally.refused_role += 1,
                    Refusal::Empty => tally.refused_empty += 1,
                }
                continue;
            }
        };
        let (ids, mask) = example::encode_supervised(
            tokenizer,
            &wire_messages(&converted.messages)?,
        )?;
        if ids.len() > width {
            tally.too_long += 1;
            continue;
        }
        let supervised = mask.iter().filter(|slot| **slot != 0).count();
        snafu::ensure_whatever!(
            supervised > 0,
            "row {}: converted example supervises no position",
            row.id
        );
        tally.supervised_tokens += supervised;
        tally.stripped_system += converted.stripped_system;
        if converted.escaped {
            tally.escaped_rows += 1;
        }
        taken.push(example_row(row, &converted));
        tally.rows_taken += 1;
    }
    Ok((taken, tally))
}

/// `bquest mix theirs`: one class's pool rows to one example part.
pub(crate) fn mix_theirs(cli: &Cli, args: &MixTheirsArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let config = llm::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let tokenizer = llm::load_tokenizer(&config.model_dir())?;
    llm::verify_token_map(&tokenizer)?;
    let pool = read_pool_rows(&args.rows)?;
    let options = TheirsOptions {
        budget_tokens: args.budget_tokens,
        seq_len: args.seq_len,
        seed: args.seed,
        english: args.english,
        no_code_fences: args.no_code_fences,
    };
    let (taken, tally) = convert_rows(&tokenizer, pool, &options)?;
    snafu::ensure_whatever!(
        !taken.is_empty(),
        "{}: no rows survived conversion",
        args.rows.display()
    );
    let table = harness::nu::Value::list(taken, span());
    harness::nu::conform(&table, &harness::nu::parse_typedef(example::SFT_TYPEDEF)?)?;
    harness::nu::save_value(&args.out, &table)?;

    let tally_record = |tally: &TheirsTally| {
        harness::nu::record! {
            "rows_available" => v_int(tally.rows_available as i64),
            "rows_taken" => v_int(tally.rows_taken as i64),
            "supervised_tokens" => v_int(tally.supervised_tokens as i64),
            "budget_tokens" => v_int(args.budget_tokens as i64),
            "met" => v_bool(tally.supervised_tokens >= args.budget_tokens),
            "refused_markers" => v_int(tally.refused_markers as i64),
            "refused_language" => v_int(tally.refused_language as i64),
            "refused_code" => v_int(tally.refused_code as i64),
            "refused_no_assistant" => v_int(tally.refused_no_assistant as i64),
            "refused_role" => v_int(tally.refused_role as i64),
            "refused_empty" => v_int(tally.refused_empty as i64),
            "stripped_system" => v_int(tally.stripped_system as i64),
            "escaped_rows" => v_int(tally.escaped_rows as i64),
            "too_long" => v_int(tally.too_long as i64),
        }
    };
    let mut provenance = tally_record(&tally);
    provenance.push("rows", v_str(&args.rows.display().to_string()));
    provenance.push("seq_len", v_int(args.seq_len as i64));
    provenance.push("seed", v_int(args.seed as i64));
    provenance.push("english", v_bool(args.english));
    provenance.push("no_code_fences", v_bool(args.no_code_fences));
    provenance.push("training_version", v_str(harness::consts::TRAINING_VERSION));
    provenance.push("bquest_version", v_str(env!("CARGO_PKG_VERSION")));
    provenance.push("converted_at", v_int(epoch_seconds()));
    harness::nu::save_value(
        &args.out.with_extension("provenance.nuon"),
        &harness::nu::Value::record(provenance, span()),
    )?;

    let mut summary = tally_record(&tally);
    summary.push("out", v_str(&args.out.display().to_string()));
    summary.push("seconds", v_float(started.elapsed().as_secs_f64()));
    println!(
        "{}",
        harness::nu::to_nuon_text(&harness::nu::Value::record(summary, span()))?
    );
    Ok(())
}
