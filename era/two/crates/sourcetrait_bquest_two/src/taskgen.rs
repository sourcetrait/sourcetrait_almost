//! The task generator: synthesised, mechanically-verifiable examples
//! for every training stage plus the bench, from one source.
//!
//! The family built here is CONVERT-TO-NUON. It earns its place ahead
//! of the others because its ground truth is free - `to nuon` IS the
//! correct answer, so nothing is labelled by judgment - and because it
//! puts pressure exactly where the base was measured to fail: the
//! modern table literal never appeared at all, and record braces were
//! the flaky case. Prose families and the nushell-formatting families
//! want a formatter that does not exist yet.
//!
//! ## DEV
//! One generator emits the SUPERVISED examples, the PREFERENCE pairs,
//! the REINFORCEMENT prompts and the BENCH, split at the source by
//! index. Building the bench separately would let the two drift and
//! the bench would stop measuring what the training set teaches.
//!
//! The rejected side of a preference pair is not invented: both
//! mutations reproduce failure modes the channel probe actually
//! measured on this checkpoint - dropping the record braces for
//! bare key-value lines, and wrapping a table in a record. Each is
//! CHECKED to be wrong at generation time, so a mutation that
//! happened to round-trip could never be shipped as a rejected
//! answer.
//!
//! Inputs are synthesised rather than harvested, which is a known
//! risk rather than an oversight: mock data is clean by construction
//! and real files are not, so whether this transfers is a question
//! the bench answers rather than one the generator settles.
//! ##
use crate::*;

/// The instruction every example carries. One wording, so the model
/// is not also learning to parse varied phrasings of the same ask.
const INSTRUCTION: &str = "Convert this JSON to NUON. Reply with the NUON value only.";

/// Field-name vocabulary for synthesised records (snake, as our own
/// data is).
const FIELD_NAMES: [&str; 10] = [
    "name", "count", "size", "path", "enabled", "kind", "score", "label", "depth", "owner",
];

/// Value vocabulary for synthesised strings.
const WORDS: [&str; 8] = [
    "alpha", "beta", "gamma", "delta", "north", "south", "east", "west",
];

/// The structural shapes the family covers, in coverage order so a
/// small run still exercises every one.
#[derive(Debug, Clone, Copy)]
enum Shape {
    FlatRecord,
    NestedRecord,
    ScalarList,
    Table,
    NestedTable,
    RecordOfLists,
}

const SHAPES: [Shape; 6] = [
    Shape::FlatRecord,
    Shape::NestedRecord,
    Shape::ScalarList,
    Shape::Table,
    Shape::NestedTable,
    Shape::RecordOfLists,
];

fn word(rng: &mut SplitMix64) -> String {
    WORDS[rng.next_below(WORDS.len())].to_string()
}

fn field(rng: &mut SplitMix64) -> String {
    FIELD_NAMES[rng.next_below(FIELD_NAMES.len())].to_string()
}

/// A JSON-representable scalar. The conversion family stays inside
/// what JSON can carry, so nothing is lost on the way in and the gold
/// is a faithful answer to the question asked.
fn scalar(rng: &mut SplitMix64) -> lib::nu::Value {
    match rng.next_below(4) {
        0 => v_int((rng.next_below(1000)) as i64),
        1 => v_str(&word(rng)),
        2 => v_bool(rng.next_below(2) == 1),
        _ => v_float((rng.next_below(10_000) as f64) / 100.0),
    }
}

/// A record of `width` distinct fields (a repeat draw would collide,
/// and a record's keys are unique).
fn record_of(rng: &mut SplitMix64, width: usize, mut value: impl FnMut(&mut SplitMix64) -> lib::nu::Value) -> lib::nu::Value {
    let mut record = lib::nu::Record::new();
    let mut used: Vec<String> = Vec::new();
    let mut attempts = 0;
    while used.len() < width && attempts < 64 {
        attempts += 1;
        let name = field(rng);
        if used.contains(&name) {
            continue;
        }
        used.push(name.clone());
        record.push(name, value(rng));
    }
    lib::nu::Value::record(record, span())
}

/// A uniform table: the same columns in every row, which is what
/// makes it a table rather than a list of records.
fn table_of(rng: &mut SplitMix64, rows: usize, columns: usize) -> lib::nu::Value {
    let mut names: Vec<String> = Vec::new();
    let mut attempts = 0;
    while names.len() < columns && attempts < 64 {
        attempts += 1;
        let name = field(rng);
        if !names.contains(&name) {
            names.push(name);
        }
    }
    let values: Vec<lib::nu::Value> = (0..rows)
        .map(|_| {
            let mut record = lib::nu::Record::new();
            for name in &names {
                record.push(name.clone(), scalar(rng));
            }
            lib::nu::Value::record(record, span())
        })
        .collect();
    lib::nu::Value::list(values, span())
}

/// Synthesise one value of the given shape.
fn synth(rng: &mut SplitMix64, shape: Shape) -> lib::nu::Value {
    match shape {
        Shape::FlatRecord => record_of(rng, 3, scalar),
        Shape::NestedRecord => record_of(rng, 2, |rng| record_of(rng, 2, scalar)),
        Shape::ScalarList => {
            let count = 3 + rng.next_below(3);
            let values: Vec<lib::nu::Value> = (0..count).map(|_| scalar(rng)).collect();
            lib::nu::Value::list(values, span())
        }
        Shape::Table => {
            let rows = 2 + rng.next_below(3);
            let columns = 2 + rng.next_below(2);
            table_of(rng, rows, columns)
        }
        Shape::NestedTable => {
            let inner = table_of(rng, 2, 2);
            let mut record = lib::nu::Record::new();
            record.push(String::from("rows"), inner);
            record.push(String::from("total"), scalar(rng));
            lib::nu::Value::record(record, span())
        }
        Shape::RecordOfLists => record_of(rng, 2, |rng| {
            lib::nu::Value::list((0..3).map(|_| scalar(rng)).collect(), span())
        }),
    }
}

/// The prompt text: the instruction plus the JSON rendering of the
/// value the reply must reproduce as NUON.
fn prompt_text(value: &lib::nu::Value) -> BquestResult<String> {
    let json = value_to_json(value)?;
    let rendered = serde_json::to_string_pretty(&json)?;
    Ok(format!("{INSTRUCTION}\n\n{rendered}"))
}

/// The brace-dropping failure: a record emitted as bare key-value
/// lines, which is the YAML-ish shape the probe saw.
fn without_braces(gold: &str) -> Option<String> {
    let inner = gold.strip_prefix('{')?.strip_suffix('}')?;
    Some(
        inner
            .split(", ")
            .map(|pair| pair.trim().to_string())
            .collect::<Vec<String>>()
            .join("\n"),
    )
}

/// The wrapper-record failure: a table handed back inside an invented
/// record instead of as a table literal.
fn wrapped_table(gold: &str) -> Option<String> {
    if gold.starts_with('[') {
        Some(format!("{{rows: {gold}}}"))
    } else {
        None
    }
}

/// A wrong answer for the preference pair, checked to be wrong.
fn rejected_for(gold: &str) -> BquestResult<Option<String>> {
    let candidate = without_braces(gold).or_else(|| wrapped_table(gold));
    let Some(candidate) = candidate else {
        return Ok(None);
    };
    // A mutation that still parses to the same value is not a wrong
    // answer, and shipping it would teach the model to avoid a
    // correct rendering.
    if example::verify(example::VERIFIER_NUON, &candidate, gold)? > 0.0 {
        return Ok(None);
    }
    Ok(Some(candidate))
}

/// One generated item, before it is split across the stage artifacts.
struct Item {
    prompt: String,
    gold: String,
    rejected: Option<String>,
}

fn generate_items(count: usize, seed: u64) -> BquestResult<Vec<Item>> {
    let mut rng = SplitMix64::new(seed);
    let mut items = Vec::with_capacity(count);
    for index in 0..count {
        let shape = SHAPES[index % SHAPES.len()];
        let value = synth(&mut rng, shape);
        let gold = lib::nu::to_nuon_text(&value)?;
        // The gold must be its own correct answer, or the family is
        // grading against something the renderer cannot produce.
        snafu::ensure_whatever!(
            example::verify(example::VERIFIER_NUON, &gold, &gold)? > 0.0,
            "the generated gold does not verify against itself"
        );
        items.push(Item {
            prompt: prompt_text(&value)?,
            rejected: rejected_for(&gold)?,
            gold,
        });
    }
    Ok(items)
}

fn messages_value(prompt: &str) -> lib::nu::Value {
    lib::nu::Value::list(
        vec![lib::nu::Value::record(
            lib::nu::record! {
                "role" => v_str("user"),
                "content" => v_str(prompt),
            },
            span(),
        )],
        span(),
    )
}

fn write_table(path: &Path, rows: Vec<lib::nu::Value>, typedef: &str) -> BquestResult<usize> {
    let table = lib::nu::Value::list(rows, span());
    lib::nu::conform(&table, &lib::nu::parse_typedef(typedef)?)?;
    let count = table.as_list().map(|rows| rows.len()).unwrap_or(0);
    lib::nu::save_value(path, &table)?;
    Ok(count)
}

/// `bquest taskgen nuon`: emit the supervised, preference and
/// reinforcement artifacts plus the held-out bench, all from one
/// seeded generation.
pub(crate) fn taskgen_nuon(args: &TaskgenNuonArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    snafu::ensure_whatever!(args.count > 0, "nothing to generate");
    snafu::ensure_whatever!(
        args.bench_every >= 2,
        "a bench share of every {} item leaves no training data",
        args.bench_every
    );
    let items = generate_items(args.count, args.seed)?;
    fs::create_dir_all(&args.out)?;

    let mut sft_train = Vec::new();
    let mut sft_bench = Vec::new();
    let mut dpo_train = Vec::new();
    let mut rlvr_train = Vec::new();
    let mut rlvr_bench = Vec::new();
    let mut without_rejected = 0usize;

    for (index, item) in items.iter().enumerate() {
        // Split at the SOURCE, so the bench measures exactly what the
        // training set teaches.
        let bench = index % args.bench_every == 0;
        let supervised = lib::nu::Value::record(
            lib::nu::record! {
                "messages" => lib::nu::Value::list(
                    vec![
                        lib::nu::Value::record(
                            lib::nu::record! {
                                "role" => v_str("user"),
                                "content" => v_str(&item.prompt),
                            },
                            span(),
                        ),
                        lib::nu::Value::record(
                            lib::nu::record! {
                                "role" => v_str("assistant"),
                                "content" => v_str(&item.gold),
                            },
                            span(),
                        ),
                    ],
                    span(),
                ),
            },
            span(),
        );
        let verifiable = lib::nu::Value::record(
            lib::nu::record! {
                "prompt" => messages_value(&item.prompt),
                "verifier" => v_str(example::VERIFIER_NUON),
                "reference" => v_str(&item.gold),
            },
            span(),
        );
        if bench {
            sft_bench.push(supervised);
            rlvr_bench.push(verifiable);
            continue;
        }
        sft_train.push(supervised);
        rlvr_train.push(verifiable);
        match &item.rejected {
            Some(rejected) => dpo_train.push(lib::nu::Value::record(
                lib::nu::record! {
                    "prompt" => messages_value(&item.prompt),
                    "chosen" => v_str(&item.gold),
                    "rejected" => v_str(rejected),
                },
                span(),
            )),
            None => without_rejected += 1,
        }
    }

    let sft_train_count =
        write_table(&args.out.join("sft_train.nuon"), sft_train, example::SFT_TYPEDEF)?;
    let sft_bench_count =
        write_table(&args.out.join("sft_bench.nuon"), sft_bench, example::SFT_TYPEDEF)?;
    let dpo_count =
        write_table(&args.out.join("dpo_train.nuon"), dpo_train, example::DPO_TYPEDEF)?;
    let rlvr_train_count =
        write_table(&args.out.join("rlvr_train.nuon"), rlvr_train, example::RLVR_TYPEDEF)?;
    let rlvr_bench_count =
        write_table(&args.out.join("rlvr_bench.nuon"), rlvr_bench, example::RLVR_TYPEDEF)?;

    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "family" => v_str("convert_to_nuon"),
            "generated" => v_int(items.len() as i64),
            "sft_train" => v_int(sft_train_count as i64),
            "sft_bench" => v_int(sft_bench_count as i64),
            "dpo_train" => v_int(dpo_count as i64),
            "rlvr_train" => v_int(rlvr_train_count as i64),
            "rlvr_bench" => v_int(rlvr_bench_count as i64),
            "no_wrong_answer_available" => v_int(without_rejected as i64),
            "seed" => v_int(args.seed as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}
