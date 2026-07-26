//! The task generators: synthesised, mechanically-verifiable examples
//! for every training stage plus the bench, from one source.
//!
//! Five families. Every answer is checked by machine - `to nuon` for
//! the NUON families, and the pinned nushell itself, sandboxed, for
//! the nushell ones. Nothing here is labelled by judgment.
//!
//! ## DEV
//! ONE GENERATOR emits the supervised examples, the preference pairs,
//! the reinforcement prompts and the bench, split at the source by
//! index. Building the bench separately would let the two drift and
//! it would stop measuring what the training set teaches.
//!
//! EVERY GOLD IS VERIFIED BEFORE IT SHIPS, against the verifier that
//! will grade the model on it. A generator whose own answers do not
//! pass its own grader is the one failure that would poison every
//! stage at once, silently, so it is checked per item rather than
//! reasoned about.
//!
//! EVERY REJECTED ANSWER IS CHECKED TO BE WRONG for the same reason
//! in reverse: a mutation that happened to remain correct would teach
//! the model away from a right answer.
//!
//! The error-location family takes its answer from NUSHELL'S OWN
//! diagnostic rather than from where the mutation was made. Those
//! differ: an unclosed brace opened on one line is reported where the
//! parser reaches end of input, which is a later line. Asserting the
//! mutation site would have shipped a wrong answer for that whole
//! class.
use crate::*;

/// The families, named as they appear in artifacts.
const FAMILY_CONVERT_NUON: &str = "convert_to_nuon";
const FAMILY_FORMAT_NUON: &str = "format_nuon";
const FAMILY_NU_FROM_BASH: &str = "convert_to_nushell";
const FAMILY_NU_FROM_PROSE: &str = "write_nushell";
const FAMILY_NU_ERROR: &str = "find_nushell_error";

const CONVERT_INSTRUCTION: &str =
    "Convert this JSON to NUON. Reply with the NUON value only.";
const PRETTY_INSTRUCTION: &str =
    "Format this NUON with pretty two-space indentation. Reply with the NUON only.";
const CONDENSED_INSTRUCTION: &str =
    "Format this NUON condensed, with no whitespace. Reply with the NUON only.";
const BASH_INSTRUCTION: &str =
    "Convert this shell command to nushell. Reply with the nushell pipeline only.";
const PROSE_INSTRUCTION: &str =
    "Write a nushell pipeline for this. Reply with the pipeline only.";
const ERROR_INSTRUCTION: &str = "This nushell fails. Reply with the 1-based line \
number it fails on, as a bare number.";

/// Field-name vocabulary for synthesised records (snake, as our own
/// data is).
const FIELD_NAMES: [&str; 10] = [
    "name", "count", "size", "path", "enabled", "kind", "score", "label", "depth", "owner",
];

/// Value vocabulary for synthesised strings.
const WORDS: [&str; 8] = [
    "alpha", "beta", "gamma", "delta", "north", "south", "east", "west",
];

/// The structural shapes the NUON families cover, in coverage order so
/// a small run still exercises every one.
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

/// One nushell task, as a template. `{list}` and `{n}` substitute so
/// a handful of templates yields a corpus rather than a handful of
/// items, while every variant stays deterministic and self-contained.
struct NuTemplate {
    bash: &'static str,
    prose: &'static str,
    nu: &'static str,
}

const NU_TEMPLATES: [NuTemplate; 8] = [
    NuTemplate {
        bash: "echo '{spaced}' | tr ' ' '\\n' | awk '$1 > {n}'",
        prose: "From the list {list}, keep only the items greater than {n}.",
        nu: "{list}\n| where {|x| $x > {n} }",
    },
    NuTemplate {
        bash: "echo '{spaced}' | tr ' ' '\\n' | paste -sd+ | bc",
        prose: "Sum the numbers in {list}.",
        nu: "{list}\n| math sum",
    },
    NuTemplate {
        bash: "echo '{spaced}' | tr ' ' '\\n' | sort -rn",
        prose: "Sort {list} in descending order.",
        nu: "{list}\n| sort\n| reverse",
    },
    NuTemplate {
        bash: "echo '{spaced}' | tr ' ' '\\n' | head -2",
        prose: "Take the first two items of {list}.",
        nu: "{list}\n| first 2",
    },
    NuTemplate {
        bash: "echo '{spaced}' | wc -w",
        prose: "Count the items in {list}.",
        nu: "{list}\n| length",
    },
    NuTemplate {
        bash: "echo '{spaced}' | tr ' ' '\\n' | awk '{print $1 * {n}}'",
        prose: "Multiply every item of {list} by {n}.",
        nu: "{list}\n| each {|x| $x * {n} }",
    },
    NuTemplate {
        bash: "echo '{spaced}' | tr ' ' '\\n' | sort -n | uniq",
        prose: "Sort {list} ascending and drop duplicates.",
        nu: "{list}\n| sort\n| uniq",
    },
    NuTemplate {
        bash: "echo '{spaced}' | tr ' ' '\\n' | tail -1",
        prose: "Take the last item of {list}.",
        nu: "{list}\n| last",
    },
];

/// Heads a mutation misspells to break a snippet. Each is a real
/// command, so the break is the recorded implicit-external trap - an
/// undefined name is not a parse error, it becomes an external and
/// fails at runtime.
const MUTABLE_HEADS: [(&str, &str); 7] = [
    ("where", "wher"),
    ("math sum", "math summ"),
    ("sort", "srot"),
    ("first", "firts"),
    ("length", "lenght"),
    ("each", "eahc"),
    ("uniq", "unqi"),
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
fn record_of(
    rng: &mut SplitMix64,
    width: usize,
    mut value: impl FnMut(&mut SplitMix64) -> lib::nu::Value,
) -> lib::nu::Value {
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
            let values: Vec<lib::nu::Value> = (0..3).map(|_| scalar(rng)).collect();
            lib::nu::Value::list(values, span())
        }),
    }
}

/// One generated item, before it is split across the stage artifacts.
struct Item {
    family: &'static str,
    prompt: String,
    /// The answer a supervised example teaches.
    gold: String,
    verifier: &'static str,
    /// What the verifier compares an answer against.
    reference: String,
    rejected: Option<String>,
}

impl Item {
    /// Every item is graded by its own verifier before it ships, and
    /// any rejected answer is graded too and must FAIL. A generator
    /// that ships answers its own grader rejects would poison every
    /// stage at once and leave no symptom.
    fn checked(self) -> BquestResult<Self> {
        let passes = example::verify(self.verifier, &self.gold, &self.reference)?;
        snafu::ensure_whatever!(
            passes > 0.0,
            "[{}] the gold answer does not pass its own verifier ({}):\n{}",
            self.family,
            self.verifier,
            self.gold
        );
        if let Some(rejected) = &self.rejected {
            let fails = example::verify(self.verifier, rejected, &self.reference)?;
            snafu::ensure_whatever!(
                fails == 0.0,
                "[{}] a rejected answer passes the verifier, so it is not wrong:\n{}",
                self.family,
                rejected
            );
        }
        Ok(self)
    }
}

// ---- Family 1 and 2: the NUON families -----------------------------

fn prompt_json(value: &lib::nu::Value) -> BquestResult<String> {
    let json = value_to_json(value)?;
    let rendered = serde_json::to_string_pretty(&json)?;
    Ok(format!("{CONVERT_INSTRUCTION}\n\n{rendered}"))
}

/// The brace-dropping failure: a record emitted as bare key-value
/// lines, which is the YAML-ish shape the channel probe saw.
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
    gold.starts_with('[').then(|| format!("{{rows: {gold}}}"))
}

fn nuon_families(count: usize, seed: u64) -> BquestResult<Vec<Item>> {
    let mut rng = SplitMix64::new(seed);
    let mut items = Vec::with_capacity(count);
    for index in 0..count {
        let value = synth(&mut rng, SHAPES[index % SHAPES.len()]);
        let compact = lib::nu::to_nuon_text(&value)?;
        let pretty = lib::nu::to_nuon_pretty(&value)?;
        let condensed = lib::nu::to_nuon_condensed(&value)?;

        items.push(
            Item {
                family: FAMILY_CONVERT_NUON,
                prompt: prompt_json(&value)?,
                rejected: without_braces(&compact).or_else(|| wrapped_table(&compact)),
                reference: compact.clone(),
                gold: compact.clone(),
                verifier: example::VERIFIER_NUON,
            }
            .checked()?,
        );

        // Formatting is graded on the BYTES, because the bytes are
        // the answer - so the plausible wrong answer is the other
        // style, which is a correct value in the wrong shape.
        items.push(
            Item {
                family: FAMILY_FORMAT_NUON,
                prompt: format!("{PRETTY_INSTRUCTION}\n\n{condensed}"),
                gold: pretty.clone(),
                verifier: example::VERIFIER_EXACT,
                reference: pretty.clone(),
                rejected: (condensed != pretty).then(|| condensed.clone()),
            }
            .checked()?,
        );
        items.push(
            Item {
                family: FAMILY_FORMAT_NUON,
                prompt: format!("{CONDENSED_INSTRUCTION}\n\n{pretty}"),
                gold: condensed.clone(),
                verifier: example::VERIFIER_EXACT,
                reference: condensed.clone(),
                rejected: (condensed != pretty).then(|| pretty.clone()),
            }
            .checked()?,
        );
    }
    Ok(items)
}

// ---- Families 3, 4 and 5: the nushell families ---------------------

fn fill(template: &str, list: &str, spaced: &str, n: i64) -> String {
    template
        .replace("{list}", list)
        .replace("{spaced}", spaced)
        .replace("{n}", &n.to_string())
}

/// Break a snippet by misspelling a command head, which is the
/// recorded implicit-external trap: an undefined name is not a parse
/// error, it becomes an external and fails at runtime.
fn misspell(source: &str) -> Option<String> {
    for (head, wrong) in MUTABLE_HEADS {
        if source.contains(head) {
            return Some(source.replacen(head, wrong, 1));
        }
    }
    None
}

/// The line nushell ITSELF reports the failure on, read out of its
/// diagnostic. Not the line the mutation was made on: an unclosed
/// brace opened on one line is reported where the parser reaches end
/// of input, which is a later one.
fn reported_error_line(stderr: &str) -> Option<usize> {
    let marker = stderr.find("[source:")? + "[source:".len();
    let rest = &stderr[marker..];
    let end = rest.find(':')?;
    rest[..end].parse::<usize>().ok()
}

fn nushell_families(count: usize, seed: u64) -> BquestResult<Vec<Item>> {
    nu_sandbox::require_sandbox()?;
    let mut rng = SplitMix64::new(seed ^ 0x5eed_5eed);
    let mut items = Vec::new();
    let mut skipped = 0usize;

    for index in 0..count {
        let template = &NU_TEMPLATES[index % NU_TEMPLATES.len()];
        let width = 3 + rng.next_below(3);
        let numbers: Vec<i64> = (0..width).map(|_| (rng.next_below(40) + 1) as i64).collect();
        let list = format!(
            "[{}]",
            numbers.iter().map(|n| n.to_string()).collect::<Vec<String>>().join(" ")
        );
        let spaced = numbers
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<String>>()
            .join(" ");
        let n = (rng.next_below(10) + 1) as i64;

        let gold = fill(template.nu, &list, &spaced, n);
        // The reference is the VALUE the gold produces, computed by
        // running it - so the grader compares values and never the
        // rendering, which two correct pipelines can differ on.
        let Some(value) = nu_sandbox::pipeline_value(&gold)? else {
            skipped += 1;
            continue;
        };
        let reference = lib::nu::to_nuon_text(&value)?;
        let broken = misspell(&gold);

        for (family, instruction, ask) in [
            (FAMILY_NU_FROM_BASH, BASH_INSTRUCTION, fill(template.bash, &list, &spaced, n)),
            (FAMILY_NU_FROM_PROSE, PROSE_INSTRUCTION, fill(template.prose, &list, &spaced, n)),
        ] {
            items.push(
                Item {
                    family,
                    prompt: format!("{instruction}\n\n{ask}"),
                    gold: gold.clone(),
                    verifier: example::VERIFIER_NU_VALUE,
                    reference: reference.clone(),
                    rejected: broken.clone(),
                }
                .checked()?,
            );
        }

        // The error-location family: an invalid snippet, proven
        // invalid, whose answer is where NUSHELL says it fails.
        if let Some(broken) = &broken {
            let outcome = nu_sandbox::run_nu(broken, std::time::Duration::from_secs(10))?;
            let reported = reported_error_line(&outcome.stderr);
            match (outcome.ok, reported) {
                // It must actually fail, and nushell must say where.
                (false, Some(line)) => {
                    let line_count = broken.lines().count();
                    // A plausible wrong answer is a DIFFERENT line of
                    // the same snippet, not an arbitrary number.
                    let other = (1..=line_count).find(|candidate| *candidate != line);
                    items.push(
                        Item {
                            family: FAMILY_NU_ERROR,
                            prompt: format!("{ERROR_INSTRUCTION}\n\n{broken}"),
                            gold: line.to_string(),
                            verifier: example::VERIFIER_EXACT,
                            reference: line.to_string(),
                            rejected: other.map(|candidate| candidate.to_string()),
                        }
                        .checked()?,
                    );
                }
                // A mutation that did not break it, or that broke it
                // without nushell saying where, has no answer to
                // teach. A TIMEOUT is called out separately: it means
                // the snippet hangs rather than fails, which is a
                // different defect in the template.
                _ => {
                    if outcome.timed_out {
                        eprintln!("taskgen: a broken snippet TIMED OUT rather than failing");
                    }
                    skipped += 1;
                }
            }
        }
    }
    if skipped > 0 {
        eprintln!(
            "taskgen: {skipped} nushell candidates skipped - the pipeline did not \
             run cleanly, or a mutation did not break it"
        );
    }
    Ok(items)
}

// ---- Emission ------------------------------------------------------

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

/// `bquest taskgen all`: every family's supervised, preference and
/// reinforcement artifacts plus the held-out bench, from one seeded
/// generation.
pub(crate) fn taskgen_all(args: &TaskgenAllArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    snafu::ensure_whatever!(args.count > 0, "nothing to generate");
    snafu::ensure_whatever!(
        args.bench_every >= 2,
        "a bench share of every {} item leaves no training data",
        args.bench_every
    );
    let mut items = nuon_families(args.count, args.seed)?;
    items.extend(nushell_families(args.count, args.seed)?);
    fs::create_dir_all(&args.out)?;

    let mut sft_train = Vec::new();
    let mut sft_bench = Vec::new();
    let mut dpo_train = Vec::new();
    let mut rlvr_train = Vec::new();
    let mut rlvr_bench = Vec::new();
    let mut without_rejected = 0usize;
    let mut per_family: Vec<(String, usize)> = Vec::new();

    for (index, item) in items.iter().enumerate() {
        match per_family.iter_mut().find(|(name, _)| name == item.family) {
            Some((_, tally)) => *tally += 1,
            None => per_family.push((item.family.to_string(), 1)),
        }
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
                "verifier" => v_str(item.verifier),
                "reference" => v_str(&item.reference),
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

    let families = lib::nu::Value::list(
        per_family
            .iter()
            .map(|(name, tally)| {
                lib::nu::Value::record(
                    lib::nu::record! {
                        "family" => v_str(name),
                        "items" => v_int(*tally as i64),
                    },
                    span(),
                )
            })
            .collect(),
        span(),
    );
    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "families" => families,
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
