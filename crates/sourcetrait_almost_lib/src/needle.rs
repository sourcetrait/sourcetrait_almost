use crate::*;

/// Knobs for the needle/passkey retrieval battery (A2 phase 1).
#[derive(Debug, Clone)]
pub struct NeedleOptions {
    /// Runtime settings the battery runs under (flash, eviction). The
    /// observation pass arms profile_attn itself; a settings-file
    /// profile_attn is ignored here.
    pub settings: Settings,
    /// Seeds the case generator only; decoding is greedy.
    pub seed: u64,
    /// When set, per-cell results are also written here as JSON.
    pub out: Option<PathBuf>,
    /// When set, run the A2 observation pass instead of the full grid
    /// (attn-profile build, eager only: long lengths, single mode, fewer
    /// keys) and write the per-head attention-mass profile here as JSON.
    pub profile_out: Option<PathBuf>,
}

/// Context lengths (total prompt tokens) the battery targets.
const LENGTHS: [usize; 5] = [4096, 8192, 16384, 24576, 32768];
/// Needle depths, percent into the filler span (10 = deepest history,
/// i.e. the longest retrieval distance from the trailing question).
const DEPTHS: [usize; 5] = [10, 25, 50, 75, 90];
/// Randomized keys per (length, depth) cell.
const KEYS_PER_CELL: usize = 5;
/// Observation-pass subset: the long-context lengths only (the middle
/// region is what the pass measures) with fewer keys - the per-row mass
/// statistics are dense, so a handful of prompts per cell suffices.
const PROFILE_LENGTHS: [usize; 3] = [16384, 24576, 32768];
const PROFILE_KEYS_PER_CELL: usize = 2;

const ADJECTIVES: [&str; 12] = [
    "amber", "quiet", "narrow", "restless", "pale", "sturdy", "brisk",
    "hollow", "mossy", "distant", "crooked", "faded",
];
const NOUNS: [&str; 12] = [
    "ferry", "orchard", "printing press", "lighthouse", "granary",
    "tram line", "reservoir", "workshop", "archive", "footbridge",
    "windmill", "observatory",
];
const VERBS: [&str; 12] = [
    "reopened", "changed hands", "lost its roof", "gained a keeper",
    "was repainted", "went silent", "drew a crowd", "was surveyed",
    "flooded twice", "was rewired", "shed its scaffolding", "closed early",
];
const PLACES: [&str; 12] = [
    "eastern quay", "old market", "salt flats", "upper meadow",
    "railway cut", "harbor mouth", "mill pond", "county line",
    "stone causeway", "lower terrace", "gravel pit", "north gate",
];
const TIMES: [&str; 12] = [
    "before the first frost", "during the long drought",
    "after the spring auction", "midway through the census",
    "while the road was closed", "just after the equinox",
    "before the ferry strike", "during the harvest weeks",
    "after the survey ended", "when the tolls were lifted",
    "before the new charter", "as the season turned",
];
const LABELS: [&str; 12] = [
    "the harbor ledger", "the granite archive", "the orchard trust",
    "the causeway fund", "the meadow registry", "the tramway office",
    "the windmill estate", "the quay commission", "the census bureau",
    "the terrace council", "the reservoir board", "the gatehouse vault",
];

/// SplitMix64: deterministic, dependency-free case generation.
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^ (mixed >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    fn pick(&mut self, items: &[&'static str]) -> &'static str {
        items[self.below(items.len())]
    }
}

fn filler_sentence(rng: &mut SplitMix64) -> String {
    format!(
        "The {} {} {} near the {} {}. ",
        rng.pick(&ADJECTIVES),
        rng.pick(&NOUNS),
        rng.pick(&VERBS),
        rng.pick(&PLACES),
        rng.pick(&TIMES),
    )
}

fn needle_sentence(label: &str, key: u32) -> String {
    format!("The secret passkey for {label} is {key}. ")
}

fn question(label: &str) -> String {
    format!("What is the secret passkey for {label}? Reply with just the number.")
}

/// A planted needle: filler-sentence index to insert before, label, key.
struct Plant {
    sentence_index: usize,
    label: &'static str,
    key: u32,
}

/// One battery case: the full un-wrapped user text and the expected key.
pub(crate) struct Case {
    pub(crate) text: String,
    pub(crate) expect: String,
}

/// Distinct labels and keys for one case (target first).
pub(crate) fn draw_identities(rng: &mut SplitMix64, count: usize) -> Vec<(&'static str, u32)> {
    let mut identities: Vec<(&'static str, u32)> = Vec::with_capacity(count);
    while identities.len() < count {
        let label = rng.pick(&LABELS);
        let key = 10_000 + (rng.below(90_000) as u32);
        if identities.iter().any(|(l, k)| *l == label || *k == key) {
            continue;
        }
        identities.push((label, key));
    }
    identities
}

/// Build one case: `filler_sentences` sentences with the target needle at
/// `depth` percent (plus distractors in multi mode), the question last.
pub(crate) fn build_case(
    rng: &mut SplitMix64,
    filler_sentences: usize,
    depth: usize,
    multi: bool,
) -> Case {
    let needle_count = if multi { 3 } else { 1 };
    let identities = draw_identities(rng, needle_count);
    let (target_label, target_key) = identities[0];

    let mut plants: Vec<Plant> = Vec::with_capacity(needle_count);
    plants.push(Plant {
        sentence_index: depth * filler_sentences / 100,
        label: target_label,
        key: target_key,
    });
    // Distractor depths are deterministic offsets from the target's,
    // wrapped into [5, 95] so they never collide with it.
    for (offset_index, &(label, key)) in identities.iter().enumerate().skip(1) {
        let distractor_depth = (depth + 27 * offset_index) % 91 + 5;
        plants.push(Plant {
            sentence_index: distractor_depth * filler_sentences / 100,
            label,
            key,
        });
    }
    // Insert deepest-first so earlier insertions do not shift later ones.
    plants.sort_by_key(|plant| std::cmp::Reverse(plant.sentence_index));

    let mut sentences: Vec<String> = Vec::with_capacity(filler_sentences + needle_count);
    for _ in 0..filler_sentences {
        sentences.push(filler_sentence(rng));
    }
    for plant in &plants {
        sentences.insert(
            plant.sentence_index.min(sentences.len()),
            needle_sentence(plant.label, plant.key),
        );
    }

    let mut text = sentences.concat();
    text.push_str("\n\n");
    text.push_str(&question(target_label));
    Case {
        text,
        expect: target_key.to_string(),
    }
}

/// Tokens the chat wrap + question + a needle cost beyond the filler, plus
/// the mean tokens per filler sentence - measured with the real tokenizer
/// so length targeting stays honest across tokenizer changes.
fn measure_costs(
    tokenizer: &tokenizers::Tokenizer,
    rng: &mut SplitMix64,
) -> AlmostResult<(usize, f64)> {
    let probe_question = question(LABELS[0]);
    let wrapped = chat_wrap(&format!("\n\n{probe_question}"));
    let overhead = encode_len(tokenizer, &wrapped)? + 14;
    let mut sample = String::new();
    for _ in 0..64 {
        sample.push_str(&filler_sentence(rng));
    }
    let per_sentence = encode_len(tokenizer, &sample)? as f64 / 64.0;
    Ok((overhead, per_sentence))
}

fn encode_len(tokenizer: &tokenizers::Tokenizer, text: &str) -> AlmostResult<usize> {
    match tokenizer.encode(text, true) {
        Ok(encoding) => Ok(encoding.get_ids().len()),
        Err(error) => snafu::whatever!("needle tokenization failed: {error}"),
    }
}

struct CellResult {
    length: usize,
    depth: usize,
    multi: bool,
    hits: usize,
    cases: usize,
    prompt_tokens_mean: usize,
}

/// A2 phase 1 retrieval battery: passkey needles across depth x length,
/// single and multi (2 distractors) modes, greedy exact-match scoring.
/// Reports the checkpoint's baseline; exits Ok regardless of score (the
/// gate is a later profiled-vs-exact comparison).
pub fn needle(
    paths: &ModelPaths,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    opts: &NeedleOptions,
) -> AlmostResult<()> {
    let profiling = opts.profile_out.is_some();
    if profiling {
        snafu::ensure_whatever!(
            !opts.settings.use_flash_attn,
            "the observation pass reads eager attention weights; use an eager settings profile for --profile-out"
        );
    }
    let config: Olmo3Config = serde_json::from_reader(std::fs::File::open(&paths.config)?)?;
    let tokenizer = match tokenizers::Tokenizer::from_file(&paths.tokenizer) {
        Ok(tokenizer) => tokenizer,
        Err(error) => snafu::whatever!("loading tokenizer.json failed: {error}"),
    };
    let load_start = Instant::now();
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths.shards, dtype, device)? };
    let settings = Settings {
        profile_attn: profiling,
        ..opts.settings
    };
    let mut model = Model::new(&config, settings, vb)?;
    eprintln!(
        "almost needle: weights loaded in {:.1}s; device {device:?}, dtype {dtype:?}",
        load_start.elapsed().as_secs_f32()
    );

    let mut rng = SplitMix64(opts.seed);
    let (overhead, per_sentence) = measure_costs(&tokenizer, &mut rng)?;
    let generate_options = GenerateOptions {
        greedy: true,
        temperature: consts::DEFAULT_TEMPERATURE,
        top_p: consts::DEFAULT_TOP_P,
        sample_len: 24,
        seed: opts.seed,
        speculate: false,
        dump_logits: None,
    };

    let modes: &[bool] = if profiling { &[false] } else { &[false, true] };
    let lengths: &[usize] = if profiling { &PROFILE_LENGTHS } else { &LENGTHS };
    let keys_per_cell = if profiling { PROFILE_KEYS_PER_CELL } else { KEYS_PER_CELL };

    let battery_start = Instant::now();
    let mut results: Vec<CellResult> = Vec::new();
    for &multi in modes {
        for &length in lengths {
            let filler_sentences =
                ((length.saturating_sub(overhead)) as f64 / per_sentence) as usize;
            for &depth in &DEPTHS {
                let mut hits = 0usize;
                let mut token_total = 0usize;
                for _ in 0..keys_per_cell {
                    let case = build_case(&mut rng, filler_sentences, depth, multi);
                    let mut generation = model.generate(&tokenizer, &chat_wrap(&case.text), &generate_options)?;
                    let mut answer = String::new();
                    for step in &mut generation {
                        let step = step?;
                        if let Some(chunk) = step.chunk {
                            answer.push_str(&chunk);
                        }
                    }
                    let report = generation.finish()?;
                    if let Some(rest) = &report.rest {
                        answer.push_str(rest);
                    }
                    token_total += report.prompt_token_count;
                    if answer.contains(&case.expect) {
                        hits += 1;
                    }
                }
                results.push(CellResult {
                    length,
                    depth,
                    multi,
                    hits,
                    cases: keys_per_cell,
                    prompt_tokens_mean: token_total / keys_per_cell,
                });
            }
            let row: Vec<&CellResult> = results
                .iter()
                .filter(|cell| cell.length == length && cell.multi == multi)
                .collect();
            let hits: usize = row.iter().map(|cell| cell.hits).sum();
            let cases: usize = row.iter().map(|cell| cell.cases).sum();
            eprintln!(
                "almost needle: {} {} done - {hits}/{cases} ({:.0}s elapsed)",
                if multi { "multi" } else { "single" },
                length,
                battery_start.elapsed().as_secs_f32()
            );
        }
    }

    println!("mode | length | tokens(mean) | hits/cases | per-depth {DEPTHS:?}");
    for &multi in modes {
        for &length in lengths {
            let row: Vec<&CellResult> = results
                .iter()
                .filter(|cell| cell.length == length && cell.multi == multi)
                .collect();
            let hits: usize = row.iter().map(|cell| cell.hits).sum();
            let cases: usize = row.iter().map(|cell| cell.cases).sum();
            let per_depth: Vec<String> = row
                .iter()
                .map(|cell| format!("{}/{}", cell.hits, cell.cases))
                .collect();
            let tokens_mean: usize =
                row.iter().map(|cell| cell.prompt_tokens_mean).sum::<usize>() / row.len().max(1);
            println!(
                "{} | {} | {} | {}/{} | {}",
                if multi { "multi" } else { "single" },
                length,
                tokens_mean,
                hits,
                cases,
                per_depth.join(" ")
            );
        }
    }
    let total_hits: usize = results.iter().map(|cell| cell.hits).sum();
    let total_cases: usize = results.iter().map(|cell| cell.cases).sum();
    println!("needle total | {total_hits}/{total_cases}");

    if let Some(path) = &opts.out {
        let rows: Vec<serde_json::Value> = results
            .iter()
            .map(|cell| {
                serde_json::json!({
                    "mode": if cell.multi { "multi" } else { "single" },
                    "length": cell.length,
                    "depth": cell.depth,
                    "hits": cell.hits,
                    "cases": cell.cases,
                    "prompt_tokens_mean": cell.prompt_tokens_mean,
                })
            })
            .collect();
        let payload = serde_json::json!({
            "seed": opts.seed,
            "total_hits": total_hits,
            "total_cases": total_cases,
            "cells": rows,
        });
        std::fs::write(path, serde_json::to_string_pretty(&payload)?)?;
        eprintln!("almost needle: results written to {}", path.display());
    }

    #[cfg(feature = "attn-profile")]
    if let Some(path) = &opts.profile_out {
        let layers = model.take_profile();
        let layer_rows: Vec<serde_json::Value> = layers
            .iter()
            .map(|(layer_index, report)| {
                let heads: Vec<serde_json::Value> = report
                    .heads
                    .iter()
                    .enumerate()
                    .map(|(head_index, head)| {
                        serde_json::json!({
                            "head": head_index,
                            "prefill_middle": head.prefill_middle,
                            "prefill_sink_before": head.prefill_sink_before,
                            "decode_middle": head.decode_middle,
                            "decode_sink_before": head.decode_sink_before,
                            "sink_by_position": head.sink_by_position,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "layer": layer_index,
                    "window": report.window,
                    "s_probe": report.s_probe,
                    "prefill_rows": report.prefill_rows,
                    "prefill_rows_past_window": report.prefill_rows_past_window,
                    "decode_rows": report.decode_rows,
                    "decode_rows_past_window": report.decode_rows_past_window,
                    "heads": heads,
                })
            })
            .collect();
        let payload = serde_json::json!({
            "seed": opts.seed,
            "checkpoint": consts::DEFAULT_MODEL_ID,
            "layers": layer_rows,
        });
        std::fs::write(path, serde_json::to_string_pretty(&payload)?)?;
        eprintln!("almost needle: attention profile written to {}", path.display());
    }
    Ok(())
}
