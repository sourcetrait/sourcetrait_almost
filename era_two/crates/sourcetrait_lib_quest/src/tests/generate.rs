//! The GenerationSurface (was D6) gates: the greedy trajectory replay
//! against the oracle f32 generated ids (cpu f32, exact), the bench
//! rows, and the needle retrieval grid - the refquest 05 instrument
//! re-expressed verbatim (LCG filler, calibration loop,
//! stop-before-append greedy).
use crate::*;

fn u32_field(tensors: &safetensors::SafeTensors, name: &str) -> Vec<u32> {
    let view = tensors.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::U32, "{name} dtype");
    view.data()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn dpo_dir() -> PathBuf {
    model_dir(consts::DPO_MODEL_NAME).expect("dpo dir")
}

#[test]
#[ignore = "needs the DPO checkpoint + QUEST_STATELESS_REFERENCE"]
fn generation_gate_greedy_trajectory_matches_oracle_f32() {
    // The oracle short dump's generated_ids are the original stack's
    // greedy continuation of the same 43-token chat prompt; at f32
    // the trajectories must match id-for-id (the StateCarry gates
    // already matched argmax on every reference row).
    let reference_path = env::var("QUEST_STATELESS_REFERENCE").expect(
        "QUEST_STATELESS_REFERENCE must point at short_cpu_f32_torch_eager_single.safetensors",
    );
    let bytes = fs::read(&reference_path).expect("reference reads");
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("reference parses");
    let reference_generated = u32_field(&tensors, "generated_ids");
    assert!(reference_generated.len() >= 20);

    let dir = dpo_dir();
    let config = load_config(&dir).expect("config");
    let weights = mmap_weights(&dir, candle_core::DType::F32, &candle_core::Device::Cpu)
        .expect("mmap");
    let mut model = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    let options = GenerateOptions::greedy(20);
    let mut generation = model
        .generate(&tokenizer, "What is the capital of France?", &options)
        .expect("generation starts");
    let mut ids = Vec::new();
    let mut text = String::new();
    for step in generation.by_ref() {
        let step = step.expect("step");
        ids.push(step.token_id);
        text.push_str(&step.chunk);
    }
    let report = generation.finish();
    text.push_str(&report.rest);

    println!(
        "generation trajectory: prompt {} tokens, generated {} ({:?}), text {text:?}",
        report.prompt_token_count, report.generated_token_count, report.finish_reason
    );
    assert_eq!(report.prompt_token_count, 43, "the pinned chat render");
    // The reference's greedy loop collects 64 ids with NO stop check
    // (dump.py produces ids for replay, not a transcript - it
    // generates straight past the stop token). Our surface stops AT
    // the stop, so the gate: our ids equal the reference prefix up to
    // its first stop id, and our end is StopToken.
    let stop_ids = [consts::TOKEN_IM_END, consts::TOKEN_ENDOFTEXT];
    let first_stop = reference_generated
        .iter()
        .position(|id| stop_ids.contains(id))
        .expect("the reference trajectory contains a stop token");
    assert!(
        first_stop < 20,
        "the 20-token budget must cover the reference's stop (at {first_stop})"
    );
    assert_eq!(report.finish_reason, Some(FinishReason::StopToken));
    assert_eq!(
        ids,
        reference_generated[..first_stop],
        "greedy trajectory (text: {text:?})"
    );
}

#[cfg(feature = "cuda")]
#[derive(serde::Deserialize)]
struct NeedleKey {
    name: String,
    value: String,
}

#[cfg(feature = "cuda")]
#[derive(serde::Deserialize)]
struct NeedleSpec {
    depths: Vec<usize>,
    keys: Vec<NeedleKey>,
    filler_seed: u64,
    gen_tokens: usize,
    needle_template: String,
    question_template: String,
    words: Vec<String>,
}

/// The spec's LCG: state = (state * 1103515245 + 12345) mod 2^31,
/// word = words[state mod pool]. Seeded filler_seed + target length.
#[cfg(feature = "cuda")]
fn lcg_words(words: &[String], count: usize, seed: u64) -> Vec<String> {
    let mut state = seed;
    let pool = words.len() as u64;
    (0..count)
        .map(|_| {
            state = (state.wrapping_mul(1_103_515_245).wrapping_add(12_345)) % 2_147_483_648;
            words[(state % pool) as usize].clone()
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn encoded_len(tokenizer: &tokenizers::Tokenizer, text: &str) -> usize {
    tokenizer
        .encode(text, false)
        .expect("encodes")
        .get_ids()
        .len()
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "perf rows, not a gate: the incumbent bench recipes (cuda + QUEST_BENCH_PROMPTS_DIR)"]
fn bench_rows_cuda() {
    // The IncumbentBench method: raw prompt files (the incumbent's
    // exact seeds), greedy, a FORCED decode length (ignore_stops -
    // their ignore_eos; QUEST_BENCH_DECODE overrides the standing
    // 128-token window - the DipDiagnostic knob). Rows print; nothing
    // asserts beyond completion - the CorrectnessCore targets are
    // vllm's 62 t/s short / 42.9 @32K.
    let prompts_dir = env::var("QUEST_BENCH_PROMPTS_DIR")
        .expect("QUEST_BENCH_PROMPTS_DIR must point at the refquest prompts dir");
    let dir = dpo_dir();
    let config = load_config(&dir).expect("config");
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
    let settings = match env::var("QUEST_BENCH_SETTINGS") {
        Ok(token) => LibSettings::load(Some(&token)).expect("bench settings token"),
        Err(_) => match env::var("QUEST_BENCH_GRAPH").as_deref() {
            Ok("1") => LibSettings {
                graph: true,
                ..LibSettings::default()
            },
            _ => LibSettings::default(),
        },
    };
    let bench_speculate = settings.generation.speculate;
    let mut model = OlmoHybrid::new(&config, settings, weights).expect("model");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    // QUEST_BENCH_FILES narrows the row set (csv of filenames) - the
    // nsys profiling driver runs one row per process.
    let files = env::var("QUEST_BENCH_FILES").unwrap_or_else(|_| {
        "mid_filler_2k.txt,long_filler_8k.txt,bench_filler_16k.txt,bench_filler_32k.txt"
            .to_string()
    });
    let decode_window: usize = env::var("QUEST_BENCH_DECODE")
        .map(|value| value.parse().expect("decode window"))
        .unwrap_or(128);
    for file in files.split(',').map(str::trim) {
        let prompt = fs::read_to_string(Path::new(&prompts_dir).join(file))
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        let options = GenerateOptions {
            temperature: None,
            top_p: None,
            sample_len: decode_window,
            chat: false,
            ignore_stops: true,
            speculate: bench_speculate,
            ..GenerateOptions::default()
        };
        let mut generation = model
            .generate(&tokenizer, &prompt, &options)
            .expect("generation starts");
        for step in generation.by_ref() {
            step.expect("step");
        }
        let report = generation.finish();
        let prefill_rate = report.prompt_token_count as f64 / report.prefill_seconds;
        let decode_rate =
            (report.generated_token_count.saturating_sub(1)) as f64 / report.decode_seconds;
        println!(
            "bench {file}: prompt {} tok, prefill {:.0} tok/s, decode {:.1} tok/s ({} tokens, drafted {}, accepted {})",
            report.prompt_token_count,
            prefill_rate,
            decode_rate,
            report.generated_token_count,
            report.drafted_token_count,
            report.accepted_draft_token_count
        );
        assert_eq!(
            report.generated_token_count, decode_window,
            "forced decode length"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + QUEST_NEEDLE_SPEC + a cuda card"]
fn generation_gate_needle_grid_cuda() {
    let spec_path = env::var("QUEST_NEEDLE_SPEC")
        .expect("QUEST_NEEDLE_SPEC must point at the refquest spec_v1.json");
    let spec: NeedleSpec =
        serde_json::from_slice(&fs::read(&spec_path).expect("spec reads")).expect("spec parses");
    let lengths: Vec<usize> = env::var("QUEST_NEEDLE_LENGTHS")
        .unwrap_or_else(|_| "4096,8192,16384,32768".to_string())
        .split(',')
        .map(|token| token.trim().parse().expect("length"))
        .collect();

    let dir = dpo_dir();
    let config = load_config(&dir).expect("config");
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
    let mut model = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    for &target in &lengths {
        let base_words = lcg_words(
            &spec.words,
            (target * 2).max(4096),
            spec.filler_seed + target as u64,
        );
        let probe = base_words[..2000].join(" ");
        let ratio = encoded_len(&tokenizer, &probe) as f64 / 2000.0;

        let mut found_count = 0usize;
        let mut misses: Vec<String> = Vec::new();
        for key in &spec.keys {
            for &depth in &spec.depths {
                let needle = spec
                    .needle_template
                    .replace("{key}", &key.name)
                    .replace("{value}", &key.value);
                let question = spec.question_template.replace("{key}", &key.name);
                let overhead =
                    encoded_len(&tokenizer, &chat_wrap(&format!("{needle}\n\n{question}")));
                let mut count = (((target - overhead) as f64 / ratio) as i64).max(64);

                let mut content = String::new();
                for _ in 0..4 {
                    let words = &base_words[..(count as usize).min(base_words.len())];
                    let pre_n = (words.len() * depth / 100).clamp(1, words.len() - 1);
                    content = format!(
                        "{} {} {}\n\n{}",
                        words[..pre_n].join(" "),
                        needle,
                        words[pre_n..].join(" "),
                        question,
                    );
                    let actual = encoded_len(&tokenizer, &chat_wrap(&content)) as i64;
                    if (actual - target as i64).abs() <= 8.max(target as i64 / 200) {
                        break;
                    }
                    count += ((target as i64 - actual) as f64 / ratio) as i64;
                }

                let options = GenerateOptions::greedy(spec.gen_tokens);
                let mut generation = model
                    .generate(&tokenizer, &content, &options)
                    .expect("generation starts");
                let mut text = String::new();
                for step in generation.by_ref() {
                    text.push_str(&step.expect("step").chunk);
                }
                text.push_str(&generation.finish().rest);

                if text.contains(&key.value) {
                    found_count += 1;
                } else {
                    misses.push(format!("{}@{depth}% -> {text:?}", key.name));
                }
            }
        }
        let total = spec.keys.len() * spec.depths.len();
        println!("generation needle {target}: {found_count}/{total}");
        assert_eq!(found_count, total, "misses at {target}: {misses:?}");
    }
}
