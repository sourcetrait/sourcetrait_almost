//! Needle battery: checkpoint-free construction units + the
//! QualityGatedCuts artifact driver (per-cell JSON over single +
//! multi grids).
use crate::*;
#[cfg_attr(not(feature = "cuda"), allow(unused_imports))]
use sourcetrait_lib_quest_harness_two::nu;
use crate::needle::{cell_plan, lcg_words, splice};

#[test]
fn lcg_is_deterministic_and_pool_bounded() {
    let pool: Vec<String> = ["alpha", "beta", "gamma"]
        .iter()
        .map(|w| w.to_string())
        .collect();
    let first = lcg_words(&pool, 64, 7);
    let second = lcg_words(&pool, 64, 7);
    assert_eq!(first, second, "same seed, same stream");
    assert!(first.iter().all(|word| pool.contains(word)));
    let other_seed = lcg_words(&pool, 64, 8);
    assert_ne!(first, other_seed, "seed reaches the stream");
}

#[test]
fn splice_inserts_at_word_indices_in_order() {
    let words: Vec<String> = ["a", "b", "c", "d"].iter().map(|w| w.to_string()).collect();
    let spliced = splice(&words, &[(1, "X"), (3, "Y")]);
    assert_eq!(spliced, "a X b c Y d");
}

#[test]
fn multi_plan_offsets_depths_and_cycles_keys() {
    let plan = cell_plan(NeedleMode::Multi, 5, 4, 50);
    assert_eq!(plan, vec![(50, 4), (83, 0), (17, 1)]);
    let single = cell_plan(NeedleMode::Single, 5, 2, 10);
    assert_eq!(single, vec![(10, 2)]);
    // Distractor depths never collide with the target's.
    for depth in [0usize, 10, 25, 33, 50, 67, 75, 90] {
        let plan = cell_plan(NeedleMode::Multi, 5, 0, depth);
        assert_ne!(plan[1].0, depth);
        assert_ne!(plan[2].0, depth);
        assert_ne!(plan[1].0, plan[2].0);
    }
}

/// The QualityGatedCuts battery: single + multi grids at the era-one ratchet
/// lengths, per-cell whole-value NUON artifact (lib::nu; the
/// NUON-primary doctrine). No asserts - this run PINS the
/// baseline; rungs are judged per-cell against its artifact.
/// Progress and the artifact pointer print as NUON records (a
/// record per mode-length; {needle_artifact: path} last).
/// Env: QUEST_NEEDLE_SPEC (refquest spec_v1.json), QUEST_NEEDLE_OUT
/// (artifact path); optional QUEST_NEEDLE_LENGTHS (csv),
/// QUEST_NEEDLE_MODES (csv of single|multi), QUEST_NEEDLE_SETTINGS
/// (a -s token: profile name or toml path; absent = embedded base),
/// QUEST_NEEDLE_CONFIG (a -c token; rides the adapter-aware
/// load_weights path - absent = the standing mmap instrument).
#[test]
#[cfg(feature = "cuda")]
#[ignore = "the QualityGatedCuts battery: needs the DPO checkpoint + a cuda card + QUEST_NEEDLE_SPEC/_OUT"]
fn needle_battery_artifact() {
    let spec_path = env::var("QUEST_NEEDLE_SPEC").expect("QUEST_NEEDLE_SPEC");
    let out_path = env::var("QUEST_NEEDLE_OUT").expect("QUEST_NEEDLE_OUT");
    let lengths: Vec<usize> = env::var("QUEST_NEEDLE_LENGTHS")
        .unwrap_or_else(|_| "4096,8192,16384,24576,32768".to_string())
        .split(',')
        .map(|token| token.trim().parse().expect("length"))
        .collect();
    let modes: Vec<NeedleMode> = env::var("QUEST_NEEDLE_MODES")
        .unwrap_or_else(|_| "single,multi".to_string())
        .split(',')
        .map(|token| match token.trim() {
            "single" => NeedleMode::Single,
            "multi" => NeedleMode::Multi,
            other => panic!("unknown needle mode {other:?}"),
        })
        .collect();
    let settings_token = env::var("QUEST_NEEDLE_SETTINGS").ok();
    let settings = match &settings_token {
        Some(token) => LibSettings::load(Some(token)).expect("settings token"),
        None => LibSettings::default(),
    };
    let config_token = env::var("QUEST_NEEDLE_CONFIG").ok();

    let spec = NeedleSpec::load(Path::new(&spec_path)).expect("spec loads");
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let (dir, weights) = match &config_token {
        Some(token) => {
            let lib_config = LibConfig::load(Some(token)).expect("config token");
            let weights = load_weights(&lib_config, candle_core::DType::BF16, &device)
                .expect("adapter-aware load");
            (lib_config.model_dir(), weights)
        }
        None => {
            let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
            let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
            (dir, weights)
        }
    };
    let config = load_config(&dir).expect("config");
    let mut model = OlmoHybrid::new(&config, settings, weights).expect("model");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    let span = nu::Span::unknown();
    let mut cells = Vec::new();
    for mode in modes {
        let mode_cells =
            run_needle_cells(&mut model, &tokenizer, &spec, mode, &lengths).expect("cells");
        for &length in &lengths {
            let of_length: Vec<_> = mode_cells
                .iter()
                .filter(|cell| cell.target_length == length)
                .collect();
            let found = of_length.iter().filter(|cell| cell.found).count();
            println!(
                "{}",
                nu::to_nuon_text(&nu::Value::record(
                    nu::record! {
                        "mode" => nu::Value::string(mode.label(), span),
                        "target_length" => nu::Value::int(length as i64, span),
                        "found" => nu::Value::int(found as i64, span),
                        "cells" => nu::Value::int(of_length.len() as i64, span),
                    },
                    span,
                ))
                .expect("nuon progress")
            );
        }
        cells.extend(mode_cells);
    }
    let token_value = |token: &Option<String>| match token {
        Some(token) => nu::Value::string(token.clone(), span),
        None => nu::Value::nothing(span),
    };
    let cell_values: Vec<nu::Value> = cells
        .iter()
        .map(|cell| {
            nu::Value::record(
                nu::record! {
                    "mode" => nu::Value::string(cell.mode.clone(), span),
                    "target_length" => nu::Value::int(cell.target_length as i64, span),
                    "key" => nu::Value::string(cell.key.clone(), span),
                    "depth" => nu::Value::int(cell.depth as i64, span),
                    "prompt_tokens" => nu::Value::int(cell.prompt_tokens as i64, span),
                    "found" => nu::Value::bool(cell.found, span),
                    "distractor_hits" => nu::Value::list(
                        cell.distractor_hits
                            .iter()
                            .map(|hit| nu::Value::string(hit.clone(), span))
                            .collect(),
                        span,
                    ),
                    "response" => nu::Value::string(cell.response.clone(), span),
                },
                span,
            )
        })
        .collect();
    let report = nu::Value::record(
        nu::record! {
            "settings_token" => token_value(&settings_token),
            "config_token" => token_value(&config_token),
            "lengths" => nu::Value::list(
                lengths
                    .iter()
                    .map(|length| nu::Value::int(*length as i64, span))
                    .collect(),
                span,
            ),
            "cells" => nu::Value::list(cell_values, span),
        },
        span,
    );
    nu::save_value(Path::new(&out_path), &report).expect("nuon artifact");
    println!(
        "{}",
        nu::to_nuon_text(&nu::Value::record(
            nu::record! {
                "needle_artifact" => nu::Value::string(out_path, span),
            },
            span,
        ))
        .expect("nuon line")
    );
}
