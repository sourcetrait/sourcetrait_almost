//! Needle battery: checkpoint-free construction units + the
//! QualityGatedCuts artifact driver (per-cell JSON over single +
//! multi grids).
use crate::*;
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
/// lengths, per-cell JSON artifact. No asserts - this run PINS the
/// baseline; rungs are judged per-cell against its artifact.
/// Env: ALMOST_NEEDLE_SPEC (lastmost spec_v1.json), ALMOST_NEEDLE_OUT
/// (artifact path); optional ALMOST_NEEDLE_LENGTHS (csv),
/// ALMOST_NEEDLE_MODES (csv of single|multi), ALMOST_NEEDLE_SETTINGS
/// (a -s token: profile name or toml path; absent = embedded base).
#[test]
#[cfg(feature = "cuda")]
#[ignore = "the QualityGatedCuts battery: needs the DPO checkpoint + a cuda card + ALMOST_NEEDLE_SPEC/_OUT"]
fn needle_battery_artifact() {
    let spec_path = env::var("ALMOST_NEEDLE_SPEC").expect("ALMOST_NEEDLE_SPEC");
    let out_path = env::var("ALMOST_NEEDLE_OUT").expect("ALMOST_NEEDLE_OUT");
    let lengths: Vec<usize> = env::var("ALMOST_NEEDLE_LENGTHS")
        .unwrap_or_else(|_| "4096,8192,16384,24576,32768".to_string())
        .split(',')
        .map(|token| token.trim().parse().expect("length"))
        .collect();
    let modes: Vec<NeedleMode> = env::var("ALMOST_NEEDLE_MODES")
        .unwrap_or_else(|_| "single,multi".to_string())
        .split(',')
        .map(|token| match token.trim() {
            "single" => NeedleMode::Single,
            "multi" => NeedleMode::Multi,
            other => panic!("unknown needle mode {other:?}"),
        })
        .collect();
    let settings_token = env::var("ALMOST_NEEDLE_SETTINGS").ok();
    let settings = match &settings_token {
        Some(token) => LibSettings::load(Some(token)).expect("settings token"),
        None => LibSettings::default(),
    };

    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let spec = NeedleSpec::load(Path::new(&spec_path)).expect("spec loads");
    let config = load_config(&dir).expect("config");
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
    let mut model = OlmoHybrid::new(&config, settings, weights).expect("model");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

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
                "needle {} {length}: {found}/{}",
                mode.label(),
                of_length.len()
            );
        }
        cells.extend(mode_cells);
    }
    let report = serde_json::json!({
        "settings_token": settings_token,
        "lengths": lengths,
        "cells": cells,
    });
    fs::write(&out_path, serde_json::to_vec_pretty(&report).expect("json")).expect("write");
    println!("needle artifact -> {out_path}");
}
