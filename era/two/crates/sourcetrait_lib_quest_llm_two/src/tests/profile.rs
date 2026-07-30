//! A2 observation instrument: the mass-partition unit (cpu,
//! checkpoint-free) and the profile battery driver (cuda).
#[cfg_attr(not(feature = "cuda"), allow(unused_imports))]
use crate::*;
use sourcetrait_lib_quest_harness_two::nu;
use crate::profile::ProfileAccum;

/// Hand-pinned partition: rows below the eligibility threshold are
/// ignored; eligible rows split their unit mass into sink (col 0),
/// middle [1, p-window+1), recent [p-window+1, ..].
#[test]
fn mass_partition_matches_hand_computation() {
    // 1 head, 2 rows, width 8; window 4. Row absolute positions 5, 6
    // (first_position 5): both eligible (p >= 5).
    // Row p=5: threshold col 2 -> sink {0}, middle {1}, recent {2..}.
    // Row p=6: threshold col 3 -> sink {0}, middle {1,2}, recent {3..}.
    let probs = candle_core::Tensor::from_vec(
        vec![
            0.10f32, 0.20, 0.30, 0.15, 0.05, 0.20, 0.0, 0.0, // row p=5
            0.05, 0.10, 0.25, 0.20, 0.10, 0.10, 0.20, 0.0, // row p=6
        ],
        (1, 2, 8),
        &candle_core::Device::Cpu,
    )
    .expect("probs");
    let mut accum = ProfileAccum::new(1, 4);
    accum.accumulate(&probs, 5, false).expect("accumulate");
    let report = accum.report(3);
    assert_eq!(report.layer_index, 3);
    assert_eq!(report.prefill_rows, 2);
    assert_eq!(report.decode_rows, 0);
    let head = &report.prefill[0];
    // sink: (0.10 + 0.05) / 2; middle: (0.20 + 0.35) / 2;
    // recent: (0.70 + 0.60) / 2.
    assert!((head.sink - 0.075).abs() < 1e-6, "sink {}", head.sink);
    assert!((head.middle - 0.275).abs() < 1e-6, "middle {}", head.middle);
    assert!((head.recent - 0.65).abs() < 1e-6, "recent {}", head.recent);
}

#[test]
fn ineligible_rows_accumulate_nothing() {
    let probs = candle_core::Tensor::from_vec(
        vec![0.5f32, 0.5, 0.0, 0.0],
        (1, 1, 4),
        &candle_core::Device::Cpu,
    )
    .expect("probs");
    let mut accum = ProfileAccum::new(1, 4);
    // p = 3 < window + 1 = 5 -> ineligible.
    accum.accumulate(&probs, 3, false).expect("accumulate");
    let report = accum.report(0);
    assert_eq!(report.prefill_rows, 0);
    assert_eq!(report.prefill[0].middle, 0.0);
}

/// The A2 observation battery: single-mode needle prompts, profile
/// armed, per-cell layer reports + a rows-weighted overall aggregate.
/// Env: QUEST_NEEDLE_SPEC + QUEST_PROFILE_OUT; optional
/// QUEST_PROFILE_LENGTHS (csv, default 16384,24576,32768),
/// QUEST_PROFILE_DEPTHS (csv, default 25,75), QUEST_PROFILE_WINDOW
/// (default 4096).
#[test]
#[cfg(feature = "cuda")]
#[ignore = "the A2 battery: needs the DPO checkpoint + a cuda card + QUEST_NEEDLE_SPEC/_PROFILE_OUT"]
fn attn_profile_battery() {
    use crate::needle::NeedleRig;

    let spec_path = env::var("QUEST_NEEDLE_SPEC").expect("QUEST_NEEDLE_SPEC");
    let out_path = env::var("QUEST_PROFILE_OUT").expect("QUEST_PROFILE_OUT");
    let lengths: Vec<usize> = env::var("QUEST_PROFILE_LENGTHS")
        .unwrap_or_else(|_| "16384,24576,32768".to_string())
        .split(',')
        .map(|token| token.trim().parse().expect("length"))
        .collect();
    let depths: Vec<usize> = env::var("QUEST_PROFILE_DEPTHS")
        .unwrap_or_else(|_| "25,75".to_string())
        .split(',')
        .map(|token| token.trim().parse().expect("depth"))
        .collect();
    let window: usize = env::var("QUEST_PROFILE_WINDOW")
        .unwrap_or_else(|_| "4096".to_string())
        .parse()
        .expect("window");

    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let spec = NeedleSpec::load(Path::new(&spec_path)).expect("spec loads");
    let config = load_config(&dir).expect("config");
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
    let mut model = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    // layer_index -> per-head rows-weighted sums, prefill + decode.
    struct OverallAccum {
        prefill_sums: Vec<[f64; 3]>,
        prefill_rows: u64,
        decode_sums: Vec<[f64; 3]>,
        decode_rows: u64,
    }
    let mut overall: HashMap<usize, OverallAccum> = HashMap::new();
    let span = nu::Span::unknown();
    let masses_value = |middle: f64, sink: f64, recent: f64| {
        nu::Value::record(
            nu::record! {
                "middle" => nu::Value::float(middle, span),
                "sink" => nu::Value::float(sink, span),
                "recent" => nu::Value::float(recent, span),
            },
            span,
        )
    };
    let layer_value = |layer: &LayerProfile| {
        nu::Value::record(
            nu::record! {
                "layer_index" => nu::Value::int(layer.layer_index as i64, span),
                "window" => nu::Value::int(layer.window as i64, span),
                "prefill_rows" => nu::Value::int(layer.prefill_rows as i64, span),
                "decode_rows" => nu::Value::int(layer.decode_rows as i64, span),
                "prefill" => nu::Value::list(
                    layer
                        .prefill
                        .iter()
                        .map(|masses| masses_value(masses.middle, masses.sink, masses.recent))
                        .collect(),
                    span,
                ),
                "decode" => nu::Value::list(
                    layer
                        .decode
                        .iter()
                        .map(|masses| masses_value(masses.middle, masses.sink, masses.recent))
                        .collect(),
                    span,
                ),
            },
            span,
        )
    };
    let mut cell_records: Vec<nu::Value> = Vec::new();
    let mut hits = 0usize;
    let mut total = 0usize;
    for &length in &lengths {
        for key_index in 0..spec.keys.len() {
            for &depth in &depths {
                let rig = NeedleRig {
                    spec: &spec,
                    tokenizer: &tokenizer,
                };
                let prompt = rig
                    .cell_prompt(NeedleMode::Single, length, key_index, depth)
                    .expect("cell prompt");
                model.arm_attn_profile(window);
                let options = GenerateOptions::greedy(spec.gen_tokens);
                let mut generation = model
                    .generate(&tokenizer, &prompt.content, &options)
                    .expect("generation");
                let mut response = String::new();
                for step in generation.by_ref() {
                    response.push_str(&step.expect("step").chunk);
                }
                response.push_str(&generation.finish().rest);
                let layers = model.take_attn_profile();
                let found = response.contains(&spec.keys[key_index].value);
                hits += usize::from(found);
                total += 1;
                for layer in &layers {
                    let entry = overall.entry(layer.layer_index).or_insert_with(|| {
                        OverallAccum {
                            prefill_sums: vec![[0.0; 3]; layer.prefill.len()],
                            prefill_rows: 0,
                            decode_sums: vec![[0.0; 3]; layer.decode.len()],
                            decode_rows: 0,
                        }
                    });
                    for (head, masses) in layer.prefill.iter().enumerate() {
                        entry.prefill_sums[head][0] += masses.middle * layer.prefill_rows as f64;
                        entry.prefill_sums[head][1] += masses.sink * layer.prefill_rows as f64;
                        entry.prefill_sums[head][2] += masses.recent * layer.prefill_rows as f64;
                    }
                    entry.prefill_rows += layer.prefill_rows;
                    for (head, masses) in layer.decode.iter().enumerate() {
                        entry.decode_sums[head][0] += masses.middle * layer.decode_rows as f64;
                        entry.decode_sums[head][1] += masses.sink * layer.decode_rows as f64;
                        entry.decode_sums[head][2] += masses.recent * layer.decode_rows as f64;
                    }
                    entry.decode_rows += layer.decode_rows;
                }
                cell_records.push(nu::Value::record(
                    nu::record! {
                        "length" => nu::Value::int(length as i64, span),
                        "key" => nu::Value::string(spec.keys[key_index].name.clone(), span),
                        "depth" => nu::Value::int(depth as i64, span),
                        "found" => nu::Value::bool(found, span),
                        "layers" => nu::Value::list(
                            layers.iter().map(&layer_value).collect(),
                            span,
                        ),
                    },
                    span,
                ));
                println!(
                    "{}",
                    nu::to_nuon_text(&nu::Value::record(
                        nu::record! {
                            "length" => nu::Value::int(length as i64, span),
                            "key" => nu::Value::string(
                                spec.keys[key_index].name.clone(),
                                span,
                            ),
                            "depth" => nu::Value::int(depth as i64, span),
                            "found" => nu::Value::bool(found, span),
                        },
                        span,
                    ))
                    .expect("nuon progress")
                );
            }
        }
    }

    let mut overall_records: Vec<nu::Value> = Vec::new();
    let mut layer_summaries: Vec<nu::Value> = Vec::new();
    let mut layer_indices: Vec<usize> = overall.keys().copied().collect();
    layer_indices.sort_unstable();
    for layer_index in layer_indices {
        let accum = &overall[&layer_index];
        let mean_values = |sums: &[[f64; 3]], rows: u64| -> nu::Value {
            let rows = if rows == 0 { 1 } else { rows } as f64;
            nu::Value::list(
                sums.iter()
                    .map(|sum| masses_value(sum[0] / rows, sum[1] / rows, sum[2] / rows))
                    .collect(),
                span,
            )
        };
        let middles: Vec<f64> = accum
            .prefill_sums
            .iter()
            .map(|sum| sum[0] / accum.prefill_rows.max(1) as f64)
            .collect();
        let (mut low, mut high, mut sum) = (f64::MAX, f64::MIN, 0.0);
        for middle in &middles {
            low = low.min(*middle);
            high = high.max(*middle);
            sum += middle;
        }
        layer_summaries.push(nu::Value::record(
            nu::record! {
                "layer_index" => nu::Value::int(layer_index as i64, span),
                "middle_mean" => nu::Value::float(sum / middles.len() as f64, span),
                "middle_min" => nu::Value::float(low, span),
                "middle_max" => nu::Value::float(high, span),
            },
            span,
        ));
        overall_records.push(nu::Value::record(
            nu::record! {
                "layer_index" => nu::Value::int(layer_index as i64, span),
                "prefill_rows" => nu::Value::int(accum.prefill_rows as i64, span),
                "decode_rows" => nu::Value::int(accum.decode_rows as i64, span),
                "prefill" => mean_values(&accum.prefill_sums, accum.prefill_rows),
                "decode" => mean_values(&accum.decode_sums, accum.decode_rows),
            },
            span,
        ));
    }
    println!(
        "{}",
        nu::to_nuon_text(&nu::Value::list(layer_summaries, span)).expect("nuon summary")
    );
    println!(
        "{}",
        nu::to_nuon_text(&nu::Value::record(
            nu::record! {
                "hits" => nu::Value::int(hits as i64, span),
                "cells" => nu::Value::int(total as i64, span),
            },
            span,
        ))
        .expect("nuon line")
    );

    let report = nu::Value::record(
        nu::record! {
            "window" => nu::Value::int(window as i64, span),
            "lengths" => nu::Value::list(
                lengths
                    .iter()
                    .map(|length| nu::Value::int(*length as i64, span))
                    .collect(),
                span,
            ),
            "depths" => nu::Value::list(
                depths
                    .iter()
                    .map(|depth| nu::Value::int(*depth as i64, span))
                    .collect(),
                span,
            ),
            "hits" => nu::Value::int(hits as i64, span),
            "cells" => nu::Value::int(total as i64, span),
            "overall" => nu::Value::list(overall_records, span),
            "per_cell" => nu::Value::list(cell_records, span),
        },
        span,
    );
    nu::save_value(Path::new(&out_path), &report).expect("nuon artifact");
    println!(
        "{}",
        nu::to_nuon_text(&nu::Value::record(
            nu::record! {
                "profile_artifact" => nu::Value::string(out_path, span),
            },
            span,
        ))
        .expect("nuon line")
    );
}
