//! A3 eviction locks: keep-set semantics, the row gather, the
//! last-pass score math (cpu, checkpoint-free), and the gpu epoch
//! exactness gates (#[ignore]).
#[cfg_attr(not(feature = "cuda"), allow(unused_imports))]
use crate::*;
#[cfg_attr(not(feature = "cuda"), allow(unused_imports))]
use crate::evict::{
    OVERFLOW_SLACK,
    SCORE_SLICE,
    SCORE_TAIL,
    gather_rows,
    keep_indices,
    last_pass_scores,
};

#[test]
fn keep_indices_is_identity_under_the_cap() {
    let scores = vec![0.5f32; 6];
    assert_eq!(keep_indices(&scores, 8, 2, 1), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn keep_indices_protects_sink_and_recent_and_ranks_the_middle() {
    // len 10, cap 6, recent 2, sink 1: protected {0} + {8, 9}; middle
    // [1, 8) ranked by score -> budget 3 picks 3, 5, 6.
    let scores = vec![0.0f32, 0.1, 0.2, 0.9, 0.05, 0.8, 0.7, 0.3, 0.0, 0.0];
    assert_eq!(keep_indices(&scores, 6, 2, 1), vec![0, 3, 5, 6, 8, 9]);
}

#[test]
fn keep_indices_ties_break_toward_the_older_row() {
    // Middle rows 1..6 all equal; budget 2 -> the two OLDEST (1, 2).
    let scores = vec![0.0f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0];
    assert_eq!(keep_indices(&scores, 5, 2, 1), vec![0, 1, 2, 6, 7]);
}

#[test]
fn keep_indices_comes_back_ordered_and_capped() {
    let scores: Vec<f32> = (0..100).map(|i| ((i * 37) % 71) as f32).collect();
    let keep = keep_indices(&scores, 24, 8, 4);
    assert_eq!(keep.len(), 24);
    assert!(keep.windows(2).all(|pair| pair[0] < pair[1]), "ascending");
    assert!(keep[..4].iter().eq([0, 1, 2, 3].iter()), "sink prefix");
    assert!(
        keep[16..].iter().eq((92u32..100).collect::<Vec<_>>().iter()),
        "recent suffix"
    );
}

#[test]
fn gather_rows_packs_per_head_keep_sets() {
    // 2 heads, 4 valid rows, dim 2; distinct values per (head, row).
    let values: Vec<f32> = (0..2 * 5 * 2).map(|v| v as f32).collect();
    let buffer =
        candle_core::Tensor::from_vec(values, (2, 5, 2), &candle_core::Device::Cpu)
            .expect("buffer");
    let keep = vec![vec![0u32, 2], vec![1u32, 3]];
    let gathered = gather_rows(&buffer, 4, &keep).expect("gather");
    assert_eq!(gathered.dims(), [2, 2, 2]);
    let host = gathered
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host");
    // Head 0 rows 0, 2 -> [0, 1, 4, 5]; head 1 rows 1, 3 -> [12, 13, 16, 17].
    assert_eq!(host, vec![0.0, 1.0, 4.0, 5.0, 12.0, 13.0, 16.0, 17.0]);
}

#[test]
fn last_pass_scores_masks_causally_and_averages_the_tail() {
    // 1 head, dim 2, store len 4, tail 2 (positions 2 and 3). All-equal
    // keys make each row's mass uniform over its VISIBLE columns:
    // row@2 -> 1/3 on cols 0..2, 0 on col 3; row@3 -> 1/4 everywhere.
    // Mean: cols 0..2 = 7/24, col 3 = 1/8.
    let q = candle_core::Tensor::from_vec(
        vec![1.0f32, 0.0, 1.0, 0.0],
        (1, 2, 2),
        &candle_core::Device::Cpu,
    )
    .expect("q");
    let k = candle_core::Tensor::from_vec(
        vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        (1, 4, 2),
        &candle_core::Device::Cpu,
    )
    .expect("k");
    let scores = last_pass_scores(&q, &k, 1.0, SCORE_SLICE).expect("scores");
    assert_eq!(scores.dims(), [1, 4, 1]);
    let host = scores
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host");
    for (column, value) in host.iter().take(3).enumerate() {
        assert!(
            (value - 7.0 / 24.0).abs() < 1e-6,
            "col {column}: {value}"
        );
    }
    assert!((host[3] - 0.125).abs() < 1e-6, "col 3: {}", host[3]);
}

#[cfg(feature = "cuda")]
fn evict_settings(cap: usize) -> LibSettings {
    LibSettings {
        graph: true,
        graph_bucket_grain: 128,
        eviction: Some(EvictionSettings {
            decode_cap: cap,
            recent: 128,
            sink: 4,
            score_tail: SCORE_TAIL,
            score_slice: SCORE_SLICE,
        }),
        ..LibSettings::default()
    }
}

#[cfg(feature = "cuda")]
fn cuda_model(settings: LibSettings) -> OlmoHybrid {
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("config");
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
    OlmoHybrid::new(&config, settings, weights).expect("model")
}

#[cfg(feature = "cuda")]
fn prefill_ids(model: &mut OlmoHybrid, ids: &[u32]) -> u32 {
    model.clear_cache().expect("clear");
    let tensor = candle_core::Tensor::from_vec(ids.to_vec(), ids.len(), &model.device().clone())
        .expect("ids");
    let logits = model.forward_chunk(&tensor).expect("prefill");
    let last = logits
        .narrow(0, ids.len() - 1, 1)
        .expect("last")
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host");
    let mut best = 0usize;
    for (index, value) in last.iter().enumerate() {
        if *value > last[best] {
            best = index;
        }
    }
    best as u32
}

/// Captured-vs-uncaptured EXACTLY 0.0 across overflow epochs: both
/// legs drive the staged decode over the same fed tokens with the
/// same epoch schedule; the store cap holds throughout.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + a cuda card"]
fn evict_gate_epoch_exactness() {
    const CAP: usize = 512;
    const STEPS: usize = 200;
    let evict = EvictionSettings {
        decode_cap: CAP,
        recent: 128,
        sink: 4,
        score_tail: SCORE_TAIL,
        score_slice: SCORE_SLICE,
    };
    let mut model = cuda_model(evict_settings(CAP));
    let prompt: Vec<u32> = (0..1200u32).map(|i| 1000 + i * 3).collect();
    let feed: Vec<u32> = (0..STEPS as u32).map(|i| 2000 + i * 5).collect();

    let mut run_leg = |capture: bool| -> Vec<Vec<f32>> {
        let first = prefill_ids(&mut model, &prompt);
        let _ = first;
        model.evict_post_prefill(&evict).expect("post-prefill compaction");
        assert_eq!(model.context_len(), CAP, "compacted to the cap");
        model
            .arm_graph_decode(prompt.len() + STEPS + 1)
            .expect("arm");
        model.set_graph_capture(capture).expect("capture switch");
        let mut rows = Vec::with_capacity(STEPS);
        let mut epochs = 0usize;
        for token in &feed {
            if model.context_len() >= evict.decode_cap + OVERFLOW_SLACK {
                model.evict_overflow(&evict).expect("overflow epoch");
                epochs += 1;
                assert_eq!(model.context_len(), CAP, "epoch packs to the cap");
            }
            model.graph_disarm_when_full().expect("capacity check");
            assert!(model.graph_armed(), "capacity epoch must not fire");
            rows.push(
                model
                    .graph_decode_step(*token)
                    .expect("staged step")
                    .flatten_all()
                    .expect("flat")
                    .to_vec1::<f32>()
                    .expect("host"),
            );
            assert!(
                model.context_len() < CAP + OVERFLOW_SLACK + 1,
                "store cap held"
            );
        }
        assert!(epochs >= 2, "the drill must cross epochs (saw {epochs})");
        rows
    };

    let uncaptured = run_leg(false);
    let captured = run_leg(true);
    for (step, (a, b)) in captured.iter().zip(&uncaptured).enumerate() {
        assert!(
            a == b,
            "captured-vs-uncaptured differs at step {step} (must be exactly 0.0)"
        );
    }
}

/// The real driver path end-to-end: generate() with eviction + graph
/// (post-prefill compaction, staged decode, overflow epochs inside
/// Generation::step), forced decode past several epochs.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + a cuda card"]
fn evict_gate_generation_drill() {
    const CAP: usize = 512;
    const BUDGET: usize = 300;
    let mut model = cuda_model(evict_settings(CAP));
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");
    let prompt = "the cat sat on the mat ".repeat(220);
    let options = GenerateOptions {
        temperature: None,
        top_p: None,
        sample_len: BUDGET,
        chat: false,
        ignore_stops: true,
        ..GenerateOptions::default()
    };
    let mut generation = model
        .generate(&tokenizer, &prompt, &options)
        .expect("generation starts");
    for step in generation.by_ref() {
        step.expect("step");
    }
    let report = generation.finish();
    assert_eq!(report.generated_token_count, BUDGET, "forced decode length");
    assert!(
        report.prompt_token_count > CAP,
        "the drill wants a compacting prompt ({})",
        report.prompt_token_count
    );
    assert!(
        model.context_len() < CAP + OVERFLOW_SLACK + 1,
        "store cap held at the end ({})",
        model.context_len()
    );
}
