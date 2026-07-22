//! SpeculationPort locks: the lookup-index + draft-policy unit pins
//! (era-one semantics carried verbatim), the cpu f32 spec-vs-plain
//! byte-equal gate (the standing exactness contract), and the cuda
//! echo smoke (report-only rates; bf16 trajectories may fork on
//! argmax ties - the settled kernel-path drift class).
use crate::*;
use crate::speculate::{DraftPolicy, LookupIndex, MAX_DRAFT};

#[test]
fn draft_finds_earlier_continuation() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2, 3, 9, 8, 7, 1, 2, 3]);
    // No 4-gram tail recurs; the 3-gram [1,2,3] first continued at
    // position 3.
    assert_eq!(index.draft(8), Some((3, vec![9, 8, 7, 1, 2, 3])));
}

#[test]
fn draft_respects_limit() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2, 3, 9, 8, 7, 1, 2, 3]);
    assert_eq!(index.draft(2), Some((3, vec![9, 8])));
    assert_eq!(index.draft(0), None);
}

#[test]
fn draft_none_when_tail_unseen() {
    let mut index = LookupIndex::new();
    index.extend(&[1, 2, 3, 4]);
    assert_eq!(index.draft(8), None);
}

#[test]
fn draft_none_below_min_gram() {
    let mut index = LookupIndex::new();
    index.extend(&[1]);
    assert_eq!(index.draft(8), None);
}

#[test]
fn draft_uses_previous_occurrence_when_tail_owns_latest() {
    let mut index = LookupIndex::new();
    // Every gram recurs; the tail's own insertion holds `latest`, so
    // drafting falls back to the previous continuation - via the
    // 4-gram level now.
    index.extend(&[5, 5, 5, 5, 5]);
    assert_eq!(index.draft(8), Some((4, vec![5])));
}

#[test]
fn draft_tracks_incremental_extends() {
    let mut index = LookupIndex::new();
    index.extend(&[10, 11, 12, 13]);
    index.extend(&[10, 11, 12]);
    // Tail [10,11,12] matches the opening gram, continuation 13
    // onward.
    assert_eq!(index.draft(8), Some((3, vec![13, 10, 11, 12])));
}

#[test]
fn four_gram_outranks_shorter_matches() {
    let mut index = LookupIndex::new();
    // The tail [6,7,8,9] recurs as a 4-gram (continuing 42, 41); a
    // decoy 3-gram [7,8,9] occurrence continues with 13 instead.
    // Longest match wins.
    index.extend(&[6, 7, 8, 9, 42, 41, 7, 8, 9, 13, 6, 7, 8, 9]);
    assert_eq!(index.draft(2), Some((4, vec![42, 41])));
}

#[test]
fn two_gram_floor_catches_short_repeats() {
    let mut index = LookupIndex::new();
    // Only the 2-gram tail [3,4] recurs; its earlier continuation
    // starts at 50.
    index.extend(&[3, 4, 50, 51, 52, 9, 3, 4]);
    assert_eq!(index.draft(8), Some((2, vec![50, 51, 52, 9, 3, 4])));
}

#[test]
fn policy_starts_at_max_shrinks_on_misses_tracks_accept_depth() {
    let mut policy = DraftPolicy::new();
    assert_eq!(policy.probe_limit(), MAX_DRAFT);
    policy.record(0);
    policy.record(0);
    assert_eq!(policy.probe_limit(), MAX_DRAFT / 4);
    // Tracking: the ceiling follows twice the observed accept depth.
    policy.record(1);
    assert_eq!(policy.probe_limit(), 2);
    policy.record(3);
    assert_eq!(policy.probe_limit(), 6);
    policy.record(MAX_DRAFT);
    assert_eq!(policy.probe_limit(), MAX_DRAFT);
}

#[test]
fn policy_pauses_after_cold_streak_and_resumes() {
    let mut policy = DraftPolicy::new();
    // A cold streak halves to the floor and trips the pause.
    let mut rounds = 0;
    while policy.probe_limit() > 0 {
        policy.record(0);
        rounds += 1;
        assert!(rounds < 32, "policy never paused");
    }
    // 64-round pause, then cheap probing resumes at the floor.
    for _ in 0..63 {
        assert_eq!(policy.probe_limit(), 0);
    }
    assert_eq!(policy.probe_limit(), 1);
}

#[test]
fn policy_seeds_length_by_matched_level() {
    let mut policy = DraftPolicy::new();
    // Hot (ceiling at MAX_DRAFT): every level drafts full.
    assert_eq!(policy.draft_limit(4), MAX_DRAFT);
    assert_eq!(policy.draft_limit(2), MAX_DRAFT);
    // Off the max: 3+ carries the ceiling, the 2-gram floor is short.
    policy.record(0);
    assert_eq!(policy.draft_limit(3), MAX_DRAFT / 2);
    assert_eq!(policy.draft_limit(2), 2);
    policy.record(2);
    assert_eq!(policy.draft_limit(3), 4);
}

fn spec_options(speculate: bool, sample_len: usize) -> GenerateOptions {
    GenerateOptions {
        temperature: None,
        top_p: None,
        sample_len,
        chat: false,
        ignore_stops: true,
        speculate,
        ..GenerateOptions::default()
    }
}

fn run_ids(
    model: &mut OlmoHybrid,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    options: &GenerateOptions,
) -> (Vec<u32>, GenerationReport) {
    let mut generation = model
        .generate(tokenizer, prompt, options)
        .expect("generation starts");
    let mut ids = Vec::new();
    for step in generation.by_ref() {
        ids.push(step.expect("step").token_id);
    }
    (ids, generation.finish())
}

/// The standing exactness contract: cpu f32 speculative output is
/// BYTE-EQUAL to plain greedy on a repetitive AND a novel prompt (the
/// partial-accept rollback + replay path rides the same chunked math
/// the chunked==stateless gate pins; f32 has been argmax-stable
/// across every prior reading). The repetitive leg must actually
/// accept drafts or the gate is vacuous.
#[test]
#[ignore = "needs the DPO checkpoint; ~5-8 min cpu"]
fn spec_gate_matches_plain_cpu_f32() {
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("config");
    let weights = mmap_weights(&dir, candle_core::DType::F32, &candle_core::Device::Cpu)
        .expect("mmap");
    let mut model = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    let legs: [(&str, &str, bool); 2] = [
        (
            "repetitive",
            "alpha beta gamma delta. alpha beta gamma delta. alpha beta gamma delta. alpha beta",
            true,
        ),
        ("novel", "Write one short sentence about mountains.", false),
    ];
    for (label, prompt, expect_accepts) in legs {
        let (plain_ids, plain) = run_ids(&mut model, &tokenizer, prompt, &spec_options(false, 40));
        let (spec_ids, spec) = run_ids(&mut model, &tokenizer, prompt, &spec_options(true, 40));
        println!(
            "spec gate {label}: plain {} ids, spec {} ids, drafted {}, accepted {}",
            plain_ids.len(),
            spec_ids.len(),
            spec.drafted_token_count,
            spec.accepted_draft_token_count
        );
        assert_eq!(spec_ids, plain_ids, "{label}: spec-vs-plain ids must be byte-equal");
        assert_eq!(plain.generated_token_count, 40, "{label}: forced length");
        if expect_accepts {
            assert!(
                spec.accepted_draft_token_count > 0,
                "{label}: the repetitive leg must accept drafts (gate is vacuous otherwise)"
            );
        }
    }
}

/// The cuda echo smoke (report-only): speculative vs plain rates on
/// an echo-shaped prompt at bf16. Ids are NOT asserted equal (bf16
/// argmax tie-forks are the settled class); acceptance must fire.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs the DPO checkpoint + a cuda card"]
fn spec_smoke_rates_cuda() {
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let config = load_config(&dir).expect("config");
    let device = candle_core::Device::new_cuda(0).expect("cuda");
    let weights = mmap_weights(&dir, candle_core::DType::BF16, &device).expect("mmap");
    let mut model = OlmoHybrid::new(&config, LibSettings::default(), weights).expect("model");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    let prompt = "Repeat the following sentence exactly, over and over, without stopping: \
        The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
    for (label, speculate) in [("plain", false), ("spec", true)] {
        let (ids, report) = run_ids(&mut model, &tokenizer, prompt, &spec_options(speculate, 256));
        let rate = ids.len() as f64 / report.decode_seconds.max(f64::MIN_POSITIVE);
        println!(
            "spec smoke {label}: {} tokens at {rate:.1} tok/s (drafted {}, accepted {})",
            ids.len(),
            report.drafted_token_count,
            report.accepted_draft_token_count
        );
        assert_eq!(report.generated_token_count, 256, "{label}: forced length");
        if speculate {
            assert!(
                report.accepted_draft_token_count > 0,
                "the echo smoke must accept drafts"
            );
        }
    }
}
