//! Task-generator locks: the generated answers must verify, the
//! generated WRONG answers must not, and the whole run must be
//! reproducible from its seed.
use crate::*;

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir()
        .join(format!("bquest_taskgen_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    root
}

fn run(out: &Path, count: usize, seed: u64) {
    taskgen_nuon(&TaskgenNuonArgs {
        out: out.to_path_buf(),
        count,
        bench_every: 4,
        seed,
    })
    .expect("generation runs");
}

/// Every supervised answer must pass the very verifier the
/// reinforcement stage grades with, or the two stages disagree about
/// what correct means.
#[test]
fn generated_answers_verify_under_the_reinforcement_verifier() {
    let root = temp_root("verify");
    run(&root, 48, 7);

    let train = example::load_sft(&root.join("sft_train.nuon")).expect("sft loads");
    assert!(!train.is_empty());
    for messages in &train {
        let answer = messages
            .last()
            .and_then(|message| message.content.clone())
            .expect("an assistant turn");
        let reward = example::verify(example::VERIFIER_NUON, &answer, &answer)
            .expect("verifier runs");
        assert_eq!(reward, 1.0, "a supervised answer did not verify: {answer}");
    }

    // The reinforcement prompts carry the same answers as references.
    let prompts = example::load_rlvr(&root.join("rlvr_train.nuon")).expect("rlvr loads");
    assert!(!prompts.is_empty());
    for prompt in &prompts {
        assert_eq!(prompt.verifier, example::VERIFIER_NUON);
        lib::nu::from_nuon_text(&prompt.reference).expect("reference parses as NUON");
    }
    let _ = fs::remove_dir_all(&root);
}

/// The rejected side of every preference pair must actually be
/// wrong. A mutation that happened to round-trip would teach the
/// model away from a correct rendering.
#[test]
fn rejected_answers_are_actually_wrong() {
    let root = temp_root("rejected");
    run(&root, 48, 11);

    let pairs = example::load_dpo(&root.join("dpo_train.nuon")).expect("dpo loads");
    assert!(!pairs.is_empty(), "no preference pairs were produced");
    for pair in &pairs {
        assert_eq!(
            example::verify(example::VERIFIER_NUON, &pair.chosen, &pair.chosen)
                .expect("verifier runs"),
            1.0,
            "a chosen answer did not verify"
        );
        assert_eq!(
            example::verify(example::VERIFIER_NUON, &pair.rejected, &pair.chosen)
                .expect("verifier runs"),
            0.0,
            "a rejected answer verified as correct: {}",
            pair.rejected
        );
    }
    let _ = fs::remove_dir_all(&root);
}

/// The bench is split from the SAME generation, so it must be
/// disjoint from training and non-empty.
#[test]
fn the_bench_is_held_out_from_the_same_generation() {
    let root = temp_root("split");
    run(&root, 48, 13);

    let train = example::load_sft(&root.join("sft_train.nuon")).expect("train loads");
    let bench = example::load_sft(&root.join("sft_bench.nuon")).expect("bench loads");
    assert!(!bench.is_empty(), "the bench is empty");

    let prompt_of = |messages: &Vec<lib::ChatMessage>| {
        messages[0].content.clone().expect("a user turn")
    };
    let trained: HashSet<String> = train.iter().map(prompt_of).collect();
    for messages in &bench {
        assert!(
            !trained.contains(&prompt_of(messages)),
            "a bench prompt also appears in training"
        );
    }
    let _ = fs::remove_dir_all(&root);
}

/// The loss mask must cover EXACTLY the answer and its terminator.
/// This is the supervised path's whole safety property: a mask one
/// token wide of the span trains the model on the prompt, and one
/// token short drops the stop token, which is the class ai2's own
/// masking lost.
#[test]
fn the_supervised_mask_covers_exactly_the_answer() {
    let root = temp_root("mask");
    run(&root, 24, 17);
    let model = lib::model_dir(lib::consts::DPO_MODEL_NAME).expect("model home");
    let tokenizer = lib::load_tokenizer(&model).expect("tokenizer (checkpoint-backed)");
    let examples = example::load_sft(&root.join("sft_train.nuon")).expect("sft loads");
    assert!(!examples.is_empty());

    for messages in examples.iter().take(8) {
        let answer = messages
            .last()
            .and_then(|message| message.content.clone())
            .expect("an assistant turn");
        let (ids, mask) = example::encode_supervised(&tokenizer, messages).expect("encodes");
        assert_eq!(ids.len(), mask.len(), "mask and ids disagree in length");
        let masked: Vec<u32> = ids
            .iter()
            .zip(&mask)
            .filter(|(_, flag)| **flag != 0)
            .map(|(id, _)| *id)
            .collect();
        let decoded = tokenizer.decode(&masked, false).expect("decodes");
        assert_eq!(
            decoded,
            format!("{answer}<|endoftext|>"),
            "the mask does not cover exactly the answer plus its terminator"
        );
    }
    let _ = fs::remove_dir_all(&root);
}

/// One seed, one corpus - a generation that drifted would make a
/// bench reading incomparable against the run before it.
#[test]
fn generation_is_deterministic_in_its_seed() {
    let first = temp_root("seed_first");
    let second = temp_root("seed_second");
    run(&first, 24, 29);
    run(&second, 24, 29);
    for name in [
        "sft_train.nuon",
        "sft_bench.nuon",
        "dpo_train.nuon",
        "rlvr_train.nuon",
        "rlvr_bench.nuon",
    ] {
        assert_eq!(
            fs::read(first.join(name)).expect("first"),
            fs::read(second.join(name)).expect("second"),
            "{name} differed between two runs of the same seed"
        );
    }
    let _ = fs::remove_dir_all(&first);
    let _ = fs::remove_dir_all(&second);
}
