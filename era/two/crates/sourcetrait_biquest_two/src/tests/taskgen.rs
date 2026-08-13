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
    taskgen_all(&TaskgenAllArgs {
        out: out.to_path_buf(),
        count,
        bench_every: 4,
        seed,
    })
    .expect("generation runs");
}

/// The verifier a family's prompt implies, recovered the same way an
/// auditor would: from the instruction. The artifacts do not carry a
/// verifier on the supervised side, and adding one would be surface
/// the trainer never reads.
fn verifier_for(prompt: &str) -> &'static str {
    if prompt.contains("JSON to NUON") {
        example::VERIFIER_NUON
    } else if prompt.contains("shell command to nushell")
        || prompt.contains("Write a nushell pipeline")
    {
        example::VERIFIER_NU_VALUE
    } else {
        example::VERIFIER_EXACT
    }
}

/// Every supervised answer must pass the very verifier the
/// reinforcement stage grades with, or the two stages disagree about
/// what correct means. The two artifacts are generated in lockstep,
/// so pairing them by index also proves they did not drift.
#[test]
fn generated_answers_verify_under_the_reinforcement_verifier() {
    nu_sandbox::require_sandbox().expect("bubblewrap is present");
    let root = temp_root("verify");
    run(&root, 12, 7);

    let train = example::load_sft(&root.join("sft_train.nuon")).expect("sft loads");
    let graded = example::load_rlvr(&root.join("rlvr_train.nuon")).expect("rlvr loads");
    assert!(!train.is_empty());
    assert_eq!(
        train.len(),
        graded.len(),
        "the supervised and reinforcement artifacts fell out of step"
    );

    for (messages, prompt) in train.iter().zip(&graded) {
        let asked = messages[0].content.clone().expect("a user turn");
        let answer = messages
            .last()
            .and_then(|message| message.content.clone())
            .expect("an assistant turn");
        assert_eq!(
            verifier_for(&asked),
            prompt.verifier,
            "the paired rows describe different families"
        );
        let reward = example::verify(&prompt.verifier, &answer, &prompt.reference)
            .expect("verifier runs");
        assert_eq!(
            reward, 1.0,
            "a supervised answer did not verify under {}: {answer}",
            prompt.verifier
        );
    }

    // Every reinforcement prompt names a KNOWN verifier, and the
    // value-comparing ones carry a reference that actually parses.
    let prompts = example::load_rlvr(&root.join("rlvr_train.nuon")).expect("rlvr loads");
    assert!(!prompts.is_empty());
    let mut families = HashSet::new();
    for prompt in &prompts {
        families.insert(prompt.verifier.clone());
        if prompt.verifier == example::VERIFIER_NUON
            || prompt.verifier == example::VERIFIER_NU_VALUE
        {
            harness::nu::from_nuon_text(&prompt.reference).expect("reference parses as NUON");
        }
    }
    assert!(
        families.len() >= 2,
        "the families should exercise more than one verifier, got {families:?}"
    );
    let _ = fs::remove_dir_all(&root);
}

/// The sandbox is the only thing standing between a verifier and
/// arbitrary generated code, so its confinement is asserted rather
/// than assumed: no network, and a runaway pipeline is killed.
#[test]
fn the_sandbox_confines_what_it_runs() {
    nu_sandbox::require_sandbox().expect("bubblewrap is present");

    let networked = nu_sandbox::run_nu(
        "http get https://example.com | describe",
        std::time::Duration::from_secs(20),
    )
    .expect("the run completes");
    assert!(!networked.ok, "the sandbox allowed a network call");

    let looping = nu_sandbox::run_nu("loop { }", std::time::Duration::from_secs(2))
        .expect("the run completes");
    assert!(looping.timed_out, "a non-terminating pipeline was not killed");
    assert!(!looping.ok);

    // And a well-behaved pipeline still comes back as a VALUE.
    let value = nu_sandbox::pipeline_value("[1 2 3] | where {|x| $x > 1 }")
        .expect("the run completes")
        .expect("a value came back");
    assert_eq!(value, harness::nu::from_nuon_text("[2, 3]").expect("nuon"));
}

/// The error family's answer must come from nushell's own diagnostic.
/// Asserting the mutation site instead would be wrong for a whole
/// class: an unclosed brace is reported where the parser reaches end
/// of input, not where the brace was opened.
#[test]
fn the_error_line_is_the_one_nushell_reports() {
    nu_sandbox::require_sandbox().expect("bubblewrap is present");
    let root = temp_root("errors");
    run(&root, 12, 23);
    let examples = example::load_sft(&root.join("sft_train.nuon")).expect("sft loads");

    let mut checked = 0usize;
    for messages in &examples {
        let prompt = messages[0].content.clone().expect("a user turn");
        if !prompt.contains("This nushell fails") {
            continue;
        }
        let snippet = prompt.split_once("\n\n").expect("a payload").1.to_string();
        let answer = messages
            .last()
            .and_then(|message| message.content.clone())
            .expect("an assistant turn");
        let outcome = nu_sandbox::run_nu(&snippet, std::time::Duration::from_secs(10))
            .expect("the run completes");
        assert!(!outcome.ok, "a snippet taught as broken ran cleanly:\n{snippet}");
        let reported = outcome
            .stderr
            .split_once("[source:")
            .and_then(|(_, rest)| rest.split_once(':'))
            .map(|(line, _)| line.to_string())
            .expect("nushell reported a location");
        assert_eq!(reported, answer, "the taught line is not the reported one");
        checked += 1;
    }
    assert!(checked > 0, "no error-location examples were generated");
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
        let asked = pair.prompt[0].content.clone().expect("a user turn");
        let verifier = verifier_for(&asked);
        // For an execution-graded family the reference is the VALUE
        // the chosen pipeline produces, re-derived here rather than
        // taken on trust from the generator.
        let reference = if verifier == example::VERIFIER_NU_VALUE {
            let value = nu_sandbox::pipeline_value(&pair.chosen)
                .expect("the chosen pipeline runs")
                .expect("the chosen pipeline produced a value");
            harness::nu::to_nuon_text(&value).expect("renders")
        } else {
            pair.chosen.clone()
        };
        assert_eq!(
            example::verify(verifier, &pair.chosen, &reference).expect("verifier runs"),
            1.0,
            "a chosen answer did not verify under {verifier}"
        );
        assert_eq!(
            example::verify(verifier, &pair.rejected, &reference).expect("verifier runs"),
            0.0,
            "a rejected answer verified as correct under {verifier}: {}",
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

    let prompt_of = |messages: &Vec<llm::ChatMessage>| {
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
    let model = llm::model_dir(llm::consts::DPO_MODEL_NAME).expect("model home");
    let tokenizer = llm::load_tokenizer(&model).expect("tokenizer (checkpoint-backed)");
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
