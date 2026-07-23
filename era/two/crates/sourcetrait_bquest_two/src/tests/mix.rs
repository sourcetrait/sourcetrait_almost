//! MixRender units: the end-to-end render over a synthetic corpus
//! tree - verbatim text (newlines and quotes exercised through the
//! nuon round-trip), sorted walk, the dolma-field metadata.
use crate::*;

fn temp_root(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("bquest_mix_{tag}_{}", std::process::id()))
}

#[test]
fn mix_render_documents_are_verbatim_sorted_and_typed() {
    let root = temp_root("render");
    let corpus = root.join("corpus");
    std::fs::create_dir_all(corpus.join("nested")).expect("corpus dirs");
    let rust_text = "fn main() {\n    println!(\"hi \\\"there\\\"\");\n}\n";
    let markdown_text = "# Chapter\n\nProse with `code` and\nmultiple lines.\n";
    std::fs::write(corpus.join("z_last.rs"), rust_text).expect("rust file");
    std::fs::write(corpus.join("nested/a_first.md"), markdown_text).expect("md file");

    let spec_path = root.join("spec.nuon");
    let spec = lib::nu::Value::list(
        vec![lib::nu::Value::record(
            lib::nu::record! {
                "name" => v_str("probe"),
                "corpus_dir" => v_str(&corpus.display().to_string()),
                "repo" => v_str("example/probe"),
                "license" => v_str("apache-2.0"),
                "kind" => v_str("code"),
            },
            span(),
        )],
        span(),
    );
    lib::nu::save_value(&spec_path, &spec).expect("spec write");

    let out = root.join("out");
    mix_render(&MixRenderArgs {
        spec: spec_path,
        out: out.clone(),
        stamp: Some(String::from("2026-07-23")),
    })
    .expect("render");

    let documents = lib::nu::load_value(&out.join("documents_probe.nuon")).expect("documents");
    let documents_type =
        lib::nu::parse_typedef(&format!("list<{}>", crate::mix::MIX_DOCUMENT_TYPEDEF))
            .expect("documents typedef");
    lib::nu::conform(&documents, &documents_type).expect("documents conform");
    let rows = documents.as_list().expect("rows");
    assert_eq!(rows.len(), 2);

    // Sorted walk: nested/a_first.md precedes z_last.rs.
    let first = rows[0].as_record().expect("first row");
    let second = rows[1].as_record().expect("second row");
    let text_of = |record: &lib::nu::Record| {
        record.get("text").expect("text").as_str().expect("str").to_string()
    };
    let meta_of = |record: &lib::nu::Record, field: &str| {
        record
            .get("metadata")
            .expect("metadata")
            .as_record()
            .expect("record")
            .get(field)
            .expect(field)
            .as_str()
            .expect("str")
            .to_string()
    };
    assert_eq!(meta_of(first, "path"), "nested/a_first.md");
    assert_eq!(meta_of(second, "path"), "z_last.rs");
    assert_eq!(text_of(first), markdown_text, "markdown text must be verbatim");
    assert_eq!(text_of(second), rust_text, "rust text must be verbatim");
    assert_eq!(meta_of(first, "language"), "markdown");
    assert_eq!(meta_of(second, "language"), "rust");
    assert_eq!(meta_of(first, "repo"), "example/probe");
    assert_eq!(meta_of(first, "license"), "apache-2.0");
    assert_eq!(meta_of(first, "kind"), "code");
    assert_eq!(
        first.get("id").expect("id").as_str().expect("str"),
        "example/probe/nested/a_first.md"
    );
    assert_eq!(first.get("added").expect("added").as_str().expect("str"), "2026-07-23");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn fim_transform_is_seeded_deterministic_and_reassembles() {
    use crate::mix::{FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, SplitMix64, fim_transform};

    let text = "fn main() {\n    let naïve = \"héllo\";\n    println!(\"{naïve}\");\n}\n";
    let render_all = |seed: u64| -> Vec<String> {
        let mut rng = SplitMix64::new(seed);
        (0..64).map(|_| fim_transform(text, &mut rng)).collect()
    };
    let first_pass = render_all(299_792_458);
    let second_pass = render_all(299_792_458);
    assert_eq!(first_pass, second_pass, "same seed must render identically");

    let transformed: Vec<&String> =
        first_pass.iter().filter(|t| t.as_str() != text).collect();
    let rate = transformed.len() as f64 / first_pass.len() as f64;
    assert!(
        (0.25..=0.75).contains(&rate),
        "transform rate {rate} implausible for 0.5"
    );

    let mut psm_seen = 0usize;
    let mut spm_seen = 0usize;
    for rendered in &transformed {
        if let Some(rest) = rendered.strip_prefix(FIM_PREFIX) {
            // PSM: prefix | suffix-sentinel suffix | middle-sentinel middle.
            psm_seen += 1;
            let (prefix, rest) = rest.split_once(FIM_SUFFIX).expect("suffix sentinel");
            let (suffix, middle) = rest.split_once(FIM_MIDDLE).expect("middle sentinel");
            assert_eq!(format!("{prefix}{middle}{suffix}"), text, "PSM must reassemble");
        } else {
            // SPM: suffix-sentinel suffix | prefix-sentinel prefix |
            // middle-sentinel middle.
            spm_seen += 1;
            let rest = rendered.strip_prefix(FIM_SUFFIX).expect("SPM leads with suffix");
            let (suffix, rest) = rest.split_once(FIM_PREFIX).expect("prefix sentinel");
            let (prefix, middle) = rest.split_once(FIM_MIDDLE).expect("middle sentinel");
            assert_eq!(format!("{prefix}{middle}{suffix}"), text, "SPM must reassemble");
        }
    }
    assert!(psm_seen > 0 && spm_seen > 0, "both FIM modes must occur across 64 draws");
}

#[test]
fn chunk_and_shuffle_is_seeded_lossless_and_tail_dropping() {
    use crate::mix::{SplitMix64, chunk_and_shuffle};

    let stream: Vec<u32> = (0..107).collect();
    let seq_len = 9usize;
    let (first_chunks, dropped) = chunk_and_shuffle(&stream, seq_len, &mut SplitMix64::new(7));
    let (second_chunks, _) = chunk_and_shuffle(&stream, seq_len, &mut SplitMix64::new(7));
    assert_eq!(first_chunks, second_chunks, "same seed must shuffle identically");
    assert_eq!(first_chunks.len(), 10);
    assert_eq!(dropped, 7, "107 = 10 * 10 + 7 tail");
    assert!(first_chunks.iter().all(|chunk| chunk.len() == seq_len + 1));

    // Lossless over the kept region: the shuffled chunks re-sort to
    // the original stream prefix.
    let mut recovered: Vec<u32> = first_chunks.iter().flatten().copied().collect();
    recovered.sort_unstable();
    assert_eq!(recovered, (0..100).collect::<Vec<u32>>());

    // The order genuinely shuffles (10 chunks in original order has
    // probability ~1/3.6M per seed).
    let in_order = first_chunks
        .windows(2)
        .all(|pair| pair[0][0] < pair[1][0]);
    assert!(!in_order, "chunk order must not stay sorted under the shuffle");
}

#[test]
fn tokenize_documents_joins_with_eos_and_fims_code_only() {
    use crate::mix::{PackDocument, SplitMix64, tokenize_documents};

    let model_dir = lib::model_dir(lib::consts::DPO_MODEL_NAME).expect("model home");
    let tokenizer = lib::load_tokenizer(&model_dir).expect("tokenizer (checkpoint-backed)");
    let eos_id = tokenizer.token_to_id("<|endoftext|>").expect("eos id");
    let fim_prefix_id = tokenizer.token_to_id("<|fim_prefix|>").expect("fim prefix id");
    let fim_middle_id = tokenizer.token_to_id("<|fim_middle|>").expect("fim middle id");

    let documents = vec![
        PackDocument { text: String::from("let total = 40 + 2\n"), code: true },
        PackDocument { text: String::from("Prose stays untouched.\n"), code: false },
    ];

    // FIM off: two documents, each EOS-terminated, no sentinels.
    let mut rng = SplitMix64::new(1);
    let plain = tokenize_documents(&tokenizer, &documents, false, &mut rng).expect("plain");
    assert_eq!(plain.iter().filter(|id| **id == eos_id).count(), 2);
    assert_eq!(*plain.last().expect("stream"), eos_id);
    assert!(!plain.contains(&fim_prefix_id));

    // FIM on: across many seeded draws the CODE document transforms
    // (sentinel ids appear as single tokens) while repeating the
    // prose-only set never yields a sentinel.
    let mut fim_seen = false;
    for seed in 0..16 {
        let mut rng = SplitMix64::new(seed);
        let stream =
            tokenize_documents(&tokenizer, &documents, true, &mut rng).expect("fim stream");
        if stream.contains(&fim_prefix_id) {
            fim_seen = true;
            assert!(stream.contains(&fim_middle_id), "a FIM render carries all sentinels");
        }
    }
    assert!(fim_seen, "16 seeds at rate 0.5 must transform at least once");

    let prose_only = vec![PackDocument {
        text: String::from("Only prose here.\n"),
        code: false,
    }];
    for seed in 0..16 {
        let mut rng = SplitMix64::new(seed);
        let stream =
            tokenize_documents(&tokenizer, &prose_only, true, &mut rng).expect("prose stream");
        assert!(!stream.contains(&fim_prefix_id), "docs documents never FIM");
    }
}
