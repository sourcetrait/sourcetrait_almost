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
    let spec = harness::nu::Value::list(
        vec![harness::nu::Value::record(
            harness::nu::record! {
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
    harness::nu::save_value(&spec_path, &spec).expect("spec write");

    let out = root.join("out");
    mix_render(&MixRenderArgs {
        spec: spec_path,
        out: out.clone(),
        stamp: Some(String::from("2026-07-23")),
    })
    .expect("render");

    let documents = harness::nu::load_value(&out.join("documents_probe.nuon")).expect("documents");
    let documents_type =
        harness::nu::parse_typedef(&format!("list<{}>", crate::mix::MIX_DOCUMENT_TYPEDEF))
            .expect("documents typedef");
    harness::nu::conform(&documents, &documents_type).expect("documents conform");
    let rows = documents.as_list().expect("rows");
    assert_eq!(rows.len(), 2);

    // Sorted walk: nested/a_first.md precedes z_last.rs.
    let first = rows[0].as_record().expect("first row");
    let second = rows[1].as_record().expect("second row");
    let text_of = |record: &harness::nu::Record| {
        record.get("text").expect("text").as_str().expect("str").to_string()
    };
    let meta_of = |record: &harness::nu::Record, field: &str| {
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
    use crate::mix::{FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, fim_transform};

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
    use crate::mix::chunk_and_shuffle;

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
    use crate::mix::{PackDocument, tokenize_documents};

    let model_dir = llm::model_dir(llm::consts::DPO_MODEL_NAME).expect("model home");
    let tokenizer = llm::load_tokenizer(&model_dir).expect("tokenizer (checkpoint-backed)");
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

/// The admixture-leg sampler: deterministic per seed, budget-
/// crossing semantics, provenance counts.
#[test]
fn mix_sample_is_deterministic_and_budget_crossing() {
    let dir = std::env::temp_dir().join(format!("bquest_mix_sample_{}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    let doc = |id: &str, text: &str| {
        harness::nu::Value::record(
            harness::nu::record! {
                "id" => v_str(id),
                "text" => v_str(text),
                "source" => v_str("toy/repo"),
                "added" => v_str(""),
                "created" => v_str(""),
                "metadata" => harness::nu::Value::record(
                    harness::nu::record! {
                        "repo" => v_str("toy/repo"),
                        "path" => v_str(id),
                        "language" => v_str("text"),
                        "license" => v_str("MIT"),
                        "kind" => v_str("code"),
                    },
                    span(),
                ),
            },
            span(),
        )
    };
    let table = harness::nu::Value::list(
        vec![
            doc("a", &"a".repeat(10)),
            doc("b", &"b".repeat(20)),
            doc("c", &"c".repeat(30)),
            doc("d", &"d".repeat(40)),
        ],
        span(),
    );
    let table_path = dir.join("documents_toy.nuon");
    harness::nu::save_value(&table_path, &table).expect("table write");

    let out_first = dir.join("sampled_first.nuon");
    let out_second = dir.join("sampled_second.nuon");
    let run = |out: &PathBuf| {
        mix_sample(&MixSampleArgs {
            documents: vec![table_path.clone()],
            out: out.clone(),
            budget_bytes: 35,
            seed: 7,
        })
        .expect("sample runs");
    };
    run(&out_first);
    run(&out_second);
    assert_eq!(
        fs::read(&out_first).expect("first"),
        fs::read(&out_second).expect("second"),
        "same seed must sample identically"
    );

    let sampled = harness::nu::load_value(&out_first).expect("sampled parses");
    let rows = sampled.as_list().expect("list").to_vec();
    let bytes: usize = rows
        .iter()
        .map(|row| {
            row.as_record()
                .ok()
                .and_then(|record| record.get("text"))
                .and_then(|text| text.as_str().ok())
                .map(str::len)
                .unwrap_or(0)
        })
        .sum();
    assert!(bytes >= 35, "sample stopped before the budget ({bytes} bytes)");
    assert!(rows.len() < 4, "the budget must not need every document");
    fs::remove_dir_all(&dir).ok();
}

/// The their-side ingest: a zstd dolma shard decodes into the same
/// document shape `mix render` produces for our own trees - text
/// verbatim, identity in metadata alone, a textless row counted
/// rather than fatal - and the byte budget overshoots on the crossing
/// document exactly as the sampler does.
#[test]
fn mix_rip_decodes_a_shard_into_verbatim_documents() {
    let root = temp_root("rip");
    fs::create_dir_all(&root).expect("root");

    let code = "fn main() {\n    println!(\"hi \\\"there\\\"\");\n}\n";
    let prose = "Prose with\nmultiple lines and a \"quote\".\n";
    let tail = "c".repeat(64);
    let quoted = |text: &str| serde_json::to_string(text).expect("json string");
    let payload = [
        format!(
            r#"{{"id": "up/one", "text": {}, "source": "upstream", "added": "2024-01-01"}}"#,
            quoted(code)
        ),
        format!(r#"{{"id": "up/two", "text": {}}}"#, quoted(prose)),
        // No text field at all, and text that is whitespace only:
        // both are counted and skipped rather than failing the shard.
        String::from(r#"{"id": "up/three", "metadata": {"upstream": 1}}"#),
        String::from(r#"{"id": "up/four", "text": "   \n  "}"#),
        // No id either - the fallback synthesizes one.
        format!(r#"{{"text": {}}}"#, quoted(&tail)),
    ]
    .join("\n")
        + "\n";
    let shard = root.join("shard.jsonl.zst");
    fs::write(
        &shard,
        zstd::encode_all(payload.as_bytes(), 3).expect("compress"),
    )
    .expect("shard write");

    let rip = |out: &Path, budget: usize| {
        mix_rip(&MixRipArgs {
            shard: shard.clone(),
            out: out.to_path_buf(),
            name: String::from("theirside"),
            hub: String::from("allenai/dolma3_dolmino_mix-100B-1125"),
            shard_path: None,
            kind: String::from("docs"),
            license: String::from("odc-by"),
            budget_bytes: budget,
            stamp: Some(String::from("2026-07-26")),
        })
        .expect("rip runs");
    };

    let out = root.join("documents_theirside.nuon");
    rip(&out, 10_000_000);

    let documents = harness::nu::load_value(&out).expect("documents");
    harness::nu::conform(
        &documents,
        &harness::nu::parse_typedef(&format!("list<{}>", crate::mix::MIX_DOCUMENT_TYPEDEF))
            .expect("typedef"),
    )
    .expect("documents conform");
    let rows = documents.as_list().expect("rows");
    assert_eq!(rows.len(), 3, "the two textless rows must not become documents");

    let record_of = |index: usize| rows[index].as_record().expect("row");
    let field_of = |index: usize, name: &str| {
        record_of(index).get(name).expect(name).as_str().expect("str").to_string()
    };
    let meta_of = |index: usize, name: &str| {
        record_of(index)
            .get("metadata")
            .expect("metadata")
            .as_record()
            .expect("record")
            .get(name)
            .expect(name)
            .as_str()
            .expect("str")
            .to_string()
    };

    // Text is the upstream bytes verbatim - nothing prepended, and
    // the newlines and quotes survive the nuon round-trip.
    assert_eq!(field_of(0, "text"), code);
    assert_eq!(field_of(1, "text"), prose);
    assert_eq!(field_of(2, "text"), tail);

    // Identity rides metadata; `source` carries the STREAM name, which
    // is what a wayside audit counts by - not the upstream's own
    // source field.
    assert_eq!(field_of(0, "id"), "up/one", "an upstream id is preserved");
    assert_eq!(
        field_of(2, "id"),
        "theirside/shard.jsonl.zst#5",
        "a row without an id gets one synthesized from its position"
    );
    for index in 0..3 {
        assert_eq!(field_of(index, "source"), "theirside");
        assert_eq!(field_of(index, "added"), "2026-07-26");
        assert_eq!(meta_of(index, "repo"), "allenai/dolma3_dolmino_mix-100B-1125");
        assert_eq!(meta_of(index, "path"), "shard.jsonl.zst");
        assert_eq!(meta_of(index, "license"), "odc-by");
        assert_eq!(
            meta_of(index, "kind"),
            "docs",
            "their side is never code - its code stream is already FIM-transformed"
        );
    }

    let provenance =
        harness::nu::load_value(&out.with_extension("provenance.nuon")).expect("provenance");
    let provenance = provenance.as_record().expect("provenance record");
    let count = |name: &str| provenance.get(name).expect(name).as_int().expect("int");
    assert_eq!(count("documents_read"), 5);
    assert_eq!(count("documents_taken"), 3);
    assert_eq!(count("documents_without_text"), 2);
    assert_eq!(
        count("text_bytes"),
        (code.len() + prose.len() + tail.len()) as i64
    );

    // The budget stops the READ: one byte of budget takes exactly the
    // first document, which overshoots it.
    let small = root.join("documents_small.nuon");
    rip(&small, 1);
    let rows = harness::nu::load_value(&small).expect("small");
    assert_eq!(
        rows.as_list().expect("rows").len(),
        1,
        "the crossing document overshoots and the rest are never read"
    );

    let _ = fs::remove_dir_all(&root);
}
