//! Organism-spec locks: geometry derivation, the layer cycle, the
//! config both loaders read, and the corpus packer's contract.
use crate::bucket::BucketTable;
use crate::lexer::Segmenter;
use crate::trainer::OrganismSpec;
use crate::trainer::pack_corpus;
use crate::ucd::CharClass;
use crate::ucd::CharRow;
use crate::ucd::CharacterTable;

#[test]
fn geometry_derives_the_hybrid_ratios() {
    let spec = OrganismSpec::derive(174_944, 1024, 8).expect("derives");
    assert_eq!(spec.head_dim, 32);
    assert_eq!(spec.intermediate, 2816);
    assert_eq!(spec.key_head_dim, 24);
    assert_eq!(spec.value_head_dim, 48);
    assert_eq!(spec.key_width(), 768);
    assert_eq!(spec.value_width(), 1536);
    assert_eq!(spec.conv_kernel, 4);
}

#[test]
fn the_layer_cycle_is_three_gdn_then_attention() {
    let spec = OrganismSpec::derive(1000, 1024, 8).expect("derives");
    let kinds: Vec<bool> = (0..8).map(|layer| spec.is_attention(layer)).collect();
    assert_eq!(
        kinds,
        [false, false, false, true, false, false, false, true]
    );
    assert!(OrganismSpec::derive(1000, 1024, 6).is_err());
    assert!(OrganismSpec::derive(1000, 1000, 8).is_err());
}

/// A tiny lowercase-ascii table: newline, space, digits, a-z.
fn ascii_table() -> CharacterTable {
    let categories = vec![
        String::from("Cc"),
        String::from("Zs"),
        String::from("Nd"),
        String::from("Ll"),
    ];
    let mut rows = vec![
        CharRow { code_point: 0x0A, category: 0, script: u16::MAX, white_space: true, fold: 0x0A },
        CharRow { code_point: 0x20, category: 1, script: u16::MAX, white_space: true, fold: 0x20 },
    ];
    for digit in 0x30u32..=0x39 {
        rows.push(CharRow {
            code_point: digit,
            category: 2,
            script: u16::MAX,
            white_space: false,
            fold: digit,
        });
    }
    for letter in 0x61u32..=0x7A {
        rows.push(CharRow {
            code_point: letter,
            category: 3,
            script: u16::MAX,
            white_space: false,
            fold: letter,
        });
    }
    let table = CharacterTable::from_rows(rows, categories);
    assert_eq!(table.class_of('a'), CharClass::Word);
    assert_eq!(table.class_of(' '), CharClass::Unicode);
    table
}

/// Packing arithmetic: stride seq_len, chunks of seq_len + 1 sharing
/// their boundary token, the tail dropped, the token total reported.
#[test]
fn pack_corpus_strides_and_shares_boundaries() {
    let table = ascii_table();
    let buckets = BucketTable::new();
    let words = vec![String::from("cat"), String::from("dog")];
    let segmenter = Segmenter::new(&table, &buckets, &words);

    let dir = std::env::temp_dir().join(format!("biquest_pack_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("pack fixture dir");
    // "dog cat" alternating: 2 tokens per word + separator = 9 total.
    std::fs::write(dir.join("a.txt"), "dog cat dog cat 1")
        .expect("pack fixture file");

    let files = vec![dir.join("a.txt")];
    let (chunks, total) = pack_corpus(&segmenter, &files, 4).expect("pack");
    // dog, sp, cat, sp, dog, sp, cat, sp, 1 = 9 tokens.
    assert_eq!(total, 9);
    assert_eq!(chunks.len(), 2, "floor((9 - 1) / 4) chunks");
    for chunk in &chunks {
        assert_eq!(chunk.len(), 5, "chunks carry seq_len + 1 ids");
    }
    assert_eq!(
        chunks[0][4], chunks[1][0],
        "the boundary token closes one chunk and opens the next"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A refused file fails the whole pack, naming the file: the trainer
/// never silently drops corpus content.
#[test]
fn pack_corpus_refuses_rather_than_skips() {
    let table = ascii_table();
    let buckets = BucketTable::new();
    let words = vec![String::from("cat")];
    let segmenter = Segmenter::new(&table, &buckets, &words);

    let dir = std::env::temp_dir().join(format!("biquest_refuse_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("refuse fixture dir");
    std::fs::write(dir.join("bad.txt"), "cat Zebra").expect("refuse fixture file");

    let files = vec![dir.join("bad.txt")];
    let error = pack_corpus(&segmenter, &files, 4).expect_err("uppercase must refuse");
    assert!(
        error.to_string().contains("bad.txt"),
        "the refusal must name the file: {error}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_config_reads_back_through_the_engine_loader() {
    let spec = OrganismSpec::derive(174_944, 1024, 8).expect("derives");
    let text = serde_json::to_vec(&spec.config_json()).expect("renders");
    let lib: crate::llm::OlmoHybridConfig =
        serde_json::from_slice(&text).expect("the engine's config parses");
    lib.validate().expect("the engine's policy checks pass");
    assert!(lib.is_nope());
    assert!(lib.tie_word_embeddings);
    assert_eq!(lib.layer_kind(3), crate::llm::LayerKind::FullAttention);
    assert_eq!(lib.layer_kind(4), crate::llm::LayerKind::LinearAttention);
}
