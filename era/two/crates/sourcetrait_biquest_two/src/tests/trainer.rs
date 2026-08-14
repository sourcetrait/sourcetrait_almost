//! Organism-spec locks: geometry derivation, the layer cycle, the
//! config both loaders read, and the GDN gate-init ranges.
use crate::trainer::OrganismSpec;

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
