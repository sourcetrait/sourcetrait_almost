use crate::*;
use crate::rope::{default_inv_freq, yarn_params};

const PINNED_ATTENTION_FACTOR: f64 = 1.2079441541679836;

fn pinned_scaling(attention_factor: Option<f64>, factor: Option<f64>) -> RopeScaling {
    RopeScaling {
        rope_type: String::from("yarn"),
        factor,
        original_max_position_embeddings: Some(8192),
        beta_fast: Some(32.0),
        beta_slow: Some(1.0),
        attention_factor,
    }
}

#[test]
fn attention_factor_fallback_matches_pinned_value() {
    let params = yarn_params(128, 500000.0, 65536, &pinned_scaling(None, Some(8.0))).unwrap();
    assert!((params.attention_factor - PINNED_ATTENTION_FACTOR).abs() < 1e-12);
}

#[test]
fn explicit_attention_factor_wins() {
    let params = yarn_params(128, 500000.0, 65536, &pinned_scaling(Some(1.5), Some(8.0))).unwrap();
    assert!((params.attention_factor - 1.5).abs() < 1e-12);
}

#[test]
fn absent_factor_recomputes_from_position_ratio() {
    let explicit = yarn_params(128, 500000.0, 65536, &pinned_scaling(None, Some(8.0))).unwrap();
    let recomputed = yarn_params(128, 500000.0, 65536, &pinned_scaling(None, None)).unwrap();
    assert_eq!(explicit.inv_freq, recomputed.inv_freq);
}

#[test]
fn yarn_endpoints_blend_extrapolation_and_interpolation() {
    let params = yarn_params(128, 500000.0, 65536, &pinned_scaling(None, Some(8.0))).unwrap();
    assert_eq!(params.inv_freq.len(), 64);
    // Frequency index 0 sits below the beta_fast correction bound: pure
    // extrapolation, so inv_freq = 1/theta^0 = 1.
    assert!((params.inv_freq[0] - 1.0).abs() < 1e-9);
    // The last index sits above the beta_slow bound: pure interpolation.
    let expected_last = (1.0 / (8.0 * 500000f64.powf(126.0 / 128.0))) as f32;
    let actual_last = *params.inv_freq.last().unwrap();
    assert!(
        ((actual_last - expected_last) / expected_last).abs() < 1e-6,
        "expected {expected_last}, got {actual_last}"
    );
}

#[test]
fn default_inv_freq_endpoints() {
    let inv_freq = default_inv_freq(128, 500000.0);
    assert_eq!(inv_freq.len(), 64);
    assert!((inv_freq[0] - 1.0).abs() < 1e-9);
    let expected_last = (1.0 / 500000f64.powf(126.0 / 128.0)) as f32;
    assert!(((inv_freq[63] - expected_last) / expected_last).abs() < 1e-6);
}
