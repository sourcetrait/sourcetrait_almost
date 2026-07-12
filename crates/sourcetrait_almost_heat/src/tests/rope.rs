use crate::config::HeatRopeScaling;
use crate::rope::{default_inv_freq, rope_tables, yarn_inv_freq};

const PINNED_ATTENTION_FACTOR: f64 = 1.2079441541679836;

fn pinned_scaling(attention_factor: Option<f64>) -> HeatRopeScaling {
    HeatRopeScaling {
        rope_type: String::from("yarn"),
        factor: Some(8.0),
        original_max_position_embeddings: Some(8192),
        beta_fast: Some(32.0),
        beta_slow: Some(1.0),
        attention_factor,
    }
}

#[test]
fn attention_factor_fallback_matches_pinned_value() {
    let (_, factor) = yarn_inv_freq(128, 500000.0, 65536, &pinned_scaling(None)).unwrap();
    assert!((factor - PINNED_ATTENTION_FACTOR).abs() < 1e-12);
}

#[test]
fn yarn_endpoints_blend_extrapolation_and_interpolation() {
    let (inv_freq, _) = yarn_inv_freq(128, 500000.0, 65536, &pinned_scaling(None)).unwrap();
    assert_eq!(inv_freq.len(), 64);
    assert!((inv_freq[0] - 1.0).abs() < 1e-12);
    let expected_last = 1.0 / (8.0 * 500000f64.powf(126.0 / 128.0));
    assert!(((inv_freq[63] - expected_last) / expected_last).abs() < 1e-12);
}

#[test]
fn tables_duplicate_freqs_and_carry_the_attention_factor()  {
    let inv_freq = default_inv_freq(4, 10000.0);
    let (cos, sin) = rope_tables(&inv_freq, 2.0, 3);
    // width 4 (= 2 freqs duplicated); position 0 row: cos=2 (scaled 1), sin=0.
    assert_eq!(cos.len(), 12);
    assert!((cos[0] - 2.0).abs() < 1e-6);
    assert!((cos[1] - 2.0).abs() < 1e-6);
    assert_eq!(cos[0], cos[2]);
    assert_eq!(cos[1], cos[3]);
    assert!(sin[..4].iter().all(|v| *v == 0.0));
    // position 1, freq 0 (=1.0): cos(1)*2, sin(1)*2, duplicated at index 2/3.
    assert!((cos[4] - (1f64.cos() * 2.0) as f32).abs() < 1e-6);
    assert!((sin[4] - (1f64.sin() * 2.0) as f32).abs() < 1e-6);
    assert_eq!(cos[4], cos[6]);
    assert_eq!(sin[5], sin[7]);
}
