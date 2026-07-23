use crate::evict::{evicted_mask_values, keep_set};
use crate::EvictionSettings;

#[test]
fn keep_set_is_identity_under_cap() {
    let scores = vec![0.0f32; 10];
    let keep = keep_set(&scores, 16, 4, 4);
    assert_eq!(keep, (0..10u32).collect::<Vec<_>>());
}

#[test]
fn keep_set_protects_sink_and_recent_and_keeps_top_middle() {
    // len 12, cap 8, sink 2, recent 2 -> middle budget 4 from indices
    // 2..10; the top-4 middle scores sit at 3, 5, 6, 9.
    let scores = vec![
        0.0, 0.0, // sink (protected regardless of score)
        0.1, 9.0, 0.2, 8.0, 7.0, 0.3, 0.4, 6.0, // middle
        0.0, 0.0, // recent (protected)
    ];
    let keep = keep_set(&scores, 8, 2, 2);
    assert_eq!(keep, vec![0, 1, 3, 5, 6, 9, 10, 11]);
}

#[test]
fn keep_set_breaks_ties_toward_older_entries() {
    // All middle scores equal: the keep-set prefers lower indices.
    let scores = vec![0.0f32; 12];
    let keep = keep_set(&scores, 8, 2, 2);
    assert_eq!(keep, vec![0, 1, 2, 3, 4, 5, 10, 11]);
}

#[test]
fn keep_set_is_ascending_and_capped() {
    let scores: Vec<f32> = (0..100).map(|i| ((i * 37) % 100) as f32).collect();
    let keep = keep_set(&scores, 24, 4, 8);
    assert_eq!(keep.len(), 24);
    assert!(keep.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(keep[..4].iter().enumerate().all(|(i, &v)| v == i as u32));
    assert!(keep[16..].iter().zip(92u32..100).all(|(&v, e)| v == e));
}

#[test]
fn evicted_mask_shows_store_fully_and_chunk_causally() {
    // q=3, store=4: columns 0..4 always visible; chunk block causal.
    let values = evicted_mask_values(3, 4);
    let total = 7;
    let visible = |row: usize, column: usize| values[row * total + column] == 0.0;
    for row in 0..3 {
        for column in 0..4 {
            assert!(visible(row, column));
        }
    }
    assert!(visible(0, 4) && !visible(0, 5) && !visible(0, 6));
    assert!(visible(1, 5) && !visible(1, 6));
    assert!(visible(2, 6));
}

#[test]
fn validate_rejects_degenerate_configs() {
    let no_middle = EvictionSettings {
        prefill_cap: 8,
        decode_cap: None,
        sink_keep: 4,
        recent_keep: 4,
    };
    assert!(no_middle.validate().is_err());
    let decode_over_prefill = EvictionSettings {
        prefill_cap: 4096,
        decode_cap: Some(8192),
        sink_keep: 4,
        recent_keep: 512,
    };
    assert!(decode_over_prefill.validate().is_err());
    let sane = EvictionSettings {
        prefill_cap: 8192,
        decode_cap: Some(2048),
        sink_keep: 4,
        recent_keep: 512,
    };
    assert!(sane.validate().is_ok());
}
