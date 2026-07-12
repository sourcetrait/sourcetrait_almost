use crate::model::{banded_mask_values, sliding_trim_bounds};

fn visible(values: &[f32], total: usize, row: usize, col: usize) -> bool {
    values[row * total + col] == 0.0
}

#[test]
fn causal_mask_with_offset_exposes_all_past() {
    // offset 2, q_len 2, kv_start 0: rows are absolute positions 2 and 3
    // over 4 columns.
    let values = banded_mask_values(2, 2, 0, None);
    assert_eq!(values.len(), 2 * 4);
    assert!(visible(&values, 4, 0, 0));
    assert!(visible(&values, 4, 0, 2));
    assert!(!visible(&values, 4, 0, 3));
    assert!(visible(&values, 4, 1, 3));
}

#[test]
fn sliding_mask_hides_out_of_window_keys() {
    // window 2: position p sees exactly positions p-1 and p.
    let values = banded_mask_values(0, 4, 0, Some(2));
    let total = 4;
    assert!(visible(&values, total, 0, 0));
    assert!(!visible(&values, total, 0, 1));
    assert!(visible(&values, total, 3, 2));
    assert!(visible(&values, total, 3, 3));
    assert!(!visible(&values, total, 3, 1));
    assert!(!visible(&values, total, 3, 0));
}

#[test]
fn sliding_mask_with_offset_maps_columns_absolutely() {
    // offset 3, q_len 1, window 2, untrimmed (kv_start 0): the row is
    // absolute position 3 and sees absolute columns 2 and 3 only.
    let values = banded_mask_values(3, 1, 0, Some(2));
    let total = 4;
    assert!(!visible(&values, total, 0, 0));
    assert!(!visible(&values, total, 0, 1));
    assert!(visible(&values, total, 0, 2));
    assert!(visible(&values, total, 0, 3));
}

#[test]
fn trimmed_mask_columns_start_at_the_cache_head() {
    // window 4, offset 6: a trimmed cache holds min(6, 3) = 3 entries, so
    // kv_start = 6 - 3 = 3 and columns are absolute positions 3..9. Row 0
    // is the query at position 6: within-window keys are 3..=6.
    let kv_start = 3;
    let values = banded_mask_values(6, 3, kv_start, Some(4));
    let total = 6 + 3 - kv_start;
    assert_eq!(values.len(), 3 * total);
    assert!(visible(&values, total, 0, 0)); // key 3 = p-3, on the band edge
    assert!(visible(&values, total, 0, 3)); // key 6 = p itself
    assert!(!visible(&values, total, 0, 4)); // key 7 is the future
    // Row 2 is position 8: window drops keys 3 and 4.
    assert!(!visible(&values, total, 2, 0));
    assert!(!visible(&values, total, 2, 1));
    assert!(visible(&values, total, 2, 2)); // key 5 = p-3
    assert!(visible(&values, total, 2, 5)); // key 8 = p
}

#[test]
fn trim_bounds_cap_at_window_minus_one() {
    assert_eq!(sliding_trim_bounds(2, 4), None); // under the cap
    assert_eq!(sliding_trim_bounds(3, 4), None); // exactly the cap
    assert_eq!(sliding_trim_bounds(4, 4), Some((1, 3))); // one over
    assert_eq!(sliding_trim_bounds(10, 4), Some((7, 3)));
    // The decode-step shape: a cache at the cap grows by one, trims back.
    assert_eq!(sliding_trim_bounds(4096, 4096), Some((1, 4095)));
}
