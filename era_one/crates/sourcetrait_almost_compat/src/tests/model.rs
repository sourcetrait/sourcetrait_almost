use crate::model::banded_mask_values;

fn visible(values: &[f32], total: usize, row: usize, col: usize) -> bool {
    values[row * total + col] == 0.0
}

#[test]
fn causal_mask_with_offset_exposes_all_past() {
    // offset 2, q_len 2: rows are absolute positions 2 and 3 over 4 columns.
    let values = banded_mask_values(2, 2, None);
    assert_eq!(values.len(), 2 * 4);
    assert!(visible(&values, 4, 0, 0));
    assert!(visible(&values, 4, 0, 2));
    assert!(!visible(&values, 4, 0, 3));
    assert!(visible(&values, 4, 1, 3));
}

#[test]
fn sliding_mask_hides_out_of_window_keys() {
    // window 2: position p sees exactly positions p-1 and p.
    let values = banded_mask_values(0, 4, Some(2));
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
    // offset 3, q_len 1, window 2: the row is absolute position 3 and sees
    // absolute columns 2 and 3 only.
    let values = banded_mask_values(3, 1, Some(2));
    let total = 4;
    assert!(!visible(&values, total, 0, 0));
    assert!(!visible(&values, total, 0, 1));
    assert!(visible(&values, total, 0, 2));
    assert!(visible(&values, total, 0, 3));
}
