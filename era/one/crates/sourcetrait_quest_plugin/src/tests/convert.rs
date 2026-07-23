use crate::*;

#[test]
fn nuon_round_trips_a_record() {
    let value = parse_nuon("{quiet: true, count: 3}").unwrap();
    let text = render_nuon(&value).unwrap();
    let reparsed = parse_nuon(&text).unwrap();
    assert_eq!(value, reparsed);
}

#[test]
fn nuon_parse_failure_is_reported() {
    assert!(parse_nuon("{quiet: ").is_err());
}

#[test]
fn nuon_carries_sugar_scalars() {
    let value = parse_nuon("{wait: 2sec, size: 4kb}").unwrap();
    let ty = parse_typedef("record<wait: duration, size: filesize>").unwrap();
    assert!(conforms(&value, &ty));
}

#[test]
fn nuon_cell_path_literal() {
    // The cell-path output mode's load-bearing assumption: NUON can
    // carry a cell-path value (references into the model's own input).
    let value = parse_nuon("$.12.path").unwrap();
    let ty = parse_typedef("cell-path").unwrap();
    assert!(conforms(&value, &ty));
}
