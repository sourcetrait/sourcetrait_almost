use crate::*;

#[test]
fn parses_spaced_record_typedef() {
    let ty = parse_typedef("record<quiet: bool>").unwrap();
    assert_eq!(render_typedef(&ty), "record<quiet: bool>");
}

#[test]
fn parses_native_oneof() {
    let ty = parse_typedef("oneof<int, nothing>").unwrap();
    assert_eq!(render_typedef(&ty), "oneof<int, nothing>");
}

#[test]
fn parses_nested_and_sugar_types() {
    let ty =
        parse_typedef("record<dog: table<id: int, path: glob>, when: duration>").unwrap();
    let rendered = render_typedef(&ty);
    assert!(rendered.starts_with("record<dog: table<"), "got {rendered}");
    assert!(rendered.contains("when: duration"), "got {rendered}");
}

#[test]
fn rejects_unknown_type_names() {
    assert!(parse_typedef("banana").is_err());
}

#[test]
fn conformance_is_open_and_deep() {
    let ty = parse_typedef("record<quiet: bool>").unwrap();
    assert!(conforms(&parse_nuon("{quiet: true}").unwrap(), &ty));
    // Records are OPEN: extra fields pass (the native-trust posture).
    assert!(conforms(&parse_nuon("{quiet: false, extra: 1}").unwrap(), &ty));
    assert!(!conforms(&parse_nuon("{quiet: 1}").unwrap(), &ty));
    assert!(!conforms(&parse_nuon("{loud: true}").unwrap(), &ty));
}

#[test]
fn oneof_accepts_any_member() {
    let ty = parse_typedef("oneof<record<a: int>, nothing>").unwrap();
    assert!(conforms(&parse_nuon("{a: 4}").unwrap(), &ty));
    assert!(conforms(&parse_nuon("null").unwrap(), &ty));
    assert!(!conforms(&parse_nuon("4").unwrap(), &ty));
}

#[test]
fn empty_list_binds_any_list_or_table() {
    let table_ty = parse_typedef("table<id: int>").unwrap();
    assert!(conforms(&parse_nuon("[]").unwrap(), &table_ty));
    let list_ty = parse_typedef("list<duration>").unwrap();
    assert!(conforms(&parse_nuon("[]").unwrap(), &list_ty));
}

#[test]
fn derives_table_from_uniform_record_list() {
    let derived = derive_type(&parse_nuon("[[id, name]; [1, \"a\"], [2, \"b\"]]").unwrap());
    assert_eq!(render_typedef(&derived), "table<id: int, name: string>");
}

#[test]
fn derivation_round_trips_through_parse() {
    // Value -> typedef string -> Type accepts the value back: the
    // smart command's input-def derivation contract.
    let value = parse_nuon("{quiet: true, paths: [\"a\", \"b\"], count: 3}").unwrap();
    let rendered = render_typedef(&derive_type(&value));
    let reparsed = parse_typedef(&rendered).unwrap();
    assert!(conforms(&value, &reparsed), "typedef was {rendered}");
}
