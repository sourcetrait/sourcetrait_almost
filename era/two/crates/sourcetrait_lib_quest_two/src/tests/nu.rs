//! NUON-core locks: typedef parsing (native oneof + the sugar
//! collapse), conformance semantics, text/file roundtrips, and the
//! single-line rendering the nuon-lines form leans on.
use crate::*;
use crate::nu::{
    Span,
    Value,
    append_line,
    conform,
    from_nuon_text,
    load_lines,
    load_value,
    parse_typedef,
    record,
    save_lines,
    save_value,
    to_nuon_text,
};

fn temp_root(tag: &str) -> PathBuf {
    let root = env::temp_dir().join(format!("lib_quest_nu_{tag}_{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root).expect("clean temp root");
    }
    fs::create_dir_all(&root).expect("create temp root");
    root
}

fn sample_record(count: i64) -> Value {
    Value::record(
        record! {
            "name" => Value::string(format!("row_{count}"), Span::unknown()),
            "count" => Value::int(count, Span::unknown()),
            "tags" => Value::list(
                vec![
                    Value::string("a", Span::unknown()),
                    Value::string("b", Span::unknown()),
                ],
                Span::unknown(),
            ),
        },
        Span::unknown(),
    )
}

#[test]
fn typedefs_parse_and_render_the_native_grammar() {
    let ty = parse_typedef("record<quiet: bool>").expect("record parses");
    assert_eq!(ty.to_string(), "record<quiet: bool>");
    let ty = parse_typedef("oneof<int, nothing>").expect("oneof parses");
    assert_eq!(ty.to_string(), "oneof<int, nothing>");
    let ty = parse_typedef("table<id: int, name: string>").expect("table parses");
    assert_eq!(ty.to_string(), "table<id: int, name: string>");
}

#[test]
fn sugar_collapses_to_base_except_glob() {
    // path/directory are valid SPELLINGS that collapse to string in
    // the Type enum; glob survives as its own type.
    let ty = parse_typedef("record<home: path, root: directory, pattern: glob>")
        .expect("sugar parses");
    assert_eq!(
        ty.to_string(),
        "record<home: string, root: string, pattern: glob>"
    );
}

#[test]
fn garbage_typedefs_reject() {
    assert!(parse_typedef("banana").is_err());
    assert!(parse_typedef("record<").is_err());
}

#[test]
fn conformance_is_open_and_deep_and_names_both_types() {
    let ty = parse_typedef("record<quiet: bool>").expect("typedef");
    conform(&from_nuon_text("{quiet: true}").expect("nuon"), &ty).expect("conforms");
    // Records are OPEN: extra fields pass (the native-trust posture).
    conform(&from_nuon_text("{quiet: false, extra: 1}").expect("nuon"), &ty)
        .expect("extra fields pass");
    let error = conform(&from_nuon_text("{quiet: 1}").expect("nuon"), &ty)
        .expect_err("wrong type rejects")
        .to_string();
    assert!(error.contains("declared record<quiet: bool>"), "got {error}");
    assert!(error.contains("derived"), "got {error}");
}

#[test]
fn oneof_accepts_any_member() {
    let ty = parse_typedef("oneof<record<a: int>, nothing>").expect("typedef");
    conform(&from_nuon_text("{a: 4}").expect("nuon"), &ty).expect("record member");
    conform(&from_nuon_text("null").expect("nuon"), &ty).expect("nothing member");
    assert!(conform(&from_nuon_text("4").expect("nuon"), &ty).is_err());
}

#[test]
fn sugar_scalars_and_cell_paths_carry() {
    let value = from_nuon_text("{wait: 2sec, size: 4kb, at: $.12.path}").expect("nuon");
    let ty = parse_typedef("record<wait: duration, size: filesize, at: cell-path>")
        .expect("typedef");
    conform(&value, &ty).expect("sugar values conform");
}

#[test]
fn nuon_text_roundtrips_and_renders_single_line() {
    let value = sample_record(3);
    let text = to_nuon_text(&value).expect("renders");
    assert!(
        !text.contains('\n'),
        "default rendering must be single-line, got {text:?}"
    );
    assert_eq!(from_nuon_text(&text).expect("reparses"), value);
}

#[test]
fn whole_value_files_roundtrip() {
    let root = temp_root("whole_value");
    let path = root.join("deep/dataset.nuon");
    let value = sample_record(7);
    save_value(&path, &value).expect("saves (parent created)");
    assert_eq!(load_value(&path).expect("loads"), value);
    let text = fs::read_to_string(&path).expect("raw");
    assert!(text.ends_with('\n'), "newline-terminated file");
    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn nuon_lines_roundtrip_and_append() {
    let root = temp_root("lines");
    let path = root.join("stream.nuon");
    let rows: Vec<Value> = (0..3).map(sample_record).collect();
    save_lines(&path, &rows).expect("saves");
    assert_eq!(load_lines(&path).expect("loads"), rows);
    append_line(&path, &sample_record(3)).expect("appends");
    let all = load_lines(&path).expect("reloads");
    assert_eq!(all.len(), 4);
    assert_eq!(all[3], sample_record(3));
    // Every line parses independently (the tail -f contract).
    let text = fs::read_to_string(&path).expect("raw");
    assert_eq!(text.lines().count(), 4);
    for line in text.lines() {
        from_nuon_text(line).expect("each line is one value");
    }
    fs::remove_dir_all(&root).expect("cleanup");
}
