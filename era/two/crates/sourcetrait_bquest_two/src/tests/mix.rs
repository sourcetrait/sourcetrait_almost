//! MixRender units: the end-to-end render over a synthetic corpus
//! tree - verbatim text (newlines and quotes exercised through the
//! nuon round-trip), sorted walk, the dolma-field metadata.
use crate::*;

fn temp_root(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("bquest_mix_{tag}_{}", std::process::id()))
}

#[test]
fn mix_render_documents_are_verbatim_sorted_and_typed() {
    let root = temp_root("render");
    let corpus = root.join("corpus");
    std::fs::create_dir_all(corpus.join("nested")).expect("corpus dirs");
    let rust_text = "fn main() {\n    println!(\"hi \\\"there\\\"\");\n}\n";
    let markdown_text = "# Chapter\n\nProse with `code` and\nmultiple lines.\n";
    std::fs::write(corpus.join("z_last.rs"), rust_text).expect("rust file");
    std::fs::write(corpus.join("nested/a_first.md"), markdown_text).expect("md file");

    let spec_path = root.join("spec.nuon");
    let spec = lib::nu::Value::list(
        vec![lib::nu::Value::record(
            lib::nu::record! {
                "name" => v_str("probe"),
                "corpus_dir" => v_str(&corpus.display().to_string()),
                "repo" => v_str("example/probe"),
                "license" => v_str("apache-2.0"),
                "kind" => v_str("code"),
            },
            span(),
        )],
        span(),
    );
    lib::nu::save_value(&spec_path, &spec).expect("spec write");

    let out = root.join("out");
    mix_render(&MixRenderArgs {
        spec: spec_path,
        out: out.clone(),
        stamp: Some(String::from("2026-07-23")),
    })
    .expect("render");

    let documents = lib::nu::load_value(&out.join("documents_probe.nuon")).expect("documents");
    let documents_type =
        lib::nu::parse_typedef(&format!("list<{}>", crate::mix::MIX_DOCUMENT_TYPEDEF))
            .expect("documents typedef");
    lib::nu::conform(&documents, &documents_type).expect("documents conform");
    let rows = documents.as_list().expect("rows");
    assert_eq!(rows.len(), 2);

    // Sorted walk: nested/a_first.md precedes z_last.rs.
    let first = rows[0].as_record().expect("first row");
    let second = rows[1].as_record().expect("second row");
    let text_of = |record: &lib::nu::Record| {
        record.get("text").expect("text").as_str().expect("str").to_string()
    };
    let meta_of = |record: &lib::nu::Record, field: &str| {
        record
            .get("metadata")
            .expect("metadata")
            .as_record()
            .expect("record")
            .get(field)
            .expect(field)
            .as_str()
            .expect("str")
            .to_string()
    };
    assert_eq!(meta_of(first, "path"), "nested/a_first.md");
    assert_eq!(meta_of(second, "path"), "z_last.rs");
    assert_eq!(text_of(first), markdown_text, "markdown text must be verbatim");
    assert_eq!(text_of(second), rust_text, "rust text must be verbatim");
    assert_eq!(meta_of(first, "language"), "markdown");
    assert_eq!(meta_of(second, "language"), "rust");
    assert_eq!(meta_of(first, "repo"), "example/probe");
    assert_eq!(meta_of(first, "license"), "apache-2.0");
    assert_eq!(meta_of(first, "kind"), "code");
    assert_eq!(
        first.get("id").expect("id").as_str().expect("str"),
        "example/probe/nested/a_first.md"
    );
    assert_eq!(first.get("added").expect("added").as_str().expect("str"), "2026-07-23");

    let _ = std::fs::remove_dir_all(&root);
}
