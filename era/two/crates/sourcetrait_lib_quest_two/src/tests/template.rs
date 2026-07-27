//! Template locks: the binding name's sigil, the nu value bridge, and
//! the strictness the crate brings that jinja does not.
use crate::nu;
use crate::template::{
    binding_name,
    render,
};

fn bind(pass: &str, nuon: &str) -> Vec<(String, nu::Value)> {
    vec![(
        pass.to_string(),
        nu::from_nuon_text(nuon).expect("nuon parses"),
    )]
}

#[test]
fn a_binding_keeps_its_channel_name_and_loses_the_sigil() {
    // MEASURED: liquid 0.26 rejects `$` in a variable path, so the
    // channel is addressed by name. `$in` and `in` are the same
    // channel spelled for two different languages.
    assert_eq!(binding_name("$in"), "in");
    assert_eq!(binding_name("$args"), "args");
    assert_eq!(binding_name("in"), "in");
    assert_eq!(binding_name("  $in  "), "in");
}

#[test]
fn the_worked_example_renders_from_a_bound_record() {
    let bindings = bind(
        "$in",
        "{my_name: 'Quest', their_name: 'Roy', cwd: '/tmp/foo'}",
    );
    let text = render(
        "Hello, {{ in.their_name }}. My name is {{ in.my_name }}. We are \
         currently in the `{{ in.cwd }}` directory.",
        &bindings,
    )
    .expect("renders");
    assert_eq!(
        text,
        "Hello, Roy. My name is Quest. We are currently in the `/tmp/foo` directory."
    );
}

#[test]
fn both_channels_bind_at_once() {
    let bindings = vec![
        (
            String::from("$in"),
            nu::from_nuon_text("{n: 1}").expect("nuon"),
        ),
        (
            String::from("$args"),
            nu::from_nuon_text("[a, b]").expect("nuon"),
        ),
    ];
    let text = render("{{ in.n }}{{ args[1] }}", &bindings).expect("renders");
    assert_eq!(text, "1b");
}

#[test]
fn lists_and_nested_records_cross_the_bridge() {
    let bindings = bind("$in", "{rows: [{k: 1}, {k: 2}], flag: true, nil: null}");
    let text = render(
        "{% for row in in.rows %}{{ row.k }}{% endfor %}|{{ in.flag }}|{{ in.nil }}",
        &bindings,
    )
    .expect("renders");
    assert_eq!(text, "12|true|");
}

#[test]
fn sugar_scalars_carry_their_nuon_spelling() {
    // A duration is not a liquid type, and its NUON literal is the
    // form a reader of this program already expects. Asserted against
    // the renderer rather than against a spelling typed from memory.
    let value = nu::from_nuon_text("{wait: 2sec, size: 4kb}").expect("nuon");
    let nu::Value::Record { val, .. } = &value else {
        panic!("a record parses as a record");
    };
    let wait = nu::to_nuon_text(val.get("wait").expect("wait")).expect("renders");
    let size = nu::to_nuon_text(val.get("size").expect("size")).expect("renders");
    let bindings = vec![(String::from("$in"), value.clone())];
    let text = render("{{ in.wait }}|{{ in.size }}", &bindings).expect("renders");
    assert_eq!(text, format!("{wait}|{size}"));
}

#[test]
fn an_undefined_variable_is_an_error_rather_than_an_empty_render() {
    // The crate is strict where jinja renders empty, which is better
    // for a renderer whose output becomes training data.
    let bindings = bind("$in", "{n: 1}");
    let error = render("{{ missing.field }}", &bindings)
        .expect_err("strict")
        .to_string();
    assert!(error.contains("does not render"), "got {error}");
}

#[test]
fn a_malformed_template_is_reported_as_a_parse_failure() {
    let bindings = bind("$in", "{n: 1}");
    let error = render("{% for x in %}{% endfor %}", &bindings)
        .expect_err("malformed")
        .to_string();
    assert!(error.contains("does not parse"), "got {error}");
}
