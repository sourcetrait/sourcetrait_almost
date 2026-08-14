//! DocCli locks: the rendered tree carries every category and
//! action with a summary, grammar-block indentation, no flag
//! detail, and no implicit-help noise.
use crate::*;
use crate::doc::render_cli_tree;

#[test]
fn cli_tree_lists_categories_and_actions_with_summaries() {
    let command = <Cli as clap::CommandFactory>::command();
    let tree = render_cli_tree(&command);
    let lines: Vec<&str> = tree.lines().collect();
    for expected in [
        "assemble # ",
        "associations # ",
        "disassemble # ",
        "doc # ",
        "matrix # ",
        "tokenize # ",
        "tokenizer # ",
        "trainer # ",
    ] {
        assert!(
            lines.iter().any(|line| line.starts_with(expected)),
            "category line missing: {expected:?}\n{tree}"
        );
    }
    for expected in [
        " cli # ",
        " ucd # ",
        " dictionary # ",
        " ledger # ",
        " build # ",
        " init # ",
    ] {
        assert!(
            lines.iter().any(|line| line.starts_with(expected)),
            "action line missing: {expected:?}\n{tree}"
        );
    }
}

#[test]
fn cli_tree_carries_no_help_flags_or_multiline_summaries() {
    let command = <Cli as clap::CommandFactory>::command();
    let tree = render_cli_tree(&command);
    for line in tree.lines() {
        assert!(
            !line.trim_start().starts_with("help"),
            "the implicit help node must stay out: {line:?}"
        );
        assert!(!line.contains("--"), "no flag detail: {line:?}");
        assert!(line.contains(" # "), "every line carries a summary: {line:?}");
    }
}
