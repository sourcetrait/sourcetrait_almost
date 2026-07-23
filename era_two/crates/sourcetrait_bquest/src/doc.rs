//! The self-documentation surface (DocCli): the always-moving
//! command tree as an eye-tree-style listing, so each era's surface
//! documents itself.
use crate::*;

/// One line per node - `name # summary` at one space per depth, the
/// grammar signature-block style: categories, topics, and actions
/// only, NO flag or parameter detail. Summaries are the clap about
/// strings (the derive doc comments), flattened to one line; the
/// implicit help node stays out.
pub(crate) fn render_cli_tree(command: &clap::Command) -> String {
    let mut out = String::new();
    for sub in command.get_subcommands() {
        render_node(sub, 0, &mut out);
    }
    out
}

fn render_node(command: &clap::Command, depth: usize, out: &mut String) {
    let name = command.get_name();
    if name == "help" {
        return;
    }
    let summary = command
        .get_about()
        .map(|about| about.to_string().replace('\n', " "))
        .unwrap_or_default();
    for _ in 0..depth {
        out.push(' ');
    }
    out.push_str(name);
    if !summary.is_empty() {
        out.push_str(" # ");
        out.push_str(&summary);
    }
    out.push('\n');
    for sub in command.get_subcommands() {
        render_node(sub, depth + 1, out);
    }
}

/// `bquest doc cli`: the whole tree on stdout.
pub(crate) fn doc_cli() -> BquestResult<()> {
    let command = <Cli as clap::CommandFactory>::command();
    print!("{}", render_cli_tree(&command));
    Ok(())
}
