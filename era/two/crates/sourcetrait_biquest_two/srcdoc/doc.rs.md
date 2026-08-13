# doc.rs

The command tree documents itself because the tree is never fixed. Categories
are re-organised as the tool grows, so any hand-written listing would drift, and
this renders from the parser instead.

The output deliberately matches the grammar signature block's shape - one space
per depth, `name # summary`, no separator characters - so a reader who knows one
listing can read the other. It is also why flags and parameters are absent:
depth plus a summary is what a reader needs to find a verb, and `--help` is
where its arguments live.

## fn render_cli_tree

## fn render_node

The clap `about` string is the summary, so the doc comments on the command enums
are this listing's content. A verb documented badly there reads badly here,
which is the pressure that keeps them short.

Newlines are flattened to spaces because a wrapped doc comment arrives with its
line breaks intact, and one of those would render its continuation at column
zero where the block's own grammar reads a top-level node. That is the same
hazard the signature block hit, in the same place.

The implicit `help` node is skipped rather than filtered afterwards, since clap
synthesises it into every level and it documents nothing about this tool.

## fn doc_cli
