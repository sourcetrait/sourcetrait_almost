# run.rs

One flat match over six verbs, with no global flagset above them.

That is the deliberate difference from bquest, and it follows from what the two
tools are. bquest resolves a model and settings through a profile framework, so
its verbs share a flagset. Every verb here instead takes an explicit checkpoint
choice or directory, because a baseline run is pinned by its own arguments and a
reading that inherited a profile would be a reading whose recipe is not in its
own command line.

The exit path prints to stderr and exits non-zero rather than returning to
`main`, which is what keeps `main` two lines and makes a failed baseline visible
to whatever is driving it.

## fn run
