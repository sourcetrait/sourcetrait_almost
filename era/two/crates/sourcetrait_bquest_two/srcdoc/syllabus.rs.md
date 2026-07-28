# syllabus.rs

The tool's half of the training-assets design, and it is deliberately thin.
The library discovers the tree, gates each fill against its contract,
renders, writes a file per stage and commits; what is left for a verb is
choosing which railroad the set lands in and saying where it went.

## fn syllabus_emit

What this does not do is the point of reading it. A rendered case is a
prompt and the record of what produced it. It is not a trainable example,
because nothing in the tree carries a gold answer, a rejected side or a
verifier - the library's own boundary stops short of stage semantics, and
no layer above it has been given the rule either. So this verb generates a
set, and the task generator still owns producing the artifacts the stage
loops actually consume. Reading the two as alternatives, and retiring that
generator on the strength of this, would leave the trainer with prompts and
no targets.

The railroad argument is optional in both directions on purpose. Absent
lays a fresh one, which is the ordinary generation. Present opens one
already laid, which is what makes a set editable in place and re-committed
as the next REV without rebuilding this binary - the property the railroad
exists for, and one a verb that could only lay would take away.

Re-emitting an unchanged tree into a railroad it already occupies fails,
and that is the library's contract rather than a rough edge here: a caller
committing nothing has a bug. It surfaces as git's own clean-tree message,
which git writes to stdout.

The summary carries the nom rather than only the directory, because the nom
is what a later session has to have written down to find the run again, and
a path holds it only until someone quotes the summary without it.
