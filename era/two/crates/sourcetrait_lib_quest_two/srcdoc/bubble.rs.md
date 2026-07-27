# bubble.rs

The client harness that answers by running the mode under bubblewrap. It is
ONE implementation serving both testing and production: the world is a plain
parameter, so a lock constructs one and a real use points at the machine, and
what the locks exercise is what ships. There is no test double here to be
thrown away later, and no test-mode branch for a shipped path to skip.

## struct BubbleWorld

## fn argv

A FRESH PROCFS IS MOUNTED UNCONDITIONALLY, and that is forced rather than
chosen. It cost a failing lock to find, so it is written down here rather than
left to be rediscovered.

The verifier profile beside this one - `sandbox_run` in the harness iter tool -
mounts a directory of ordinary files OVER `/proc`, which is the better property:
a path the mock does not carry is absent entirely, so an unmocked read fails
loudly instead of quietly picking up the host's real value. That profile is
verified and correct for what it confines.

It cannot be used here, because the thing being confined is NUSHELL. Nushell
resolves its own executable through `/proc/self/exe` at startup, so a plain
directory mounted over `/proc` makes it panic - "no /proc/self/exe available.
Is /proc mounted?" - before a single line of the mode is parsed. The failure is
at startup rather than in the mode, so it presents as the sandbox being broken
rather than as the world being wrong.

So a fresh procfs is mounted and the mock is bound FILE BY FILE on top of it.
The trade is real and is the reason the neighbouring profile still exists
unchanged: an unmocked path now reads the SANDBOX's procfs rather than failing.
That is not a leak of host state - the procfs belongs to the sandbox - but a
mode can read something nobody wrote for it, which the other profile would have
caught.

## fn proc_binds

Only FILES, and only what the mock directory holds. A bind needs its target to
exist and `/proc` is not writable even inside the sandbox, so an entry the real
procfs lacks cannot be conjured - a world can replace what a kernel exposes and
cannot invent a new interface.

Sorted so one world produces one argument vector, which is what makes the
profile lockable at all.

## struct BubbleHarness

## fn script

Both channels arrive as NUON LITERALS, where the inside path pipes `$in` as a
real value. That is not an inconsistency to reconcile: an external process has
no pipeline to be handed, and NUON is valid nu literal syntax by construction,
which is the same one-grammar property the channel design rests on. The cost is
a parse the inside path does not pay.

The call is appended because a `<nu>` block DECLARES a def and does not call
one, so evaluating the body alone returns nothing.

## fn run

BOTH STREAMS ARE DRAINED BY THEIR OWN THREAD while the wait loop polls, and the
obvious alternative is a deadlock rather than a style preference. Reading the
pipes after waiting works until a mode writes more than a pipe buffer holds, at
which point the child blocks on the write, never exits, and the wait never
returns - so the bug appears only for outputs above a threshold nothing in the
code names.

The timeout exists because a generated pipeline can loop forever. A mode being
killed is reported as an ending rather than raised, since it says something
about the mode rather than about the machinery.

## impl ClientHarness for BubbleHarness

A mode that fails is a RESPONSE rather than an error, so the only `Err` this
produces is the sandbox failing to start at all. Questness turns a refusal into
a repair envelope the model reads, so a failing mode is part of the
conversation; a bubblewrap that will not launch is not.

## fn serve

THE DECLARED OUTPUT TYPE IS CHECKED HERE, and this is the one place it can be.
Nushell parse-checks a STATIC result against a def's output type but does not
enforce a dynamic one, so a body that builds its answer at runtime - through
`from nuon`, say - returns off its own signature without complaint. Without this
check that answer reaches the model as a well-formed wrong value, which is the
worst shape for it to arrive in.

The locks build the mismatching value dynamically for exactly that reason: a
literal of the wrong type is rejected by nushell's own parser instead, which
would prove nothing about this check.

## fn trimmed_reason

Exit code 127 is named because it is the one a reader most often meets and most
often misdiagnoses: it means the sandbox could not run the shell at all, which
is a machine problem rather than anything the mode did.
