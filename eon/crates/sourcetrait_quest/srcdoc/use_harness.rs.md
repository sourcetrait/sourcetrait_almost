# use_harness.rs

## const SANDBOX_BIN

## const SHELL_BIN

## const PROC

## const DEFAULT_TIMEOUT

## const POLL

## struct QuestWorld

The world is a plain parameter rather than a test mode so the code the
locks exercise is the code that ships: tests construct a world and vary
it between turns, production points the same harness at the real
machine. The daemon-side sibling of this idea is the engine arriving
through a factory.

### fn argv

`--proc` is forced rather than optional because every alternative was
measured to fail: a directory mounted over `/proc` stops nushell
resolving `/proc/self/exe` and it panics before parsing anything, and
the read-only `/proc` that `--ro-bind / /` carries along breaks
differently again. The failure reads as a broken sandbox rather than a
wrong world, which is why it is forced here instead of documented.

## struct UseHarness

The client-side half of the two-harness split: ThinkHarness owns the
daemon-side turn, this runs what the model asked the CALLER to run. It
holds no state between calls, which is what makes cloning it per call
free and the plugin's residency story entirely the garbage collector's.

### fn script

The `-n` the runner passes alongside this script skips config loading,
so the composed call runs against a bare nushell rather than the box
user's customised one. Both channels render as NUON literals because an
external process has no pipeline to be handed a value on - the same
one-grammar convenience the think path uses, spelled for a process
boundary.

### fn run

The wait is a poll loop rather than a blocking wait so the timeout can
kill a runaway ask; the drain threads exist because reading a stream
after the wait deadlocks the moment an ask writes more than a pipe
buffer.

## fn bound

## fn drain

## fn trimmed_reason

Exit 127 is distinguished because it is the sandbox failing to find the
shell rather than the ask failing, and the bare code would read as the
model's fault.
