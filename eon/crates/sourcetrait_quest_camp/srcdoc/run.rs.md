# run.rs

## struct Cli

The three profile tokens are the whole command line, deliberately. Era
profiles own the settings, so camp exposes no knob farm of its own - a
decode budget, a sampling posture or a graph toggle is configured where every
other tool in the suite configures it.

THESE DOC COMMENTS ARE ALSO THE `--help` TEXT, and they follow the same rules
as every other doc comment - there is no product-text exemption. The token
rules they used to spell out in full are the suite-wide ones, documented once
in the library's config module and its mirror, so the shorter help loses
duplication rather than information.

## fn run

Diagnostics before the alternate screen opens go to stderr, and after it opens
the TUI owns stdout entirely. That split is why the pre-session lines are
`eprintln!` rather than anything ratatui draws.

## fn drive

The runtime is CURRENT-THREAD. Camp's only futures are channel receives, so
nothing here needs a driver or a thread pool, and the engine is not on this
runtime at all - it loads and runs on the session's own thread, spawned inside
`open_chat`.

The consequence a reader should hold: the loop opens in a LOADING state rather
than a ready one. `open_chat` returns as soon as the channels exist, so the
first paint happens before the model is live and `Ready` arrives later as an
ordinary event.
