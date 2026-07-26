# driver.rs

The spawn-and-stream plumbing every python-driven verb shares, and the one place
that decides what a baseline run's output looks like.

Three streams out of one, which is the design: the payload goes to stdout, one
terse summary line per event goes to stderr, and the whole JSON-lines event
record goes to a file when asked for. That split is what lets a baseline be both
readable at the terminal and exact on disk, and it is why the drivers emit
structured events rather than prose in the first place.

## fn run_driver

Only stdout is piped. The driver's own stderr passes through untouched, so a
torch warning or a triton compile note reaches the terminal as itself rather
than arriving mangled through our summariser.

Events are collected before the record is written rather than streamed to it,
which means a killed run leaves no partial record. That is the opposite of the
trainer's step log, and the reason differs rather than the judgement: a step log
exists so a crashed run still shows its progress, while a baseline record is a
measurement that is either complete or not a measurement.

The exit status is checked after the record is written, so a driver that failed
partway still leaves whatever it emitted on disk for reading.

## fn summarize_event

A line that is not JSON is printed as-is rather than dropped or raising. The
drivers are python and a stray print or a traceback line is not an event, so the
summariser has to pass it through - and that passthrough is often exactly what a
diagnosis needs.

An unrecognised event kind prints its own name and the whole line, so adding an
event to a driver does not require touching this match to see it. The known
kinds are the ones worth compressing, and the fallback keeps the rest legible.
