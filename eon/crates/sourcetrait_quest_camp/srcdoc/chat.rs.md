# chat.rs

The TUI: a transcript above a multiline input box, driven by an async select
over two channels.

THE LOOP IS EVENT-DRIVEN WITH NO POLL CADENCE. It selects over the bridge
events channel and a blocking-read input pump, and redraws once per event.
Nothing here ticks, which is what keeps an idle camp at zero cost while a
generation streams at the engine's own rate.

THE CONVERSATION CONTEXT LIVES ENGINE-SIDE. Camp sends turn text and renders
events; it holds a transcript for display and a log, and nothing that the
model's next turn depends on. That is why `.new` is a `Reset` request rather
than anything camp does to its own state alone.

## const INPUT_MAX_ROWS

## const INPUT_CAPACITY

## enum Speaker

`Note` is camp's own voice - hints, errors, refusals - and it is a THIRD
speaker rather than a styled variant of the other two, so nothing camp says
about the session can be mistaken for something the model said.

## struct Turn

## struct Session

A session is the nom, its log path and the transcript together, so `.new`
replaces one value rather than resetting four fields and risking a stale one.

### fn new

### fn write_log

Rewrites the WHOLE transcript per turn rather than appending. At chat sizes
that is cheap, and it means a turn that edits the last entry - which streamed
chunks do constantly - never needs the log to be rewound.

## fn sessions_dir

Camp's OWN cache namespace rather than the quest suite's, because CAMP writes
this file. The suite's config home is for what the suite configures; a session
log is camp's artifact and belongs under camp's name.

Logs are on by default, which is the_user's ruling rather than a default that
drifted in.

## fn base62

## fn render_log

## fn byte_index

Every cursor position in this file is a CHAR index and every string operation
needs a BYTE index, so this conversion is load-bearing rather than a
convenience. Using the char index directly would panic on any multi-byte
input the moment a cursor sat past it.

## fn wrap_line

Greedy and space-aware, with overlong words hard-split so a pasted URL cannot
push the layout wider than the pane.

An empty line yields ONE empty row rather than none, which is what makes the
blank separator between turns survive wrapping.

## fn transcript_lines

## fn cursor_row_col

## fn draw

The input box grows with its content up to a cap and then scrolls, and the
scroll offset is derived from the CURSOR rather than from the buffer's length,
so a cursor moved up into a long input stays visible.

## struct LinkState

Four flags rather than an enum because they are not mutually exclusive: a
session can be ready and generating, or closed carrying an error from before
it closed.

## enum LoopAction

## fn apply_event

A `Chunk` appends to the last turn when that turn is the model's, and starts
one otherwise. That is what lets the submit path push an EMPTY model turn
immediately - the transcript shows the model has the floor before any text
arrives.

## fn spawn_input_pump

A blocking crossterm read on its own thread, feeding a channel the async loop
can select on.

THE READ HAS NO SHUTDOWN HANDLE, and the thread ends with the process. That is
a deliberate seam rather than an oversight: the alternative is an async event
stream, and this shape is the smaller one until something actually needs to
stop the pump without exiting.

## fn chat

Terminal setup and teardown wrap the loop, and the teardown runs on EVERY
path, including the error path - the result is held and returned after
restoring the terminal. Returning early on an error would leave the terminal
in raw mode with the alternate screen still up.

Keyboard enhancement is probed rather than assumed. Distinguishing shift+enter
from enter needs the kitty protocol; elsewhere shift+enter arrives as a plain
Enter and submits, which is why the hint text differs by terminal rather than
promising a key that will not work.

## fn chat_loop

The engine-initiated close is handled AFTER the select rather than inside it,
so the error surfaces once the terminal has been restored. Camp's own quits
return before this point without waiting for the engine to answer.

## fn apply_terminal_event

DOT COMMANDS COUNT ONLY AS THE FIRST CHARACTER OF THE FIRST LINE, and the
first line alone is the command. A dot anywhere else is ordinary text, which
is what keeps a multi-line message that happens to begin a later line with a
period from being eaten as a command.

An unrecognized dot line NOTES rather than reaching the model, so a typo'd
command does not silently become a prompt.

Cancel uses `try_send` where every other request awaits. The point of Esc is
that it works while the loop is busy; awaiting a full channel would defer the
cancel behind the very backlog it is meant to interrupt.

A control-modified character is swallowed rather than inserted, so an unbound
chord cannot type a stray letter into the input buffer.

Paste normalizes line endings on the way in, because a pasted block from
another terminal or platform otherwise carries carriage returns that the
wrapper would render as blanks.
