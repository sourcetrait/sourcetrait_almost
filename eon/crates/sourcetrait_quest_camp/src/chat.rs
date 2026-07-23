use crate::*;

/// Cap on the input box's visible text rows (it scrolls beyond this).
const INPUT_MAX_ROWS: u16 = 8;

/// Terminal-event channel bound (keystrokes and pastes are small and
/// consumer-paced).
const INPUT_CAPACITY: usize = 64;

/// Who a transcript turn belongs to (drives the line style).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Speaker {
    You,
    Quest,
    Note,
}

pub(crate) struct Turn {
    pub(crate) speaker: Speaker,
    pub(crate) text: String,
}

/// One conversation: its base62 nom (the top-right title and the log
/// file's stem) and the transcript. /new replaces the whole thing and
/// asks the engine to Reset its context.
struct Session {
    nom: String,
    log_path: Option<PathBuf>,
    turns: Vec<Turn>,
    /// Transcript lines scrolled back from the bottom (PgUp/PgDn).
    scroll_back: usize,
}

impl Session {
    fn new() -> CampResult<Self> {
        let nom = base62(RandomState::new().build_hasher().finish());
        let dir = sessions_dir()?;
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            nom: nom.clone(),
            log_path: Some(dir.join(format!("{nom}.txt"))),
            turns: Vec::new(),
            scroll_back: 0,
        })
    }

    /// Rewrite the session log with the whole transcript (each turn
    /// updates it; cheap at chat sizes).
    fn write_log(&self) -> CampResult<()> {
        if let Some(path) = &self.log_path {
            std::fs::write(path, render_log(&self.turns))?;
        }
        Ok(())
    }
}

/// The session-log home: camp's OWN cache namespace,
/// $XDG_CACHE_HOME/sourcetrait/camp/session (the spec fallback
/// ~/.cache) - camp writes the log, so the home is camp's, not the
/// quest suite's. Logs are on by default (the_user ruling).
fn sessions_dir() -> CampResult<PathBuf> {
    let base = match std::env::var("XDG_CACHE_HOME") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => match std::env::var("HOME") {
            Ok(home) if !home.is_empty() => PathBuf::from(home).join(".cache"),
            _ => snafu::whatever!("neither XDG_CACHE_HOME nor HOME is set"),
        },
    };
    Ok(base.join("sourcetrait/camp/session"))
}

/// base62 rendering of a u64 (0-9A-Za-z), the session nom form.
pub(crate) fn base62(mut value: u64) -> String {
    const DIGITS: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    if value == 0 {
        return String::from("0");
    }
    let mut out: Vec<u8> = Vec::new();
    while value > 0 {
        out.push(DIGITS[(value % 62) as usize]);
        value /= 62;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

/// The transcript as plain text (the session log's file form).
pub(crate) fn render_log(turns: &[Turn]) -> String {
    let mut text = String::new();
    for turn in turns {
        let label = match turn.speaker {
            Speaker::You => "you: ",
            Speaker::Quest => "quest: ",
            Speaker::Note => "",
        };
        text.push_str(label);
        text.push_str(&turn.text);
        text.push_str("\n\n");
    }
    text
}

/// Byte offset of a char index into `text` (UTF-8 safe; clamps to end).
pub(crate) fn byte_index(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

/// Greedy space-aware wrap of one logical line into rows of at most
/// `width` chars; overlong words hard-split. An empty line yields one
/// empty row (the blank separator between turns).
pub(crate) fn wrap_line(line: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut rows: Vec<String> = Vec::new();
    let mut row = String::new();
    let mut row_len = 0usize;
    for word in line.split(' ') {
        let word_len = word.chars().count();
        if row_len > 0 && row_len + 1 + word_len > width {
            rows.push(std::mem::take(&mut row));
            row_len = 0;
        }
        if row_len > 0 {
            row.push(' ');
            row_len += 1;
        }
        if word_len > width {
            for character in word.chars() {
                if row_len == width {
                    rows.push(std::mem::take(&mut row));
                    row_len = 0;
                }
                row.push(character);
                row_len += 1;
            }
        } else {
            row.push_str(word);
            row_len += word_len;
        }
    }
    rows.push(row);
    rows
}

/// The transcript as wrapped, styled display lines.
fn transcript_lines(turns: &[Turn], width: usize) -> Vec<ratatui::text::Line<'static>> {
    let mut lines: Vec<ratatui::text::Line<'static>> = Vec::new();
    for turn in turns {
        let (label, style) = match turn.speaker {
            Speaker::You => ("you: ", ratatui::style::Style::new().fg(ratatui::style::Color::Cyan)),
            Speaker::Quest => ("quest: ", ratatui::style::Style::new()),
            Speaker::Note => ("", ratatui::style::Style::new().fg(ratatui::style::Color::DarkGray)),
        };
        let combined = format!("{label}{}", turn.text);
        for logical in combined.split('\n') {
            for row in wrap_line(logical, width) {
                lines.push(ratatui::text::Line::styled(row, style));
            }
        }
        lines.push(ratatui::text::Line::raw(""));
    }
    lines
}

/// (row, col) of the cursor char index within the input buffer.
fn cursor_row_col(input: &str, cursor: usize) -> (u16, u16) {
    let before = &input[..byte_index(input, cursor)];
    let row = before.matches('\n').count() as u16;
    let col = before.chars().rev().take_while(|c| *c != '\n').count() as u16;
    (row, col)
}

fn draw(
    terminal: &mut ratatui::DefaultTerminal,
    session: &Session,
    input: &str,
    cursor: usize,
    status: &str,
) -> CampResult<()> {
    terminal.draw(|frame| {
        let input_rows = (input.split('\n').count() as u16).clamp(1, INPUT_MAX_ROWS);
        let chunks = ratatui::layout::Layout::vertical([
            ratatui::layout::Constraint::Min(1),
            ratatui::layout::Constraint::Length(input_rows + 2),
        ])
        .split(frame.area());

        let session_width = chunks[0].width.saturating_sub(2).max(1) as usize;
        let session_height = chunks[0].height.saturating_sub(2) as usize;
        let lines = transcript_lines(&session.turns, session_width);
        let bottom = lines.len().saturating_sub(session_height);
        let scroll = bottom.saturating_sub(session.scroll_back) as u16;
        frame.render_widget(
            ratatui::widgets::Paragraph::new(ratatui::text::Text::from(lines))
                .scroll((scroll, 0))
                .block(
                    ratatui::widgets::Block::bordered()
                        .title("quest chat")
                        .title(ratatui::text::Line::from(session.nom.clone()).right_aligned()),
                ),
            chunks[0],
        );

        let (cursor_row, cursor_col) = cursor_row_col(input, cursor);
        let input_scroll = (cursor_row + 1).saturating_sub(input_rows);
        frame.render_widget(
            ratatui::widgets::Paragraph::new(input)
                .scroll((input_scroll, 0))
                .block(ratatui::widgets::Block::bordered().title(status.to_string())),
            chunks[1],
        );
        frame.set_cursor_position(ratatui::layout::Position::new(
            chunks[1].x + 1 + cursor_col.min(chunks[1].width.saturating_sub(2)),
            chunks[1].y + 1 + (cursor_row - input_scroll),
        ));
    })?;
    Ok(())
}

/// The bridge link's view of the session: loading until Ready,
/// generating between a submitted Turn and its TurnDone, closed when
/// the engine says so (or its channel drops).
struct LinkState {
    ready: bool,
    generating: bool,
    closed: bool,
    last_error: Option<String>,
}

/// What a handled terminal event tells the loop to do next.
enum LoopAction {
    Continue,
    Quit,
}

/// Fold one bridge event into the transcript and state.
fn apply_event(
    event: bridge::all::ChatEvent,
    session: &mut Session,
    status: &mut String,
    state: &mut LinkState,
) -> CampResult<()> {
    match event {
        bridge::all::ChatEvent::Ready(info) => {
            state.ready = true;
            *status = format!("ready - era {} ({})", info.era, info.model);
        }
        bridge::all::ChatEvent::Chunk { text } => {
            match session.turns.last_mut() {
                Some(last) if last.speaker == Speaker::Quest => last.text.push_str(&text),
                _ => session.turns.push(Turn { speaker: Speaker::Quest, text }),
            }
        }
        bridge::all::ChatEvent::TurnDone(report) => {
            state.generating = false;
            let rate = report.generated_token_count as f64
                / report.decode_seconds.max(f64::EPSILON);
            let marker = match report.finish {
                bridge::all::FinishReason::Cancelled => " (stopped)",
                bridge::all::FinishReason::SampleLen => " (budget spent)",
                bridge::all::FinishReason::StopToken => "",
            };
            *status = format!(
                "ready - {} tokens at {rate:.1} tok/s{marker}",
                report.generated_token_count
            );
            session.write_log()?;
        }
        bridge::all::ChatEvent::Error { message } => {
            session.turns.push(Turn {
                speaker: Speaker::Note,
                text: format!("error: {message}"),
            });
            state.last_error = Some(message);
        }
        bridge::all::ChatEvent::Closed => {
            state.closed = true;
        }
    }
    Ok(())
}

/// Pump blocking crossterm reads into a channel the async loop
/// selects on. The read has no shutdown handle - the thread ends
/// with the process (this seam swaps to an async EventStream if
/// that ever matters).
fn spawn_input_pump() -> CampResult<r::mpsc::Receiver<r::term::Event>> {
    let (events_tx, events_rx) = r::mpsc::channel(INPUT_CAPACITY);
    let spawned = std::thread::Builder::new()
        .name(String::from("camp-input"))
        .spawn(move || {
            while let Ok(event) = r::term::event::read() {
                if events_tx.blocking_send(event).is_err() {
                    break;
                }
            }
        });
    if let Err(e) = spawned {
        snafu::whatever!("terminal input pump spawn failed: {e}");
    }
    Ok(events_rx)
}

/// Interactive chat TUI over a bridge ChatSession: the transcript
/// above a multiline input box. Enter submits, shift+enter inserts a
/// newline (distinguishing the two needs a kitty-protocol terminal;
/// elsewhere shift+enter arrives as plain Enter and submits), .exit
/// quits, .new starts a fresh session (new nom + log, engine Reset) -
/// dot commands count only as the first character of the first line -
/// PgUp/PgDn scroll the transcript, Esc cancels a generation early,
/// ctrl+c quits anywhere. The conversation context lives engine-side;
/// camp only sends turn text and renders events. Sessions log by
/// default to <cache>/sourcetrait/camp/session/<nom>.txt per turn.
/// The loop is async and event-driven: a select over the bridge
/// events channel and a blocking-read input pump, redrawn per event -
/// no poll cadence. This wraps it in terminal setup/teardown; the
/// model loads on the engine thread, so the loop opens in a loading
/// state.
pub(crate) async fn chat(link: bridge::all::ChatSession) -> CampResult<()> {
    let session = Session::new()?;
    if let Some(path) = &session.log_path {
        eprintln!("camp: logging session to {}", path.display());
    }
    let bridge::all::ChatSession { requests, events } = link;
    let mut terminal = ratatui::try_init()?;
    let enhanced = r::term::supports_keyboard_enhancement().unwrap_or(false);
    if enhanced {
        let _ = r::term::execute!(
            io::stdout(),
            r::term::PushKeyboardEnhancementFlags(
                r::term::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        );
    }
    let _ = r::term::execute!(io::stdout(), r::term::EnableBracketedPaste);
    let result = match spawn_input_pump() {
        Ok(input_events) => {
            chat_loop(&mut terminal, &requests, events, input_events, session, enhanced).await
        }
        Err(error) => Err(error),
    };
    let _ = r::term::execute!(io::stdout(), r::term::DisableBracketedPaste);
    if enhanced {
        let _ = r::term::execute!(io::stdout(), r::term::PopKeyboardEnhancementFlags);
    }
    ratatui::restore();
    result
}

async fn chat_loop(
    terminal: &mut ratatui::DefaultTerminal,
    requests: &r::mpsc::Sender<bridge::all::ChatRequest>,
    mut events: r::mpsc::Receiver<bridge::all::ChatEvent>,
    mut input_events: r::mpsc::Receiver<r::term::Event>,
    mut session: Session,
    enhanced: bool,
) -> CampResult<()> {
    let hint = if enhanced {
        "enter submits, shift+enter newline, .new restarts, .exit quits, pgup/pgdn scroll"
    } else {
        "enter submits (shift+enter needs a kitty-protocol terminal), .new restarts, .exit quits"
    };
    session.turns.push(Turn { speaker: Speaker::Note, text: String::from(hint) });
    let mut input = String::new();
    let mut cursor = 0usize;
    let mut status = String::from("loading model...");
    let mut state = LinkState {
        ready: false,
        generating: false,
        closed: false,
        last_error: None,
    };

    loop {
        draw(terminal, &session, &input, cursor, &status)?;
        tokio::select! {
            event = events.recv() => match event {
                Some(event) => apply_event(event, &mut session, &mut status, &mut state)?,
                None => state.closed = true,
            },
            terminal_event = input_events.recv() => match terminal_event {
                Some(terminal_event) => {
                    let action = apply_terminal_event(
                        terminal_event,
                        requests,
                        &mut session,
                        &mut status,
                        &mut state,
                        &mut input,
                        &mut cursor,
                        hint,
                    )
                    .await?;
                    if let LoopAction::Quit = action {
                        return Ok(());
                    }
                }
                None => snafu::whatever!("the terminal input pump ended"),
            },
        }
        if state.closed {
            // Engine-initiated end (camp's own quits return above
            // without waiting) - surface it after terminal restore.
            match state.last_error.take() {
                Some(message) => snafu::whatever!("the engine closed the session: {message}"),
                None => snafu::whatever!("the engine closed the session"),
            }
        }
    }
}

/// Fold one terminal event into the input buffer / session / bridge
/// requests.
#[allow(clippy::too_many_arguments)]
async fn apply_terminal_event(
    terminal_event: r::term::Event,
    requests: &r::mpsc::Sender<bridge::all::ChatRequest>,
    session: &mut Session,
    status: &mut String,
    state: &mut LinkState,
    input: &mut String,
    cursor: &mut usize,
    hint: &str,
) -> CampResult<LoopAction> {
    match terminal_event {
        r::term::Event::Key(key) if key.kind != r::term::KeyEventKind::Release => {
            match key.code {
                r::term::KeyCode::Enter
                    if key.modifiers.contains(r::term::KeyModifiers::SHIFT) =>
                {
                    input.insert(byte_index(input, *cursor), '\n');
                    *cursor += 1;
                }
                r::term::KeyCode::Enter => {
                    if input.starts_with('.') {
                        // Dot commands: the dot counts only as the
                        // FIRST character of the FIRST line; the
                        // first line is the command. A dot line that
                        // is no known command notes instead of
                        // reaching the model.
                        let command = input
                            .split('\n')
                            .next()
                            .unwrap_or("")
                            .trim_end()
                            .to_string();
                        input.clear();
                        *cursor = 0;
                        if command == ".exit" {
                            let _ = requests.send(bridge::all::ChatRequest::Close).await;
                            return Ok(LoopAction::Quit);
                        }
                        if command == ".new" {
                            let _ = requests.send(bridge::all::ChatRequest::Reset).await;
                            *session = Session::new()?;
                            session.turns.push(Turn {
                                speaker: Speaker::Note,
                                text: String::from(hint),
                            });
                            *status = String::from("ready - new session");
                            return Ok(LoopAction::Continue);
                        }
                        session.turns.push(Turn {
                            speaker: Speaker::Note,
                            text: format!("unknown command: {command}"),
                        });
                        return Ok(LoopAction::Continue);
                    }
                    let text = input.trim().to_string();
                    if text.is_empty() {
                        return Ok(LoopAction::Continue);
                    }
                    input.clear();
                    *cursor = 0;
                    if !state.ready {
                        session.turns.push(Turn {
                            speaker: Speaker::Note,
                            text: String::from("the model is still loading"),
                        });
                        return Ok(LoopAction::Continue);
                    }
                    if state.generating {
                        session.turns.push(Turn {
                            speaker: Speaker::Note,
                            text: String::from("a turn is generating - esc stops it"),
                        });
                        return Ok(LoopAction::Continue);
                    }
                    session.scroll_back = 0;
                    session.turns.push(Turn { speaker: Speaker::You, text: text.clone() });
                    session.turns.push(Turn { speaker: Speaker::Quest, text: String::new() });
                    if requests
                        .send(bridge::all::ChatRequest::Turn { text })
                        .await
                        .is_err()
                    {
                        state.closed = true;
                        return Ok(LoopAction::Continue);
                    }
                    state.generating = true;
                    *status = String::from("generating - esc stops");
                }
                r::term::KeyCode::Esc if state.generating => {
                    let _ = requests.try_send(bridge::all::ChatRequest::Cancel);
                }
                r::term::KeyCode::PageUp => {
                    session.scroll_back = session.scroll_back.saturating_add(8);
                }
                r::term::KeyCode::PageDown => {
                    session.scroll_back = session.scroll_back.saturating_sub(8);
                }
                r::term::KeyCode::Char('c')
                    if key.modifiers.contains(r::term::KeyModifiers::CONTROL) =>
                {
                    let _ = requests.send(bridge::all::ChatRequest::Close).await;
                    return Ok(LoopAction::Quit);
                }
                r::term::KeyCode::Char(_)
                    if key.modifiers.contains(r::term::KeyModifiers::CONTROL) => {}
                r::term::KeyCode::Char(character) => {
                    input.insert(byte_index(input, *cursor), character);
                    *cursor += 1;
                }
                r::term::KeyCode::Backspace => {
                    if *cursor > 0 {
                        *cursor -= 1;
                        input.remove(byte_index(input, *cursor));
                    }
                }
                r::term::KeyCode::Left => {
                    *cursor = cursor.saturating_sub(1);
                }
                r::term::KeyCode::Right if *cursor < input.chars().count() => {
                    *cursor += 1;
                }
                _ => {}
            }
        }
        r::term::Event::Paste(pasted) => {
            let normalized = pasted.replace("\r\n", "\n").replace('\r', "\n");
            input.insert_str(byte_index(input, *cursor), &normalized);
            *cursor += normalized.chars().count();
        }
        _ => {}
    }
    Ok(LoopAction::Continue)
}
