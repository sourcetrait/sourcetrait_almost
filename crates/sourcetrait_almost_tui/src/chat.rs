use crate::*;

/// Cap on the input box's visible text rows (it scrolls beyond this).
const INPUT_MAX_ROWS: u16 = 8;

/// Who a transcript turn belongs to (drives the line style).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Speaker {
    You,
    Almost,
    Note,
}

pub(crate) struct Turn {
    pub(crate) speaker: Speaker,
    pub(crate) text: String,
}

/// One conversation: its base62 nom (the top-right title and the log
/// file's stem), the transcript, and the model-side continuation state.
/// /new replaces the whole thing (and clears the KV cache).
struct Session {
    nom: String,
    log_path: Option<PathBuf>,
    turns: Vec<Turn>,
    restored: Option<lib::RestoredContext>,
    /// Transcript lines scrolled back from the bottom (PgUp/PgDn).
    scroll_back: usize,
}

impl Session {
    fn new(log: bool) -> lib::AlmostResult<Self> {
        let nom = base62(RandomState::new().build_hasher().finish());
        let log_path = if log {
            let dir = lib::default_sessions_dir();
            std::fs::create_dir_all(&dir)?;
            Some(dir.join(format!("{nom}.txt")))
        } else {
            None
        };
        Ok(Self {
            nom,
            log_path,
            turns: Vec::new(),
            restored: None,
            scroll_back: 0,
        })
    }

    /// Rewrite the session log with the whole transcript (each turn
    /// updates it; cheap at chat sizes).
    fn write_log(&self) -> lib::AlmostResult<()> {
        if let Some(path) = &self.log_path {
            std::fs::write(path, render_log(&self.turns))?;
        }
        Ok(())
    }
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
            Speaker::Almost => "almost: ",
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
            Speaker::Almost => ("almost: ", ratatui::style::Style::new()),
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
) -> lib::AlmostResult<()> {
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
                        .title("almost chat")
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

/// Run one submitted turn: stream the generation into the transcript,
/// honoring Esc (stop this generation) and ctrl+c (stop + quit), then
/// update the session log. Returns whether the app should quit. Takes
/// the whole LoadedModel so the tui never names the tokenizer's crate
/// (field borrows split it).
fn run_turn(
    terminal: &mut ratatui::DefaultTerminal,
    loaded: &mut lib::LoadedModel,
    options: &lib::GenerateOptions,
    session: &mut Session,
    status: &mut String,
    text: &str,
) -> lib::AlmostResult<bool> {
    session.scroll_back = 0;
    session.turns.push(Turn { speaker: Speaker::You, text: String::from(text) });
    session.turns.push(Turn { speaker: Speaker::Almost, text: String::new() });
    *status = String::from("generating - esc stops");
    draw(terminal, session, "", 0, status)?;

    let rendered = match session.restored {
        Some(_) => lib::chat_continue(text),
        None => lib::chat_wrap(text),
    };
    let started = match session.restored.as_ref() {
        Some(context) => loaded.model.generate_from(&loaded.tokenizer, context, &rendered, options),
        None => loaded.model.generate(&loaded.tokenizer, &rendered, options),
    };
    let mut generation = match started {
        Ok(generation) => generation,
        Err(error) => {
            session.turns.push(Turn { speaker: Speaker::Note, text: format!("error: {error}") });
            *status = String::from("ready");
            session.write_log()?;
            return Ok(false);
        }
    };

    let mut stopped = false;
    let mut quit = false;
    for step in &mut generation {
        match step {
            Ok(step) => {
                if let Some(chunk) = step.chunk
                    && let Some(last) = session.turns.last_mut()
                {
                    last.text.push_str(&chunk);
                }
            }
            Err(error) => {
                session.turns.push(Turn { speaker: Speaker::Note, text: format!("error: {error}") });
                break;
            }
        }
        draw(terminal, session, "", 0, status)?;
        while r::term::event::poll(Duration::from_millis(0))? {
            if let r::term::Event::Key(key) = r::term::event::read()?
                && key.kind != r::term::KeyEventKind::Release
            {
                match key.code {
                    r::term::KeyCode::Esc => stopped = true,
                    r::term::KeyCode::Char('c')
                        if key.modifiers.contains(r::term::KeyModifiers::CONTROL) =>
                    {
                        stopped = true;
                        quit = true;
                    }
                    _ => {}
                }
            }
        }
        if stopped {
            break;
        }
    }
    let report = generation.finish()?;
    if let Some(rest) = &report.rest
        && let Some(last) = session.turns.last_mut()
        && last.speaker == Speaker::Almost
    {
        last.text.push_str(rest);
    }
    let context_len = report.context_ids.len();
    let rate = report.generated_token_count as f64 / report.decode_seconds.max(f64::EPSILON);
    *status = format!(
        "ready - {context_len} ctx tokens, last {:.1} tok/s{}",
        rate,
        if stopped { " (stopped)" } else { "" }
    );
    session.restored = Some(lib::RestoredContext {
        context_len,
        context_ids: report.context_ids,
    });
    session.write_log()?;
    Ok(quit)
}

/// Interactive chat TUI: the session transcript above a multiline input
/// box. Enter submits, shift+enter inserts a newline (distinguishing the
/// two needs a kitty-protocol terminal; elsewhere shift+enter arrives as
/// plain Enter and submits), /exit quits, /new starts a fresh session
/// (new nom + log, cleared transcript and KV), PgUp/PgDn scroll the
/// transcript, Esc stops a generation early, ctrl+c quits anywhere.
/// Turns stay in VRAM: each generation's report seeds the next turn's
/// RestoredContext, so the KV cache carries the whole conversation
/// without disk snapshots. Sessions log to
/// <cache>/sourcetrait/almost/session/<nom>.txt per turn when
/// config [chat] log = true (the default). The model arrives loaded
/// (stderr diagnostics happen before the alternate screen); this wraps
/// the event loop in terminal setup/teardown.
pub(crate) fn chat(loaded: lib::LoadedModel, config: &lib::Config) -> lib::AlmostResult<()> {
    let session = Session::new(config.chat.log)?;
    if let Some(path) = &session.log_path {
        eprintln!("talmost: logging session to {}", path.display());
    }
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
    let result = chat_loop(&mut terminal, loaded, config, session, enhanced);
    let _ = r::term::execute!(io::stdout(), r::term::DisableBracketedPaste);
    if enhanced {
        let _ = r::term::execute!(io::stdout(), r::term::PopKeyboardEnhancementFlags);
    }
    ratatui::restore();
    result
}

fn chat_loop(
    terminal: &mut ratatui::DefaultTerminal,
    loaded: lib::LoadedModel,
    config: &lib::Config,
    session: Session,
    enhanced: bool,
) -> lib::AlmostResult<()> {
    let mut loaded = loaded;
    let mut session = session;
    let options = lib::GenerateOptions {
        greedy: config.generation.greedy,
        temperature: config.generation.temperature,
        top_p: config.generation.top_p,
        sample_len: config.generation.sample_len,
        seed: config.generation.seed,
        speculate: config.generation.speculate,
        dump_logits: None,
    };
    let hint = if enhanced {
        "enter submits, shift+enter newline, /new restarts, /exit quits, pgup/pgdn scroll"
    } else {
        "enter submits (shift+enter needs a kitty-protocol terminal), /new restarts, /exit quits"
    };
    session.turns.push(Turn { speaker: Speaker::Note, text: String::from(hint) });
    let mut input = String::new();
    let mut cursor = 0usize;
    let mut status = String::from("ready");

    loop {
        draw(terminal, &session, &input, cursor, &status)?;
        match r::term::event::read()? {
            r::term::Event::Key(key) if key.kind != r::term::KeyEventKind::Release => {
                match key.code {
                    r::term::KeyCode::Enter
                        if key.modifiers.contains(r::term::KeyModifiers::SHIFT) =>
                    {
                        input.insert(byte_index(&input, cursor), '\n');
                        cursor += 1;
                    }
                    r::term::KeyCode::Enter => {
                        let text = input.trim().to_string();
                        if text.is_empty() {
                            continue;
                        }
                        input.clear();
                        cursor = 0;
                        if text == "/exit" {
                            return Ok(());
                        }
                        if text == "/new" {
                            loaded.model.clear_kv_cache();
                            session = Session::new(config.chat.log)?;
                            session.turns.push(Turn {
                                speaker: Speaker::Note,
                                text: String::from(hint),
                            });
                            status = String::from("ready - new session");
                            continue;
                        }
                        let quit = run_turn(
                            terminal,
                            &mut loaded,
                            &options,
                            &mut session,
                            &mut status,
                            &text,
                        )?;
                        if quit {
                            return Ok(());
                        }
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
                        return Ok(());
                    }
                    r::term::KeyCode::Char(_)
                        if key.modifiers.contains(r::term::KeyModifiers::CONTROL) => {}
                    r::term::KeyCode::Char(character) => {
                        input.insert(byte_index(&input, cursor), character);
                        cursor += 1;
                    }
                    r::term::KeyCode::Backspace => {
                        if cursor > 0 {
                            cursor -= 1;
                            input.remove(byte_index(&input, cursor));
                        }
                    }
                    r::term::KeyCode::Left => {
                        cursor = cursor.saturating_sub(1);
                    }
                    r::term::KeyCode::Right if cursor < input.chars().count() => {
                        cursor += 1;
                    }
                    _ => {}
                }
            }
            r::term::Event::Paste(pasted) => {
                let normalized = pasted.replace("\r\n", "\n").replace('\r', "\n");
                input.insert_str(byte_index(&input, cursor), &normalized);
                cursor += normalized.chars().count();
            }
            _ => {}
        }
    }
}
