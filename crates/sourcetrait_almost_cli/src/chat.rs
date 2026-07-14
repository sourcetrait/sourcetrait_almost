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
    turns: &[Turn],
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
        let lines = transcript_lines(turns, session_width);
        let scroll = lines.len().saturating_sub(session_height) as u16;
        frame.render_widget(
            ratatui::widgets::Paragraph::new(ratatui::text::Text::from(lines))
                .scroll((scroll, 0))
                .block(ratatui::widgets::Block::bordered().title("almost chat")),
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
/// honoring Esc (stop this generation) and ctrl+c (stop + quit).
/// Returns whether the app should quit. Takes the whole LoadedModel so
/// the cli never names the tokenizer's crate (field borrows split it).
fn run_turn(
    terminal: &mut ratatui::DefaultTerminal,
    loaded: &mut lib::LoadedModel,
    options: &lib::GenerateOptions,
    turns: &mut Vec<Turn>,
    restored: &mut Option<lib::RestoredContext>,
    status: &mut String,
    text: &str,
) -> lib::AlmostResult<bool> {
    turns.push(Turn { speaker: Speaker::You, text: String::from(text) });
    turns.push(Turn { speaker: Speaker::Almost, text: String::new() });
    *status = String::from("generating - esc stops");
    draw(terminal, turns, "", 0, status)?;

    let rendered = match restored {
        Some(_) => lib::chat_continue(text),
        None => lib::chat_wrap(text),
    };
    let started = match restored.as_ref() {
        Some(context) => loaded.model.generate_from(&loaded.tokenizer, context, &rendered, options),
        None => loaded.model.generate(&loaded.tokenizer, &rendered, options),
    };
    let mut generation = match started {
        Ok(generation) => generation,
        Err(error) => {
            turns.push(Turn { speaker: Speaker::Note, text: format!("error: {error}") });
            *status = String::from("ready");
            return Ok(false);
        }
    };

    let mut stopped = false;
    let mut quit = false;
    for step in &mut generation {
        match step {
            Ok(step) => {
                if let Some(chunk) = step.chunk
                    && let Some(last) = turns.last_mut()
                {
                    last.text.push_str(&chunk);
                }
            }
            Err(error) => {
                turns.push(Turn { speaker: Speaker::Note, text: format!("error: {error}") });
                break;
            }
        }
        draw(terminal, turns, "", 0, status)?;
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
        && let Some(last) = turns.last_mut()
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
    *restored = Some(lib::RestoredContext {
        context_len,
        context_ids: report.context_ids,
    });
    Ok(quit)
}

/// Interactive chat TUI: the session transcript above a multiline input
/// box. Enter submits, shift+enter inserts a newline (distinguishing the
/// two needs a kitty-protocol terminal; elsewhere shift+enter arrives as
/// plain Enter and submits), /exit quits, Esc stops a generation early,
/// ctrl+c quits anywhere. Turns stay in VRAM: each generation's report
/// seeds the next turn's RestoredContext, so the KV cache carries the
/// whole conversation without disk snapshots. The model arrives loaded
/// (stderr diagnostics happen before the alternate screen); this wraps
/// the event loop in terminal setup/teardown.
pub(crate) fn chat(loaded: lib::LoadedModel, config: &lib::Config) -> lib::AlmostResult<()> {
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
    let result = chat_loop(&mut terminal, loaded, config, enhanced);
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
    enhanced: bool,
) -> lib::AlmostResult<()> {
    let mut loaded = loaded;
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
        "enter submits, shift+enter newline, /exit quits"
    } else {
        "enter submits (shift+enter needs a kitty-protocol terminal), /exit quits"
    };
    let mut turns: Vec<Turn> = vec![Turn { speaker: Speaker::Note, text: String::from(hint) }];
    let mut input = String::new();
    let mut cursor = 0usize;
    let mut restored: Option<lib::RestoredContext> = None;
    let mut status = String::from("ready");

    loop {
        draw(terminal, &turns, &input, cursor, &status)?;
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
                        let quit = run_turn(
                            terminal,
                            &mut loaded,
                            &options,
                            &mut turns,
                            &mut restored,
                            &mut status,
                            &text,
                        )?;
                        if quit {
                            return Ok(());
                        }
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
