//! Shared spawn-and-stream plumbing for the pinned python drivers.
use crate::*;

/// Spawn a prepared driver command: stream its JSON-lines stdout into
/// payload (stdout) + summaries (stderr), tee everything to `record`.
pub(crate) fn run_driver(mut cmd: process::Command, record: Option<&Path>) -> RefquestResult<()> {
    cmd.stdout(process::Stdio::piped());
    let mut child = cmd.spawn()?;
    let Some(stdout) = child.stdout.take() else {
        snafu::whatever!("driver stdout was not captured");
    };

    let mut events: Vec<String> = Vec::new();
    for line in io::BufRead::lines(io::BufReader::new(stdout)) {
        let line = line?;
        summarize_event(&line);
        events.push(line);
    }
    let status = child.wait()?;

    if let Some(record) = record {
        if let Some(parent) = record.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(record, events.join("\n") + "\n")?;
    }
    if !status.success() {
        snafu::whatever!("driver exited with {status}");
    }
    Ok(())
}

/// Payload to stdout; one terse summary line per event to stderr.
fn summarize_event(line: &str) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
        eprintln!("[driver] {line}");
        return;
    };
    match value["event"].as_str().unwrap_or("") {
        "text" => {
            if let Some(text) = value["text"].as_str() {
                println!("{text}");
            }
            eprintln!(
                "[stop] {} ({} tokens)",
                value["stop_reason"].as_str().unwrap_or("?"),
                value["ids"].as_array().map(|a| a.len()).unwrap_or(0),
            );
        }
        "timing" => eprintln!(
            "[timing] load {}s prefill {}s ({} tok/s) decode {}s ({} tok/s, {} steps)",
            value["load_s"],
            value["prefill_s"],
            value["prefill_tok_s"],
            value["decode_s"],
            value["decode_tok_s"],
            value["decode_steps"],
        ),
        "vram" => eprintln!(
            "[vram] max_allocated {} max_reserved {}",
            value["max_allocated"], value["max_reserved"],
        ),
        "meta" => eprintln!(
            "[meta] prompt {} tokens, device {} dtype {} attn {} fla_blocked {} determinism {} mode {}",
            value["prompt_tokens"],
            value["device"],
            value["dtype"],
            value["attn"],
            value["fla_blocked"],
            value["determinism"],
            value["mode"],
        ),
        "dump" => eprintln!(
            "[dump] {} rows x {} -> {} in {}s ({} rows/s)",
            value["rows"], value["vocab"], value["out"], value["forward_s"], value["rows_per_s"],
        ),
        "error" => eprintln!("[error] {}", value["message"]),
        other => eprintln!("[{other}] {line}"),
    }
}
