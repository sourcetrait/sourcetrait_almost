//! The chat-engine drill: the three live behaviors the camp TUI
//! rides - a short exchange, a multi-turn continuation, and a
//! mid-turn Cancel - through the real checkpoint on gpu.
#![cfg(feature = "cuda")]
use crate::*;

use crate::bridge::all::Era;

/// Poll the events receiver off-runtime with a deadline.
fn next_event(
    events: &mut r::mpsc::Receiver<bridge::all::ChatEvent>,
    deadline: std::time::Duration,
) -> Option<bridge::all::ChatEvent> {
    let started = std::time::Instant::now();
    loop {
        match events.try_recv() {
            Ok(event) => return Some(event),
            Err(r::mpsc::TryRecvError::Disconnected) => return None,
            Err(r::mpsc::TryRecvError::Empty) => {
                if started.elapsed() > deadline {
                    return None;
                }
                thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
}

/// Drain events until TurnDone, tallying chunks.
fn drain_turn(
    events: &mut r::mpsc::Receiver<bridge::all::ChatEvent>,
) -> (usize, Option<bridge::all::TurnReport>) {
    let mut chunks = 0usize;
    loop {
        match next_event(events, std::time::Duration::from_secs(120)) {
            Some(bridge::all::ChatEvent::Chunk { .. }) => chunks += 1,
            Some(bridge::all::ChatEvent::TurnDone(report)) => return (chunks, Some(report)),
            Some(bridge::all::ChatEvent::Error { message }) => {
                panic!("engine error mid-turn: {message}")
            }
            Some(other) => panic!("unexpected event mid-turn: {other:?}"),
            None => return (chunks, None),
        }
    }
}

/// The drill runs deterministic and fixed-length: a one-off settings
/// toml (greedy + ignore_stops + a small budget) makes every plain
/// turn decode exactly `sample_len` tokens, so the mid-turn Cancel
/// always has a live window to land in.
#[test]
#[ignore]
fn chat_drill_exchange_continuation_cancel_cuda() {
    let sample_len = 48usize;
    let scratch = std::env::temp_dir().join(format!("bridge_two_drill_{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("scratch dir");
    let settings_path = scratch.join("lib.toml");
    std::fs::write(
        &settings_path,
        "[generation]\ngreedy = true\nignore_stops = true\nsample_len = 48\n",
    )
    .expect("settings toml");

    let era = BridgeTwo;
    let info = era.info();
    assert_eq!(info.era, "two");
    let mut session = era
        .open_chat(&bridge::all::ChatOptions {
            dir: None,
            config: None,
            settings: Some(settings_path.display().to_string()),
            sample_len: None,
        })
        .expect("open_chat");

    // Ready follows the load (generous first-load deadline).
    match next_event(&mut session.events, std::time::Duration::from_secs(300)) {
        Some(bridge::all::ChatEvent::Ready(ready)) => assert_eq!(ready.era, "two"),
        other => panic!("expected Ready, got {other:?}"),
    }

    // Turn 1: the short exchange.
    session
        .requests
        .blocking_send(bridge::all::ChatRequest::Turn {
            text: String::from("Reply with the word ok."),
        })
        .expect("turn 1 send");
    let (chunks, report) = drain_turn(&mut session.events);
    let report = report.expect("turn 1 report");
    assert!(chunks > 0, "turn 1 produced no chunks");
    assert_eq!(report.finish, bridge::all::FinishReason::SampleLen);
    assert_eq!(report.generated_token_count, sample_len);

    // Turn 2: the continuation (generate_from off the carried trail -
    // a trail/cache mismatch would error here, not report).
    session
        .requests
        .blocking_send(bridge::all::ChatRequest::Turn {
            text: String::from("Now reply with the word again."),
        })
        .expect("turn 2 send");
    let (chunks, report) = drain_turn(&mut session.events);
    let report = report.expect("turn 2 report");
    assert!(chunks > 0, "turn 2 produced no chunks");
    assert_eq!(report.finish, bridge::all::FinishReason::SampleLen);
    assert_eq!(report.generated_token_count, sample_len);

    // Turn 3: Cancel mid-turn after the first chunks land.
    session
        .requests
        .blocking_send(bridge::all::ChatRequest::Turn {
            text: String::from("Count upward from one."),
        })
        .expect("turn 3 send");
    let mut seen = 0usize;
    while seen < 3 {
        match next_event(&mut session.events, std::time::Duration::from_secs(60)) {
            Some(bridge::all::ChatEvent::Chunk { .. }) => seen += 1,
            Some(bridge::all::ChatEvent::TurnDone(report)) => {
                panic!("turn 3 finished before the cancel window: {report:?}")
            }
            other => panic!("unexpected event awaiting chunks: {other:?}"),
        }
    }
    session
        .requests
        .blocking_send(bridge::all::ChatRequest::Cancel)
        .expect("cancel send");
    let (_, report) = drain_turn(&mut session.events);
    let report = report.expect("turn 3 report");
    assert_eq!(report.finish, bridge::all::FinishReason::Cancelled);
    assert!(
        report.generated_token_count < sample_len,
        "cancel landed after the full budget ({} tokens)",
        report.generated_token_count
    );

    // Close: the engine answers Closed and exits.
    session
        .requests
        .blocking_send(bridge::all::ChatRequest::Close)
        .expect("close send");
    match next_event(&mut session.events, std::time::Duration::from_secs(60)) {
        Some(bridge::all::ChatEvent::Closed) => {}
        other => panic!("expected Closed, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&scratch);
}
