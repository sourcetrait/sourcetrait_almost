//! The channel-shaped chat engine: one named thread constructs and
//! privately holds the model (OlmoHybrid is not Send - the Rc graph
//! cache) and drives lib generations off the requests channel; only
//! Strings and small structs ride the channels.
use crate::*;

/// Event-channel bound: the engine blocks on a full channel, so a
/// stalled consumer backpressures generation instead of ballooning.
const EVENTS_CAPACITY: usize = 256;
/// Requests are small and consumer-paced.
const REQUESTS_CAPACITY: usize = 8;

/// Spawn the engine thread and hand back the session's channel pair.
/// The model loads ON the thread: Ready(EraInfo) follows the load,
/// Error + Closed follow a load failure.
pub(crate) fn open_chat(
    options: &bridge::all::ChatOptions,
) -> bridge::BridgeResult<bridge::all::ChatSession> {
    let (requests_tx, requests_rx) = r::mpsc::channel(REQUESTS_CAPACITY);
    let (events_tx, events_rx) = r::mpsc::channel(EVENTS_CAPACITY);
    let options = options.clone();
    let spawned = thread::Builder::new()
        .name(String::from("bridge-two-chat"))
        .spawn(move || engine(options, requests_rx, events_tx));
    if let Err(e) = spawned {
        snafu::whatever!("chat engine thread spawn failed: {e}");
    }
    Ok(bridge::all::ChatSession {
        requests: requests_tx,
        events: events_rx,
    })
}

/// What a finished turn tells the engine loop to do next.
enum TurnOutcome {
    Continue,
    Reset,
    Close,
}

/// The engine loop: load, announce Ready, then serve requests until
/// Close arrives or the requests sender drops. Post-load failures are
/// events - the session stays open and the thread never panics for
/// protocol reasons.
fn engine(
    options: bridge::all::ChatOptions,
    mut requests: r::mpsc::Receiver<bridge::all::ChatRequest>,
    events: r::mpsc::Sender<bridge::all::ChatEvent>,
) {
    let (mut model, tokenizer, model_coordinate) = match load(&options) {
        Ok(loaded) => loaded,
        Err(error) => {
            let _ = events.blocking_send(bridge::all::ChatEvent::Error {
                message: error.to_string(),
            });
            let _ = events.blocking_send(bridge::all::ChatEvent::Closed);
            return;
        }
    };
    if events
        .blocking_send(bridge::all::ChatEvent::Ready(bridge::all::EraInfo {
            era: String::from("two"),
            model: model_coordinate,
        }))
        .is_err()
    {
        return;
    }

    // The consumed-id trail: exactly the cache contents after each
    // turn (report.context_ids). Turns 2+ continue from it through
    // generate_from over a constructed RestoredContext - no snapshot
    // files (the bquest speculate-record precedent).
    let mut trail: Vec<u32> = Vec::new();
    while let Some(request) = requests.blocking_recv() {
        match request {
            bridge::all::ChatRequest::Turn { text } => {
                match run_turn(&mut model, &tokenizer, &mut requests, &events, &mut trail, &text) {
                    TurnOutcome::Continue => {}
                    TurnOutcome::Reset => reset(&mut model, &mut trail, &events),
                    TurnOutcome::Close => break,
                }
            }
            // Nothing is in flight between turns.
            bridge::all::ChatRequest::Cancel => {}
            bridge::all::ChatRequest::Reset => reset(&mut model, &mut trail, &events),
            bridge::all::ChatRequest::Close => break,
        }
    }
    let _ = events.blocking_send(bridge::all::ChatEvent::Closed);
}

/// Resolve the profiles and build the model on this thread. Chat is
/// always capture-off (chained sessions regrow the kv buffers, which
/// parks captured graph buffers for near-zero replay benefit - the
/// era-one ruling, enforced era-side); everything else rides the
/// profile.
fn load(
    options: &bridge::all::ChatOptions,
) -> lib::LibQuestResult<(lib::OlmoHybrid, tokenizers::Tokenizer, String)> {
    let config =
        lib::LibConfig::load_from_dir(options.dir.as_deref(), options.config.as_deref())?;
    let mut settings =
        lib::LibSettings::load_from_dir(options.dir.as_deref(), options.settings.as_deref())?;
    settings.graph = false;
    if let Some(sample_len) = options.sample_len {
        settings.generation.sample_len = sample_len;
    }

    #[cfg(feature = "cuda")]
    let (device, dtype) = (
        candle_core::Device::new_cuda(0)?,
        candle_core::DType::BF16,
    );
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let model_dir = config.model_dir();
    let checkpoint = lib::load_config(&model_dir)?;
    let tokenizer = lib::load_tokenizer(&model_dir)?;
    lib::verify_token_map(&tokenizer)?;
    let weights = lib::mmap_weights(&model_dir, dtype, &device)?;
    let model = lib::OlmoHybrid::new(&checkpoint, settings, weights)?;
    Ok((model, tokenizer, config.model))
}

/// Reset the conversation: clear the trail and the carried caches (a
/// fresh session on the same loaded model).
fn reset(
    model: &mut lib::OlmoHybrid,
    trail: &mut Vec<u32>,
    events: &r::mpsc::Sender<bridge::all::ChatEvent>,
) {
    trail.clear();
    if let Err(error) = model.clear_cache() {
        let _ = events.blocking_send(bridge::all::ChatEvent::Error {
            message: error.to_string(),
        });
    }
}

/// Drive one turn: start the generation (fresh or continued off the
/// trail), stream Chunks under the bounded events channel, poll
/// requests between decode steps for Cancel-class interrupts
/// (dropping the iterator is the clean early stop), then report
/// TurnDone and advance the trail.
fn run_turn(
    model: &mut lib::OlmoHybrid,
    tokenizer: &tokenizers::Tokenizer,
    requests: &mut r::mpsc::Receiver<bridge::all::ChatRequest>,
    events: &r::mpsc::Sender<bridge::all::ChatEvent>,
    trail: &mut Vec<u32>,
    text: &str,
) -> TurnOutcome {
    let generate_options = model.settings().generation.clone();
    let started = if trail.is_empty() {
        model.generate(tokenizer, text, &generate_options)
    } else {
        let restored = lib::RestoredContext {
            context_len: model.context_len(),
            context_ids: trail.clone(),
        };
        model.generate_from(tokenizer, &restored, text, &generate_options)
    };
    let mut generation = match started {
        Ok(generation) => generation,
        Err(error) => {
            let _ = events.blocking_send(bridge::all::ChatEvent::Error {
                message: error.to_string(),
            });
            return TurnOutcome::Continue;
        }
    };

    let mut cancelled = false;
    let mut outcome = TurnOutcome::Continue;
    for step in &mut generation {
        match step {
            Ok(step) => {
                if !step.chunk.is_empty()
                    && events
                        .blocking_send(bridge::all::ChatEvent::Chunk { text: step.chunk })
                        .is_err()
                {
                    // The consumer is gone: stop and close.
                    cancelled = true;
                    outcome = TurnOutcome::Close;
                    break;
                }
            }
            Err(error) => {
                // The iterator fused; the turn still reports below.
                let _ = events.blocking_send(bridge::all::ChatEvent::Error {
                    message: error.to_string(),
                });
                break;
            }
        }
        match requests.try_recv() {
            Ok(bridge::all::ChatRequest::Cancel) => {
                cancelled = true;
                break;
            }
            Ok(bridge::all::ChatRequest::Close) => {
                cancelled = true;
                outcome = TurnOutcome::Close;
                break;
            }
            Ok(bridge::all::ChatRequest::Reset) => {
                cancelled = true;
                outcome = TurnOutcome::Reset;
                break;
            }
            Ok(bridge::all::ChatRequest::Turn { .. }) => {
                let _ = events.blocking_send(bridge::all::ChatEvent::Error {
                    message: String::from("a turn is already generating"),
                });
            }
            Err(r::mpsc::TryRecvError::Empty) => {}
            Err(r::mpsc::TryRecvError::Disconnected) => {
                cancelled = true;
                outcome = TurnOutcome::Close;
                break;
            }
        }
    }

    let report = generation.finish();
    if !report.rest.is_empty() {
        let _ = events.blocking_send(bridge::all::ChatEvent::Chunk {
            text: report.rest.clone(),
        });
    }
    let finish = if cancelled {
        bridge::all::FinishReason::Cancelled
    } else {
        match report.finish_reason {
            Some(lib::FinishReason::StopToken) => bridge::all::FinishReason::StopToken,
            Some(lib::FinishReason::SampleLen) => bridge::all::FinishReason::SampleLen,
            // An error-broken turn ended early without a Cancel.
            None => bridge::all::FinishReason::Cancelled,
        }
    };
    let _ = events.blocking_send(bridge::all::ChatEvent::TurnDone(bridge::all::TurnReport {
        finish,
        prompt_token_count: report.prompt_token_count,
        generated_token_count: report.generated_token_count,
        prefill_seconds: report.prefill_seconds,
        decode_seconds: report.decode_seconds,
    }));
    *trail = report.context_ids;
    outcome
}
