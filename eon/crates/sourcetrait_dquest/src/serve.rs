//! The listener, and one session per accepted connection.
use crate::*;

/// Chunks are the hot path; bounded so a slow client slows the model.
const CHUNK_CAPACITY: usize = 256;

/// The framed halves of one accepted connection, named once.
type Reader = r::tokio::FramedRead<
    tokio::io::ReadHalf<r::tls::ServerStream<tokio::net::TcpStream>>,
    bridge::BitcodeCodec<bridge::ClientToServer>,
>;
type Writer = r::tokio::FramedWrite<
    tokio::io::WriteHalf<r::tls::ServerStream<tokio::net::TcpStream>>,
    bridge::BitcodeCodec<bridge::ServerToClient>,
>;

/// Take a loopback port, reporting the one actually bound.
pub(crate) async fn bind(
    address: std::net::SocketAddr,
) -> DquestResult<tokio::net::TcpListener> {
    match tokio::net::TcpListener::bind(address).await {
        Ok(listener) => Ok(listener),
        Err(source) => Err(DquestError::Listen { address, source }),
    }
}

/// Accept until the listener dies, giving each connection its own task.
///
/// One bad connection is that connection's problem: a handshake failure
/// or a peer that never speaks must not take the daemon down with it.
pub(crate) async fn serve(
    listener: tokio::net::TcpListener,
    config: rustls::ServerConfig,
    container: ContainerHandle,
    manager: manager::SessionManager,
) {
    let acceptor = r::tls::TlsAcceptor::from(std::sync::Arc::new(config));
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let acceptor = acceptor.clone();
        let container = container.clone();
        // Admitted BEFORE the handshake, so a peer that never completes
        // one is still visible as a connection being attempted - and the
        // ticket drops on that path exactly as on any other.
        let ticket = manager.admit();
        let log = manager.log(&ticket);
        // Every session in this space shares the one turn owner, so a
        // caller returning with an ask's result resumes the conversation
        // it left rather than starting a new one.
        let questness = manager.questness();
        tokio::spawn(async move {
            let _ticket = ticket;
            let Ok(stream) = acceptor.accept(stream).await else {
                return;
            };
            session(stream, container, questness, log).await;
        });
    }
}

/// Append one labelled record, if this session is logged at all.
///
/// A logging failure is dropped rather than reported, which is the same
/// rule the Questness side holds to: the log is a record of the work and
/// never part of it, so nothing a turn does depends on it landing.
fn record(
    log: &Option<log::SessionLog>,
    file: log::LogFile,
    label: &str,
    body: &str,
) {
    if let Some(log) = log {
        let _ = log.append(file, label, body);
    }
}

/// One connection's whole conversation.
pub(crate) async fn session(
    stream: r::tls::ServerStream<tokio::net::TcpStream>,
    container: ContainerHandle,
    questness: QuestnessHandle,
    log: Option<log::SessionLog>,
) {
    let (read, write) = tokio::io::split(stream);
    let mut reader = Reader::new(read, bridge::BitcodeCodec::new());
    let mut writer = Writer::new(write, bridge::BitcodeCodec::new());

    while let Some(Ok(message)) = reader.next().await {
        let carry_on = match message {
            bridge::ClientToServer::Open(request) => {
                let answer = match container.open(request.options).await {
                    Ok(info) => {
                        record(
                            &log,
                            log::LogFile::Transport,
                            "open",
                            &format!("{} {}", info.era, info.model),
                        );
                        bridge::ServerToClient::Open(bridge::OpenResponse { info })
                    }
                    Err(message) => {
                        record(&log, log::LogFile::Transport, "open-refused", &message);
                        bridge::ServerToClient::OpenRefused(
                            bridge::OpenRefusedResponse { message },
                        )
                    }
                };
                writer.send(answer).await.is_ok()
            }
            bridge::ClientToServer::Turn(request) => {
                turn(&mut reader, &mut writer, &container, &questness, request, &log).await
            }
            // Nothing is generating, or the turn arm would be running.
            bridge::ClientToServer::Cancel(_) => writer
                .send(bridge::ServerToClient::Cancel(bridge::CancelResponse {
                    stopped: false,
                }))
                .await
                .is_ok(),
            bridge::ClientToServer::Reset(_) => {
                let answer = match container.reset().await {
                    Ok(()) => {
                        record(&log, log::LogFile::Transport, "reset", "");
                        bridge::ServerToClient::Reset(bridge::ResetResponse)
                    }
                    Err(message) => {
                        record(&log, log::LogFile::Transport, "fault", &message);
                        bridge::ServerToClient::Notice(bridge::ServerNotice::Fault(
                            bridge::ServerFaultNotice { message },
                        ))
                    }
                };
                writer.send(answer).await.is_ok()
            }
            bridge::ClientToServer::Close => {
                record(&log, log::LogFile::Transport, "close", "");
                let _ = writer.send(bridge::ServerToClient::Close).await;
                false
            }
        };
        if !carry_on {
            break;
        }
    }
    let _ = writer.close().await;
}

/// Drive one turn to an ending, generating as many times as it takes.
///
/// A THINK NEVER LEAVES THIS LOOP. Questness runs it on the Thinkspace's
/// own evaluator and hands back the text to feed the model again, so one
/// wire turn can be several generations. Only an answer, an ask, an
/// insufficiency or a repair ends it.
///
/// THIS REQUIRES THE MULTI-THREADED RUNTIME. Questness is blocking all
/// the way down - its evaluator spawns a sized thread and joins it - so
/// the turn owner is driven through `block_in_place`, which panics on a
/// current-thread runtime rather than degrading.
async fn turn(
    reader: &mut Reader,
    writer: &mut Writer,
    container: &ContainerHandle,
    questness: &QuestnessHandle,
    request: bridge::InferRequest,
    log: &Option<log::SessionLog>,
) -> bool {
    // Held for the whole turn: the conversation is the Thinkspace's, so
    // a second turn interleaving generations would corrupt it.
    let mut owner = questness.lock().await;
    let mut feed = match tokio::task::block_in_place(|| owner.assemble(&request)) {
        Ok(text) => text,
        Err(message) => {
            teardown(container, log).await;
            return failed(writer, log, message).await;
        }
    };
    record(log, log::LogFile::Emission, "assembled", &feed);
    let keep = owner.keeps_conversation();

    // A bare call is a fresh conversation, always: anything standing - a
    // kept conversation, a stale pending ask - is torn down before the
    // model sees this turn. A continuation (`output`) resumes instead,
    // and a keep call continues what a keep call left.
    if request.output.is_none() && !keep {
        teardown(container, log).await;
    }

    loop {
        let (emission, report) = match generate(reader, writer, container, feed, log).await {
            Generated::Emitted { emission, report } => (emission, report),
            Generated::Failed(message) => {
                teardown(container, log).await;
                return failed(writer, log, message).await;
            }
            Generated::Gone => {
                teardown(container, log).await;
                return false;
            }
        };
        let step = match tokio::task::block_in_place(|| owner.step(&emission, false)) {
            Ok(step) => step,
            Err(message) => {
                teardown(container, log).await;
                return failed(writer, log, message).await;
            }
        };
        let answer = match step {
            // A think leaves a MARKER on the turn log and its text on the
            // emission log, so the turn file stays a readable outline of
            // what happened while the thought itself sits with the other
            // model-facing text.
            bridge::all::Step::Continue(next) => {
                record(log, log::LogFile::Turn, "think", "the turn generates again");
                record(log, log::LogFile::Emission, "thought", &next);
                feed = next;
                continue;
            }
            bridge::all::Step::Answered {
                output,
                text,
                config,
            } => bridge::InferResponse {
                output,
                text,
                config,
                report: Some(report),
                ..bridge::InferResponse::default()
            },
            // The turn PAUSES here. The caller runs it on its own engine
            // and returns the result on the next request, where this
            // Questness resumes the same conversation.
            bridge::all::Step::Ask { form, inputs } => bridge::InferResponse {
                nu: Some(form),
                inputs,
                ..bridge::InferResponse::default()
            },
            bridge::all::Step::Insufficient => bridge::InferResponse {
                insufficient: true,
                report: Some(report),
                ..bridge::InferResponse::default()
            },
            bridge::all::Step::Repair(rows) => {
                let message = rows
                    .iter()
                    .map(|row| format!("{}: {}", row.kind, row.message))
                    .collect::<Vec<String>>()
                    .join("; ");
                teardown(container, log).await;
                return failed(writer, log, message).await;
            }
        };
        record(log, log::LogFile::Turn, "answered", &summary(&answer));
        // An ask carries `nu` and pauses the conversation, so it is the
        // one response that never tears down; a terminal answer does
        // unless this turn's config kept it.
        let terminal = answer.nu.is_none();
        let delivered = writer
            .send(bridge::ServerToClient::Turn(answer))
            .await
            .is_ok();
        if terminal && !keep {
            teardown(container, log).await;
        }
        return delivered;
    }
}

/// Tear the engine's conversation down; the default end of every turn.
///
/// A reset that fails is a fault on the record rather than a failed
/// turn: the answer is already decided, and the fresh-start reset ahead
/// of the next bare call is the second line.
async fn teardown(container: &ContainerHandle, log: &Option<log::SessionLog>) {
    match container.reset().await {
        Ok(()) => record(log, log::LogFile::Turn, "teardown", ""),
        Err(message) => record(log, log::LogFile::Transport, "fault", &message),
    }
}

/// What one generation produced, or why it produced nothing.
enum Generated {
    Emitted {
        emission: String,
        report: bridge::all::TurnReport,
    },
    Failed(String),
    /// The peer went away, or the connection broke.
    Gone,
}

/// One generation: stream its chunks, and stay listening while it runs.
///
/// THE INBOUND HALF IS READ DURING THE TURN, which is what makes Cancel
/// reachable at all. The container serialises work, so a Cancel sent as
/// a unit would queue BEHIND the turn it means to stop and arrive after
/// it finished.
async fn generate(
    reader: &mut Reader,
    writer: &mut Writer,
    container: &ContainerHandle,
    text: String,
    log: &Option<log::SessionLog>,
) -> Generated {
    record(log, log::LogFile::Emission, "generate", &text);
    // Held open for the whole generation and written per chunk. This is
    // the ONLY record that appears while the model is still working -
    // every other one here is written after the generation has already
    // ended, which is what made a long turn indistinguishable from a
    // hung one.
    let mut streaming = log.as_ref().and_then(|log| log.chunks().ok());
    let (chunks, mut arriving) = r::tokio::channel(CHUNK_CAPACITY);
    let running = container.turn(text, chunks);
    tokio::pin!(running);

    // The emission is reassembled because Questness must parse the WHOLE
    // of it. The chunks still go out as they arrive, so a caller watching
    // raw output pays a copy rather than a wait.
    let mut answer = String::new();
    let mut report = None;
    let mut draining = false;
    let mut cancelled = false;
    let mut alive = true;

    while alive && !(report.is_some() && draining) {
        tokio::select! {
            chunk = arriving.recv(), if !draining => match chunk {
                Some(chunk) => {
                    answer.push_str(&chunk.text);
                    if let Some(sink) = streaming.as_mut() {
                        sink.write(&chunk.text);
                    }
                    alive = writer.send(bridge::ServerToClient::TurnChunk(chunk)).await.is_ok();
                }
                None => draining = true,
            },
            answer = &mut running, if report.is_none() => {
                report = Some(answer);
            }
            inbound = reader.next(), if !cancelled => match inbound {
                Some(Ok(bridge::ClientToServer::Cancel(_))) => {
                    cancelled = true;
                    // Closing the chunk channel IS the cancel: the next
                    // emit fails, so the engine stops and reports.
                    arriving.close();
                    alive = writer
                        .send(bridge::ServerToClient::Cancel(bridge::CancelResponse {
                            stopped: true,
                        }))
                        .await
                        .is_ok();
                }
                Some(Ok(bridge::ClientToServer::Close)) | None => {
                    cancelled = true;
                    arriving.close();
                    alive = false;
                }
                // A turn is already running; the second one is refused
                // rather than queued behind it.
                Some(Ok(_)) => {
                    alive = writer
                        .send(bridge::ServerToClient::Notice(bridge::ServerNotice::Fault(
                            bridge::ServerFaultNotice {
                                message: String::from("a turn is already generating"),
                            },
                        )))
                        .await
                        .is_ok();
                }
                Some(Err(_)) => {
                    cancelled = true;
                    arriving.close();
                    alive = false;
                }
            },
        }
    }

    // Logged whatever the ending was, including a peer that vanished
    // mid-turn: a partial emission is the interesting one to have kept.
    if let Some(sink) = streaming.as_mut() {
        sink.end();
    }
    record(log, log::LogFile::Emission, "emission", &answer);
    if !alive {
        return Generated::Gone;
    }
    match report {
        Some(Ok(report)) => {
            record(log, log::LogFile::Turn, "report", &accounting(&report));
            Generated::Emitted {
                emission: answer,
                report,
            }
        }
        Some(Err(message)) => Generated::Failed(message),
        None => Generated::Gone,
    }
}

/// Answer a turn that did not run, and say why.
async fn failed(
    writer: &mut Writer,
    log: &Option<log::SessionLog>,
    message: String,
) -> bool {
    record(log, log::LogFile::Turn, "turn-failed", &message);
    writer
        .send(bridge::ServerToClient::TurnFailed(
            bridge::TurnFailedResponse { message },
        ))
        .await
        .is_ok()
}

/// One generation's accounting, as the log records it.
fn accounting(report: &bridge::all::TurnReport) -> String {
    format!(
        "{:?}; prompt {}, generated {}, prefill {:.3}s, decode {:.3}s",
        report.finish,
        report.prompt_token_count,
        report.generated_token_count,
        report.prefill_seconds,
        report.decode_seconds,
    )
}

/// What a response carries, without rendering the values themselves.
fn summary(answer: &bridge::InferResponse) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if answer.output.is_some() {
        parts.push("output");
    }
    if answer.text.is_some() {
        parts.push("text");
    }
    if answer.config.is_some() {
        parts.push("config");
    }
    if answer.nu.is_some() {
        parts.push("ask");
    }
    if answer.insufficient {
        parts.push("insufficient");
    }
    match parts.is_empty() {
        true => String::from("nothing"),
        false => parts.join(", "),
    }
}