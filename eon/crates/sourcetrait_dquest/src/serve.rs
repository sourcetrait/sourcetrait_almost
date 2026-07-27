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
        tokio::spawn(async move {
            let _ticket = ticket;
            let Ok(stream) = acceptor.accept(stream).await else {
                return;
            };
            session(stream, container, log).await;
        });
    }
}

/// Append one labelled record, if this session is logged at all.
///
/// A logging failure is dropped rather than reported, which is the same
/// rule the Questness side holds to: the log is a record of the work and
/// never part of it, so nothing a turn does depends on it landing.
fn record(log: &Option<log::SessionLog>, label: &str, body: &str) {
    if let Some(log) = log {
        let _ = log.append(label, body);
    }
}

/// One connection's whole conversation.
pub(crate) async fn session(
    stream: r::tls::ServerStream<tokio::net::TcpStream>,
    container: ContainerHandle,
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
                        record(&log, "open", &format!("{} {}", info.era, info.model));
                        bridge::ServerToClient::Open(bridge::OpenResponse { info })
                    }
                    Err(message) => {
                        record(&log, "open-refused", &message);
                        bridge::ServerToClient::OpenRefused(
                            bridge::OpenRefusedResponse { message },
                        )
                    }
                };
                writer.send(answer).await.is_ok()
            }
            bridge::ClientToServer::Turn(request) => {
                turn(&mut reader, &mut writer, &container, request.text, &log).await
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
                        record(&log, "reset", "");
                        bridge::ServerToClient::Reset(bridge::ResetResponse)
                    }
                    Err(message) => {
                        record(&log, "fault", &message);
                        bridge::ServerToClient::Notice(bridge::ServerNotice::Fault(
                            bridge::ServerFaultNotice { message },
                        ))
                    }
                };
                writer.send(answer).await.is_ok()
            }
            bridge::ClientToServer::Close => {
                record(&log, "close", "");
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

/// Drive one turn: stream its chunks, and stay listening while it runs.
///
/// THE INBOUND HALF IS READ DURING THE TURN, which is what makes Cancel
/// reachable at all. The container serialises work, so a Cancel sent as
/// a unit would queue BEHIND the turn it means to stop and arrive after
/// it finished.
async fn turn(
    reader: &mut Reader,
    writer: &mut Writer,
    container: &ContainerHandle,
    text: String,
    log: &Option<log::SessionLog>,
) -> bool {
    record(log, "turn", &text);
    let (chunks, mut arriving) = r::tokio::channel(CHUNK_CAPACITY);
    let running = container.turn(text, chunks);
    tokio::pin!(running);

    // The answer is reassembled here ONLY to log it. The chunks still go
    // out as they arrive, so the log costs a copy rather than a wait.
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
    // mid-turn: a partial answer is the interesting one to have kept.
    record(log, "answer", &answer);
    if !alive {
        return false;
    }
    let outcome = match report {
        Some(Ok(report)) => {
            record(
                log,
                "report",
                &format!(
                    "{:?}; prompt {}, generated {}, prefill {:.3}s, decode {:.3}s",
                    report.finish,
                    report.prompt_token_count,
                    report.generated_token_count,
                    report.prefill_seconds,
                    report.decode_seconds,
                ),
            );
            bridge::ServerToClient::Turn(bridge::TurnResponse { report })
        }
        Some(Err(message)) => {
            record(log, "turn-failed", &message);
            bridge::ServerToClient::TurnFailed(bridge::TurnFailedResponse { message })
        }
        None => return false,
    };
    writer.send(outcome).await.is_ok()
}