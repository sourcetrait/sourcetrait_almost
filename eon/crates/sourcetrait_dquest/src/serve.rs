//! The listener, and one session per accepted connection.
//!
//! Built and locked, not yet reached from `run`: serving needs a model
//! container, and the container needs an engine, which is the era
//! library's and the next piece. The allow goes with that wiring.
#![allow(dead_code)]
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
) {
    let acceptor = r::tls::TlsAcceptor::from(std::sync::Arc::new(config));
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let acceptor = acceptor.clone();
        let container = container.clone();
        tokio::spawn(async move {
            let Ok(stream) = acceptor.accept(stream).await else {
                return;
            };
            session(stream, container).await;
        });
    }
}

/// One connection's whole conversation.
pub(crate) async fn session(
    stream: r::tls::ServerStream<tokio::net::TcpStream>,
    container: ContainerHandle,
) {
    let (read, write) = tokio::io::split(stream);
    let mut reader = Reader::new(read, bridge::BitcodeCodec::new());
    let mut writer = Writer::new(write, bridge::BitcodeCodec::new());

    while let Some(Ok(message)) = reader.next().await {
        let carry_on = match message {
            bridge::ClientToServer::Open(request) => {
                let answer = match container.open(request.options).await {
                    Ok(info) => bridge::ServerToClient::Open(bridge::OpenResponse { info }),
                    Err(message) => bridge::ServerToClient::OpenRefused(
                        bridge::OpenRefusedResponse { message },
                    ),
                };
                writer.send(answer).await.is_ok()
            }
            bridge::ClientToServer::Turn(request) => {
                turn(&mut reader, &mut writer, &container, request.text).await
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
                    Ok(()) => bridge::ServerToClient::Reset(bridge::ResetResponse),
                    Err(message) => bridge::ServerToClient::Notice(
                        bridge::ServerNotice::Fault(bridge::ServerFaultNotice { message }),
                    ),
                };
                writer.send(answer).await.is_ok()
            }
            bridge::ClientToServer::Close => {
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
) -> bool {
    let (chunks, mut arriving) = r::tokio::channel(CHUNK_CAPACITY);
    let running = container.turn(text, chunks);
    tokio::pin!(running);

    let mut report = None;
    let mut draining = false;
    let mut cancelled = false;
    let mut alive = true;

    while alive && !(report.is_some() && draining) {
        tokio::select! {
            chunk = arriving.recv(), if !draining => match chunk {
                Some(chunk) => {
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

    if !alive {
        return false;
    }
    let answer = match report {
        Some(Ok(report)) => bridge::ServerToClient::Turn(bridge::TurnResponse { report }),
        Some(Err(message)) => {
            bridge::ServerToClient::TurnFailed(bridge::TurnFailedResponse { message })
        }
        None => return false,
    };
    writer.send(answer).await.is_ok()
}
