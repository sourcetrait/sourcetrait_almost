//! The client half: one connection, driven by a task behind a handle.
use crate::*;

/// Requests are consumer-paced and small.
const OUTBOUND_CAPACITY: usize = 8;

/// Inbound is BOUNDED deliberately. A full channel stops the reader,
/// which stops reading the socket, which pushes back through TCP to the
/// daemon's own generation loop - so a slow consumer slows the model
/// rather than growing a queue without bound.
const INBOUND_CAPACITY: usize = 256;

/// How many messages one `recv` takes at most.
const RECV_BATCH: usize = 16;

/// How long to wait for the far end to answer a close.
const GOODBYE: std::time::Duration = std::time::Duration::from_secs(2);

/// The framed halves of one connection, named once.
type Reader = r::tokio::FramedRead<
    tokio::io::ReadHalf<r::tls::ClientStream<tokio::net::TcpStream>>,
    BitcodeCodec<ServerToClient>,
>;
type Writer = r::tokio::FramedWrite<
    tokio::io::WriteHalf<r::tls::ClientStream<tokio::net::TcpStream>>,
    BitcodeCodec<ClientToServer>,
>;

/// Where to connect, and the material to present on the way in.
pub struct TlsClientOptions {
    pub address: std::net::SocketAddr,
    pub files: srcert::CertFiles,
}

/// The connection task: it owns the stream for the connection's life.
struct TlsClient {
    cancel: r::tokio::CancellationToken,
    inbound: r::tokio::Sender<ServerToClient>,
    outbound: r::tokio::Receiver<ClientToServer>,
    stream: r::tls::ClientStream<tokio::net::TcpStream>,
}

/// What a caller holds: send, receive, and close.
///
/// Separate from the task so the connection has exactly one owner - a
/// caller cannot reach the stream, so it cannot be torn out from under a
/// write in progress.
pub struct TlsClientHandle {
    cancel: r::tokio::CancellationToken,
    outbound: r::tokio::Sender<ClientToServer>,
    inbound: r::tokio::Receiver<ServerToClient>,
    task: Option<tokio::task::JoinHandle<BridgeResult<()>>>,
}

impl TlsClientHandle {
    /// Connect, handshake, and put a task behind the connection.
    ///
    /// The handshake is awaited HERE rather than inside the task, so a
    /// refused certificate is an error the caller receives rather than a
    /// message it has to go looking for.
    pub async fn connect(options: TlsClientOptions) -> BridgeResult<Self> {
        let config = tls::client_config(&options.files)?;
        let connector = r::tls::TlsConnector::from(std::sync::Arc::new(config));
        let stream = match tokio::net::TcpStream::connect(options.address).await {
            Ok(stream) => stream,
            Err(error) => snafu::whatever!("connecting to {}: {error}", options.address),
        };
        let stream = match connector.connect(tls::loopback_server_name()?, stream).await {
            Ok(stream) => stream,
            Err(error) => {
                snafu::whatever!("the handshake with {} failed: {error}", options.address)
            }
        };

        let (outbound_tx, outbound_rx) = r::tokio::channel(OUTBOUND_CAPACITY);
        let (inbound_tx, inbound_rx) = r::tokio::channel(INBOUND_CAPACITY);
        let cancel = r::tokio::CancellationToken::new();
        let client = TlsClient {
            cancel: cancel.clone(),
            inbound: inbound_tx,
            outbound: outbound_rx,
            stream,
        };
        Ok(Self {
            cancel,
            outbound: outbound_tx,
            inbound: inbound_rx,
            task: Some(tokio::spawn(client.run())),
        })
    }

    /// Hand one message to the connection.
    pub async fn send(&self, message: ClientToServer) -> BridgeResult<()> {
        match self.outbound.send(message).await {
            Ok(()) => Ok(()),
            Err(_) => snafu::whatever!("the connection is gone"),
        }
    }

    /// Take whatever has arrived, or None once the connection is over.
    pub async fn recv(&mut self) -> Option<Vec<ServerToClient>> {
        let mut batch = Vec::new();
        match self.inbound.recv_many(&mut batch, RECV_BATCH).await {
            0 => None,
            _ => Some(batch),
        }
    }

    /// Close politely, then stop waiting.
    ///
    /// A graceful close means telling the far end and hearing back, so a
    /// peer that has already gone would otherwise hold this forever. The
    /// timeout is what bounds that, and the task is aborted only if it
    /// outstays it.
    pub async fn close(&mut self, timeout: std::time::Duration) {
        let Some(task) = self.task.take() else {
            return;
        };
        self.cancel.cancel();
        let abort = task.abort_handle();
        if tokio::time::timeout(timeout, task).await.is_err() {
            abort.abort();
        }
    }
}

impl Drop for TlsClientHandle {
    /// Dropping the handle ends the connection rather than leaking it.
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl TlsClient {
    async fn run(self) -> BridgeResult<()> {
        let Self {
            cancel,
            inbound,
            mut outbound,
            stream,
        } = self;
        let (read, write) = tokio::io::split(stream);
        let mut reader = Reader::new(read, BitcodeCodec::<ServerToClient>::new());
        let mut writer = Writer::new(write, BitcodeCodec::<ClientToServer>::new());
        let mut failure = None;

        loop {
            tokio::select! {
                () = cancel.cancelled() => {
                    say_goodbye(&mut reader, &mut writer).await;
                    break;
                }
                message = reader.next() => match message {
                    Some(Ok(ServerToClient::Close)) => {
                        let _ = writer.send(ClientToServer::Close).await;
                        break;
                    }
                    Some(Ok(message)) => {
                        if inbound.send(message).await.is_err() {
                            break;
                        }
                    }
                    Some(Err(error)) => {
                        failure = Some(error.to_string());
                        break;
                    }
                    None => break,
                },
                request = outbound.recv() => match request {
                    Some(request) => {
                        if writer.send(request).await.is_err() {
                            break;
                        }
                    }
                    // Every sender dropped: the handle is gone, which is
                    // the implicit teardown.
                    None => {
                        say_goodbye(&mut reader, &mut writer).await;
                        break;
                    }
                },
            }
        }

        let _ = writer.close().await;
        if let Some(failure) = failure {
            snafu::whatever!("the connection failed: {failure}");
        }
        Ok(())
    }
}

/// Say Close and wait briefly to hear it back, so the far end learns
/// this was deliberate rather than reading a dropped socket.
async fn say_goodbye(reader: &mut Reader, writer: &mut Writer) {
    if writer.send(ClientToServer::Close).await.is_err() {
        return;
    }
    let _ = tokio::time::timeout(GOODBYE, async {
        while let Some(message) = reader.next().await {
            match message {
                Ok(ServerToClient::Close) | Err(_) => break,
                Ok(_) => continue,
            }
        }
    })
    .await;
}
