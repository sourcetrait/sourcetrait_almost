//! Client locks: a handle over a real connection, and the endings.
use futures_util::{
    SinkExt,
    StreamExt,
};
use tokio_util::codec::{
    FramedRead,
    FramedWrite,
};

use crate::all::{
    ChatOptions,
    EraInfo,
};
use crate::client::{
    TlsClientHandle,
    TlsClientOptions,
};
use crate::tests::tls::Material;
use crate::tls::server_config;
use crate::wire::{
    BitcodeCodec,
    ClientToServer,
    OpenRequest,
    OpenResponse,
    ServerToClient,
    TurnChunk,
};

fn an_open_response() -> ServerToClient {
    ServerToClient::Open(OpenResponse {
        info: EraInfo {
            era: String::from("two"),
            model: String::from("test"),
        },
    })
}

/// A server that answers each request with the next scripted message,
/// answers a Close with a Close, and reports what it received.
async fn serve(
    listener: tokio::net::TcpListener,
    config: rustls::ServerConfig,
    script: Vec<ServerToClient>,
) -> Vec<ClientToServer> {
    let acceptor = tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(config));
    let (stream, _) = listener.accept().await.expect("accepts");
    let stream = acceptor.accept(stream).await.expect("handshakes");
    let (read, write) = tokio::io::split(stream);
    let mut inbound = FramedRead::new(read, BitcodeCodec::<ClientToServer>::new());
    let mut outbound = FramedWrite::new(write, BitcodeCodec::<ServerToClient>::new());

    let mut seen = Vec::new();
    let mut script = script.into_iter();
    while let Some(Ok(message)) = inbound.next().await {
        let closing = message == ClientToServer::Close;
        seen.push(message);
        if closing {
            let _ = outbound.send(ServerToClient::Close).await;
            break;
        }
        if let Some(next) = script.next() {
            let _ = outbound.send(next).await;
        }
    }
    seen
}

/// A server that says its piece unprompted and then closes.
async fn announce_then_close(
    listener: tokio::net::TcpListener,
    config: rustls::ServerConfig,
    script: Vec<ServerToClient>,
) {
    let acceptor = tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(config));
    let (stream, _) = listener.accept().await.expect("accepts");
    let stream = acceptor.accept(stream).await.expect("handshakes");
    let (_read, write) = tokio::io::split(stream);
    let mut outbound = FramedWrite::new(write, BitcodeCodec::<ServerToClient>::new());
    for message in script {
        let _ = outbound.send(message).await;
    }
    let _ = outbound.send(ServerToClient::Close).await;
}

async fn bind() -> (tokio::net::TcpListener, std::net::SocketAddr) {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("binds a loopback port");
    let address = listener.local_addr().expect("has an address");
    (listener, address)
}

/// The whole handle, over a real connection: send, receive, close - and
/// the far end learns the close was deliberate.
#[tokio::test]
async fn a_handle_carries_a_request_and_its_answer_then_closes_politely() {
    let material = Material::mint("client_round");
    let (listener, address) = bind().await;
    let server = tokio::spawn(serve(
        listener,
        server_config(&material.files).expect("a server configuration"),
        vec![an_open_response()],
    ));

    let mut handle = TlsClientHandle::connect(TlsClientOptions {
        address,
        files: material.files.clone(),
    })
    .await
    .expect("connects");

    handle
        .send(ClientToServer::Open(OpenRequest {
            options: ChatOptions::default(),
        }))
        .await
        .expect("sends");
    let batch = handle.recv().await.expect("an answer arrives");
    assert_eq!(batch, vec![an_open_response()]);

    handle.close(std::time::Duration::from_secs(5)).await;

    let seen = server.await.expect("the server task joins");
    assert_eq!(
        seen.last(),
        Some(&ClientToServer::Close),
        "the far end was told, rather than reading a dropped socket"
    );
}

/// A server-initiated close ends the handle's stream, so a caller
/// looping on recv leaves rather than hanging.
#[tokio::test]
async fn a_server_close_ends_the_handles_stream() {
    let material = Material::mint("client_close");
    let (listener, address) = bind().await;
    let server = tokio::spawn(announce_then_close(
        listener,
        server_config(&material.files).expect("a server configuration"),
        vec![ServerToClient::TurnChunk(TurnChunk {
            text: String::from("half a thought"),
        })],
    ));

    let mut handle = TlsClientHandle::connect(TlsClientOptions {
        address,
        files: material.files.clone(),
    })
    .await
    .expect("connects");

    let mut received = Vec::new();
    while let Some(batch) = handle.recv().await {
        received.extend(batch);
    }
    assert_eq!(
        received,
        vec![ServerToClient::TurnChunk(TurnChunk {
            text: String::from("half a thought"),
        })],
        "everything before the close is delivered, and the close is not"
    );
    let _ = server.await;
}

/// Accept, handshake, then say nothing ever again.
async fn accept_then_go_silent(
    listener: tokio::net::TcpListener,
    config: rustls::ServerConfig,
) {
    let acceptor = tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(config));
    let (stream, _) = listener.accept().await.expect("accepts");
    let stream = acceptor.accept(stream).await.expect("handshakes");
    // Hold the connection open without reading or answering.
    std::future::pending::<()>().await;
    drop(stream);
}

/// The bound on the polite close, which is the branch nothing else
/// reaches. A peer that has gone silent never answers the goodbye, so
/// without the timeout `close` would wait the full grace period - and
/// without the abort it would wait forever on a peer that also never
/// drops the socket.
#[tokio::test]
async fn closing_against_a_silent_peer_is_bounded_by_its_timeout() {
    let material = Material::mint("client_silent");
    let (listener, address) = bind().await;
    let server = tokio::spawn(accept_then_go_silent(
        listener,
        server_config(&material.files).expect("a server configuration"),
    ));

    let mut handle = TlsClientHandle::connect(TlsClientOptions {
        address,
        files: material.files.clone(),
    })
    .await
    .expect("connects");

    let started = std::time::Instant::now();
    handle.close(std::time::Duration::from_millis(150)).await;
    let waited = started.elapsed();

    assert!(
        waited < std::time::Duration::from_secs(1),
        "close waited {waited:?}, so the timeout did not bound it"
    );
    server.abort();
}

/// A refused handshake is an ERROR FROM CONNECT rather than a message
/// the caller has to go looking for, which is why the handshake is
/// awaited before the task is spawned.
#[tokio::test]
async fn a_foreign_authority_fails_at_connect() {
    let ours = Material::mint("client_ours");
    let theirs = Material::mint("client_theirs");
    let (listener, address) = bind().await;
    let server = tokio::spawn(serve(
        listener,
        server_config(&theirs.files).expect("a server configuration"),
        Vec::new(),
    ));

    let outcome = TlsClientHandle::connect(TlsClientOptions {
        address,
        files: ours.files.clone(),
    })
    .await;

    assert!(outcome.is_err(), "a foreign authority connected");
    let error = outcome.err().expect("an error").to_string();
    assert!(error.contains("handshake"), "{error}");
    server.abort();
}
