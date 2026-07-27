//! TLS locks: a real loopback handshake, and proof that BOTH ends
//! authenticate. A single-sided configuration comes up working, so the
//! refusals are the half worth testing.
use futures_util::{
    SinkExt,
    StreamExt,
};
use rustls_pki_types::pem::PemObject;
use tokio_util::codec::{
    FramedRead,
    FramedWrite,
};

use crate::tls::{
    client_config,
    loopback_server_name,
    server_config,
};
use crate::wire::{
    BitcodeCodec,
    ClientToServer,
    OpenRequest,
    OpenResponse,
    ServerToClient,
};
use crate::all::{
    ChatOptions,
    EraInfo,
};

/// A throwaway authority and leaf, so a lock depends on no installed
/// state and leaves none behind.
pub(crate) struct Material {
    root: std::path::PathBuf,
    pub(crate) files: sourcetrait_cert_lib::CertFiles,
}

impl Material {
    pub(crate) fn mint(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("bridge_tls_{name}"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch root");
        let text = format!(
            "name = \"{name}\"\n\
             organization = \"SourceTrait\"\n\
             validity_days = 30\n\
             \n\
             [authority]\n\
             common_name = \"SourceTrait Bridge Test CA\"\n\
             \n\
             [entity]\n\
             common_name = \"localhost\"\n\
             subject_alt_names = [\"127.0.0.1\", \"::1\", \"localhost\"]\n\
             usages = [\"server\", \"client\"]\n"
        );
        let config = sourcetrait_cert_lib::config::CertGenConfig::parse(
            &text,
            std::path::Path::new("test.toml"),
        )
        .expect("the test profile parses");
        let files = sourcetrait_cert_lib::generate(&config, &root).expect("mints");
        Self { root, files }
    }
}

impl Drop for Material {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn an_open_request() -> ClientToServer {
    ClientToServer::Open(OpenRequest {
        options: ChatOptions::default(),
    })
}

fn an_open_response() -> ServerToClient {
    ServerToClient::Open(OpenResponse {
        info: EraInfo {
            era: String::from("two"),
            model: String::from("test"),
        },
    })
}

/// Accept one connection, answer one request, and report what happened.
async fn serve_once(
    listener: tokio::net::TcpListener,
    config: rustls::ServerConfig,
) -> Result<ClientToServer, String> {
    let acceptor = tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(config));
    let (stream, _) = listener.accept().await.map_err(|e| e.to_string())?;
    let stream = acceptor.accept(stream).await.map_err(|e| e.to_string())?;
    let (read, write) = tokio::io::split(stream);
    let mut inbound = FramedRead::new(read, BitcodeCodec::<ClientToServer>::new());
    let mut outbound = FramedWrite::new(write, BitcodeCodec::<ServerToClient>::new());

    let request = inbound
        .next()
        .await
        .ok_or_else(|| String::from("the client sent nothing"))?
        .map_err(|e| e.to_string())?;
    outbound
        .send(an_open_response())
        .await
        .map_err(|e| e.to_string())?;
    Ok(request)
}

/// Connect, send one request, and read one answer back.
async fn exchange_once(
    address: std::net::SocketAddr,
    config: rustls::ClientConfig,
) -> Result<ServerToClient, String> {
    let connector = tokio_rustls::TlsConnector::from(std::sync::Arc::new(config));
    let stream = tokio::net::TcpStream::connect(address)
        .await
        .map_err(|e| e.to_string())?;
    let stream = connector
        .connect(loopback_server_name().map_err(|e| e.to_string())?, stream)
        .await
        .map_err(|e| e.to_string())?;
    let (read, write) = tokio::io::split(stream);
    let mut inbound = FramedRead::new(read, BitcodeCodec::<ServerToClient>::new());
    let mut outbound = FramedWrite::new(write, BitcodeCodec::<ClientToServer>::new());

    outbound
        .send(an_open_request())
        .await
        .map_err(|e| e.to_string())?;
    inbound
        .next()
        .await
        .ok_or_else(|| String::from("the server sent nothing"))?
        .map_err(|e| e.to_string())
}

async fn bind() -> (tokio::net::TcpListener, std::net::SocketAddr) {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("binds a loopback port");
    let address = listener.local_addr().expect("has an address");
    (listener, address)
}

/// The whole point, end to end: one leaf, presented at both ends, over a
/// real handshake, carrying a real frame each way.
#[tokio::test]
async fn one_leaf_authenticates_both_ends_and_frames_flow() {
    let material = Material::mint("agree");
    let (listener, address) = bind().await;

    let server = tokio::spawn(serve_once(
        listener,
        server_config(&material.files).expect("a server configuration"),
    ));
    let answer = exchange_once(
        address,
        client_config(&material.files).expect("a client configuration"),
    )
    .await
    .expect("the exchange completes");

    assert_eq!(answer, an_open_response(), "the answer survives the wire");
    assert_eq!(
        server.await.expect("the server task joins").expect("served"),
        an_open_request(),
        "the request survives the wire"
    );
}

/// The refusal that a single-sided example would not have caught. A
/// client presenting no certificate must be turned away by the server,
/// or the channel is authenticated in one direction only.
#[tokio::test]
async fn a_client_presenting_no_certificate_is_refused() {
    let material = Material::mint("noclient");
    let (listener, address) = bind().await;

    let roots = {
        let certificate = rustls_pki_types::CertificateDer::from_pem_file(
            &material.files.authority_public,
        )
        .expect("reads the authority");
        let mut roots = rustls::RootCertStore::empty();
        roots.add(certificate).expect("a usable authority");
        roots
    };
    let anonymous = rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("protocol versions")
    .with_root_certificates(roots)
    .with_no_client_auth();

    let server = tokio::spawn(serve_once(
        listener,
        server_config(&material.files).expect("a server configuration"),
    ));
    let outcome = exchange_once(address, anonymous).await;

    assert!(
        outcome.is_err(),
        "an anonymous client completed an exchange: {outcome:?}"
    );
    assert!(
        server.await.expect("the server task joins").is_err(),
        "the server must not have served an anonymous client"
    );
}

/// A leaf from another authority is refused, so trust is our own CA
/// rather than anything the platform happens to carry.
#[tokio::test]
async fn a_leaf_from_another_authority_is_refused() {
    let ours = Material::mint("ours");
    let theirs = Material::mint("theirs");
    let (listener, address) = bind().await;

    let server = tokio::spawn(serve_once(
        listener,
        server_config(&theirs.files).expect("a server configuration"),
    ));
    let outcome = exchange_once(
        address,
        client_config(&ours.files).expect("a client configuration"),
    )
    .await;

    assert!(
        outcome.is_err(),
        "a foreign authority was accepted: {outcome:?}"
    );
    let _ = server.await;
}

/// Absent material names the file rather than failing opaquely, since
/// this is what an operator meets when nothing has been minted.
#[test]
fn absent_material_names_what_is_missing() {
    let files = crate::tls::material(std::path::Path::new("/nonexistent"), "quest");
    let error = client_config(&files).expect_err("no material");
    assert!(
        error.to_string().contains("authority_quest.pem"),
        "{error}"
    );
}
