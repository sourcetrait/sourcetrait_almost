//! Serve locks: the daemon's own end, driven by the real bridge client
//! over a real connection, with the world behind the engine as a
//! parameter rather than a double.
use crate::container::{
    ContainerHandle,
    Engine,
};
use crate::serve::{
    bind,
    serve,
};

/// An engine whose whole behaviour is its script, so a lock says what
/// the daemon does rather than what a model happens to.
struct Scripted {
    era: String,
    chunks: Vec<String>,
    refuse_open: bool,
    fail_turn: bool,
    resets: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Engine for Scripted {
    fn open(
        &mut self,
        _options: &sourcetrait_quest_bridge::all::ChatOptions,
    ) -> Result<sourcetrait_quest_bridge::all::EraInfo, String> {
        if self.refuse_open {
            return Err(String::from("the checkpoint did not load"));
        }
        Ok(sourcetrait_quest_bridge::all::EraInfo {
            era: self.era.clone(),
            model: String::from("scripted"),
        })
    }

    fn turn(
        &mut self,
        _text: &str,
        chunk: &mut dyn FnMut(sourcetrait_quest_bridge::TurnChunk) -> bool,
    ) -> Result<sourcetrait_quest_bridge::all::TurnReport, String> {
        if self.fail_turn {
            return Err(String::from("the engine faulted mid-generation"));
        }
        let mut emitted = 0usize;
        for text in &self.chunks {
            if !chunk(sourcetrait_quest_bridge::TurnChunk { text: text.clone() }) {
                break;
            }
            emitted += 1;
        }
        Ok(sourcetrait_quest_bridge::all::TurnReport {
            finish: sourcetrait_quest_bridge::all::FinishReason::StopToken,
            prompt_token_count: 7,
            generated_token_count: emitted,
            prefill_seconds: 0.25,
            decode_seconds: 0.5,
        })
    }

    fn reset(&mut self) -> Result<(), String> {
        self.resets
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

/// Stand the daemon up on a loopback port and hand back the way in.
async fn daemon(
    material: &crate::tests::material::Material,
    engine: Scripted,
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<()>,
    crate::manager::SessionManager,
) {
    let listener = bind(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .expect("binds");
    let address = listener.local_addr().expect("has an address");
    let config =
        sourcetrait_quest_bridge::server_config(&material.files).expect("a server configuration");
    let container = ContainerHandle::spawn(move || engine);
    let manager = crate::manager::SessionManager::for_user("box");
    let task = tokio::spawn(serve(listener, config, container, manager.clone()));
    (address, task, manager)
}

async fn connect(
    address: std::net::SocketAddr,
    material: &crate::tests::material::Material,
) -> sourcetrait_quest_bridge::TlsClientHandle {
    sourcetrait_quest_bridge::TlsClientHandle::connect(
        sourcetrait_quest_bridge::TlsClientOptions {
            address,
            files: material.files.clone(),
        },
    )
    .await
    .expect("connects")
}

/// Read until one message satisfies the predicate, collecting the rest.
async fn read_until(
    handle: &mut sourcetrait_quest_bridge::TlsClientHandle,
    mut done: impl FnMut(&sourcetrait_quest_bridge::ServerToClient) -> bool,
) -> Vec<sourcetrait_quest_bridge::ServerToClient> {
    let mut seen = Vec::new();
    while let Some(batch) = handle.recv().await {
        let mut finished = false;
        for message in batch {
            finished |= done(&message);
            seen.push(message);
        }
        if finished {
            break;
        }
    }
    seen
}

fn scripted(chunks: &[&str]) -> Scripted {
    Scripted {
        era: String::from("two"),
        chunks: chunks.iter().map(|c| (*c).to_string()).collect(),
        refuse_open: false,
        fail_turn: false,
        resets: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    }
}

/// The whole round trip through both halves: open, a turn that streams,
/// and its accounting.
#[tokio::test]
async fn a_turn_streams_its_chunks_then_reports() {
    let material = crate::tests::material::Material::mint("dquest_turn");
    let (address, server, _manager) = daemon(&material, scripted(&["Hel", "lo", " there"])).await;
    let mut handle = connect(address, &material).await;

    handle
        .send(sourcetrait_quest_bridge::ClientToServer::Open(
            sourcetrait_quest_bridge::OpenRequest {
                options: sourcetrait_quest_bridge::all::ChatOptions::default(),
            },
        ))
        .await
        .expect("sends");
    let opened = read_until(&mut handle, |m| {
        matches!(m, sourcetrait_quest_bridge::ServerToClient::Open(_))
    })
    .await;
    assert_eq!(opened.len(), 1, "{opened:?}");

    handle
        .send(sourcetrait_quest_bridge::ClientToServer::Turn(
            sourcetrait_quest_bridge::TurnRequest {
                text: String::from("say hello"),
            },
        ))
        .await
        .expect("sends");
    let turn = read_until(&mut handle, |m| {
        matches!(m, sourcetrait_quest_bridge::ServerToClient::Turn(_))
    })
    .await;

    let text: String = turn
        .iter()
        .filter_map(|m| match m {
            sourcetrait_quest_bridge::ServerToClient::TurnChunk(chunk) => Some(chunk.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello there", "the chunks arrive whole and in order");

    let Some(sourcetrait_quest_bridge::ServerToClient::Turn(response)) = turn.last() else {
        panic!("the turn ends with its own response: {turn:?}");
    };
    assert_eq!(
        response.report.generated_token_count, 3,
        "the accounting follows the chunks"
    );

    handle.close(std::time::Duration::from_secs(5)).await;
    server.abort();
}

/// Several clients at once against the one container, and the registry
/// sees each of them arrive and leave.
#[tokio::test]
async fn several_sessions_share_one_container() {
    let material = crate::tests::material::Material::mint("dquest_several");
    let (address, server, manager) = daemon(&material, scripted(&["one"])).await;

    let mut handles = Vec::new();
    for _ in 0..3 {
        let mut handle = connect(address, &material).await;
        handle
            .send(sourcetrait_quest_bridge::ClientToServer::Open(
                sourcetrait_quest_bridge::OpenRequest {
                    options: sourcetrait_quest_bridge::all::ChatOptions::default(),
                },
            ))
            .await
            .expect("sends");
        let opened = read_until(&mut handle, |m| {
            matches!(m, sourcetrait_quest_bridge::ServerToClient::Open(_))
        })
        .await;
        assert_eq!(opened.len(), 1, "each session opens against the container");
        handles.push(handle);
    }

    assert_eq!(manager.live(), 3, "the registry sees all three");
    assert_eq!(manager.sessions().len(), 3, "and can name them");

    for mut handle in handles {
        handle.close(std::time::Duration::from_secs(5)).await;
    }
    // The tickets drop with their tasks, which the close only starts.
    for _ in 0..200 {
        if manager.live() == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(manager.live(), 0, "every place was given back");
    server.abort();
}

/// One bad connection is that connection's problem. A peer that fails
/// the handshake must end its own task and nothing else, or a single
/// malformed client is a denial of service.
#[tokio::test]
async fn a_failed_handshake_does_not_take_the_daemon_down() {
    let material = crate::tests::material::Material::mint("dquest_survives");
    let foreign = crate::tests::material::Material::mint("dquest_foreign");
    let (address, server, manager) = daemon(&material, scripted(&["after"])).await;

    let refused = sourcetrait_quest_bridge::TlsClientHandle::connect(
        sourcetrait_quest_bridge::TlsClientOptions {
            address,
            files: foreign.files.clone(),
        },
    )
    .await;
    assert!(refused.is_err(), "a foreign authority was admitted");

    // The daemon keeps serving, and the refused peer left no place behind.
    let mut handle = connect(address, &material).await;
    handle
        .send(sourcetrait_quest_bridge::ClientToServer::Open(
            sourcetrait_quest_bridge::OpenRequest {
                options: sourcetrait_quest_bridge::all::ChatOptions::default(),
            },
        ))
        .await
        .expect("sends");
    let opened = read_until(&mut handle, |m| {
        matches!(m, sourcetrait_quest_bridge::ServerToClient::Open(_))
    })
    .await;
    assert_eq!(opened.len(), 1, "the daemon still serves");

    handle.close(std::time::Duration::from_secs(5)).await;
    for _ in 0..200 {
        if manager.live() == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        manager.live(),
        0,
        "the refused peer's ticket dropped with its task"
    );
    server.abort();
}

/// A refused open is its own message, so a caller is not left reading a
/// fault notice to learn its session never started.
#[tokio::test]
async fn a_refused_open_answers_the_open() {
    let material = crate::tests::material::Material::mint("dquest_refused");
    let mut engine = scripted(&[]);
    engine.refuse_open = true;
    let (address, server, _manager) = daemon(&material, engine).await;
    let mut handle = connect(address, &material).await;

    handle
        .send(sourcetrait_quest_bridge::ClientToServer::Open(
            sourcetrait_quest_bridge::OpenRequest {
                options: sourcetrait_quest_bridge::all::ChatOptions::default(),
            },
        ))
        .await
        .expect("sends");
    let seen = read_until(&mut handle, |m| {
        matches!(
            m,
            sourcetrait_quest_bridge::ServerToClient::OpenRefused(_)
        )
    })
    .await;
    assert!(matches!(
        seen.last(),
        Some(sourcetrait_quest_bridge::ServerToClient::OpenRefused(_))
    ));

    handle.close(std::time::Duration::from_secs(5)).await;
    server.abort();
}

/// A failed turn answers the TURN rather than arriving as a notice, so a
/// caller knows which request died.
#[tokio::test]
async fn a_failed_turn_answers_the_turn() {
    let material = crate::tests::material::Material::mint("dquest_failed");
    let mut engine = scripted(&[]);
    engine.fail_turn = true;
    let (address, server, _manager) = daemon(&material, engine).await;
    let mut handle = connect(address, &material).await;

    handle
        .send(sourcetrait_quest_bridge::ClientToServer::Turn(
            sourcetrait_quest_bridge::TurnRequest {
                text: String::from("anything"),
            },
        ))
        .await
        .expect("sends");
    let seen = read_until(&mut handle, |m| {
        matches!(m, sourcetrait_quest_bridge::ServerToClient::TurnFailed(_))
    })
    .await;
    assert!(matches!(
        seen.last(),
        Some(sourcetrait_quest_bridge::ServerToClient::TurnFailed(_))
    ));

    handle.close(std::time::Duration::from_secs(5)).await;
    server.abort();
}

/// The reason the session reads inbound DURING a turn. A cancel sent as
/// ordinary work would queue behind the very turn it means to stop.
#[tokio::test]
async fn a_cancel_reaches_a_running_turn() {
    let material = crate::tests::material::Material::mint("dquest_cancel");
    let many: Vec<&str> = vec!["tick"; 100_000];
    let (address, server, _manager) = daemon(&material, scripted(&many)).await;
    let mut handle = connect(address, &material).await;

    handle
        .send(sourcetrait_quest_bridge::ClientToServer::Turn(
            sourcetrait_quest_bridge::TurnRequest {
                text: String::from("go on at length"),
            },
        ))
        .await
        .expect("sends");
    // Wait for it to be genuinely under way before interrupting.
    let _ = read_until(&mut handle, |m| {
        matches!(m, sourcetrait_quest_bridge::ServerToClient::TurnChunk(_))
    })
    .await;
    handle
        .send(sourcetrait_quest_bridge::ClientToServer::Cancel(
            sourcetrait_quest_bridge::CancelRequest,
        ))
        .await
        .expect("sends");

    let seen = read_until(&mut handle, |m| {
        matches!(m, sourcetrait_quest_bridge::ServerToClient::Turn(_))
    })
    .await;
    assert!(
        seen.iter().any(|m| matches!(
            m,
            sourcetrait_quest_bridge::ServerToClient::Cancel(response) if response.stopped
        )),
        "the cancel was answered while the turn ran"
    );

    let Some(sourcetrait_quest_bridge::ServerToClient::Turn(response)) = seen.last() else {
        panic!("a cancelled turn still reports");
    };
    assert!(
        response.report.generated_token_count < many.len(),
        "the turn stopped early rather than running to its end"
    );

    handle.close(std::time::Duration::from_secs(5)).await;
    server.abort();
}
