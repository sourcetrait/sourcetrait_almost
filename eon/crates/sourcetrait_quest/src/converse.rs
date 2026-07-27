//! The conversation with the daemon, driven to one answer.
use crate::*;

/// How long a polite close waits for the far end to answer it.
const GOODBYE: std::time::Duration = std::time::Duration::from_secs(2);

/// Ask one turn and drive it, sub-turns included, to an answer.
///
/// The turn loop runs HERE rather than in the daemon, which is where the
/// design puts it. What the current wire carries is a text turn and a
/// text answer, so a `TurnRequest` cannot yet carry a config, a prompt
/// and its bindings - moving the loop across is that wire change rather
/// than a rewrite of this function.
pub(crate) fn ask(
    plugin: &QuestPlugin,
    request: &lib::Request,
) -> QuestPluginResult<lib::Answer> {
    let mut questness = plugin.questness();
    let assembled = questness.assemble(request)?;
    plugin
        .runtime()
        .block_on(converse(&mut questness, assembled.text))
}

/// Connect, open, and step until the turn ends.
async fn converse<H: lib::harness::ClientHarness>(
    questness: &mut lib::Questness<H>,
    opening: String,
) -> QuestPluginResult<lib::Answer> {
    let mut client = connect().await?;
    let outcome = drive(questness, &mut client, opening).await;
    // Closed on every path, including a failed one: a daemon reading a
    // dropped socket cannot tell a crashed client from a finished one.
    client.close(GOODBYE).await;
    outcome
}

/// The step loop, once a session is open.
async fn drive<H: lib::harness::ClientHarness>(
    questness: &mut lib::Questness<H>,
    client: &mut bridge::TlsClientHandle,
    opening: String,
) -> QuestPluginResult<lib::Answer> {
    open(client).await?;
    let mut text = opening;
    loop {
        let emission = turn(client, text).await?;
        // Insufficiency is detected by TOKEN ID, and the wire carries
        // decoded text, so this side can only ever say no. The engine is
        // where the ids are and the wire is what would have to carry the
        // verdict.
        match questness.step(&emission, false)? {
            lib::Step::Continue(next) => text = next,
            lib::Step::Answered(answer) => return Ok(answer),
            lib::Step::Insufficient => return Err(QuestPluginError::Insufficient),
            lib::Step::Repair(envelope) => {
                return Err(QuestPluginError::envelope(&envelope));
            }
        }
    }
}

/// Connect to the daemon with this transport's installed material.
async fn connect() -> QuestPluginResult<bridge::TlsClientHandle> {
    let files = bridge::installed_material()?;
    Ok(bridge::TlsClientHandle::connect(bridge::TlsClientOptions {
        address: bridge::LOOPBACK_ADDRESS,
        files,
    })
    .await?)
}

/// Open a session, and learn which era answered.
///
/// The options are sent empty because the daemon cannot honour them: one
/// container owns one model for the service's life. Sending a decode
/// budget would read as a request that was quietly ignored.
async fn open(client: &mut bridge::TlsClientHandle) -> QuestPluginResult<()> {
    client
        .send(bridge::ClientToServer::Open(bridge::OpenRequest {
            options: bridge::all::ChatOptions::default(),
        }))
        .await?;
    loop {
        let Some(batch) = client.recv().await else {
            snafu::whatever!("the daemon closed before answering the open");
        };
        for message in batch {
            match message {
                bridge::ServerToClient::Open(_) => return Ok(()),
                bridge::ServerToClient::OpenRefused(refused) => {
                    snafu::whatever!("the daemon refused the session: {}", refused.message);
                }
                bridge::ServerToClient::Close => {
                    snafu::whatever!("the daemon closed before answering the open");
                }
                other => refuse_unexpected(other)?,
            }
        }
    }
}

/// Send one turn and reassemble its emission from the chunks.
///
/// The chunks are reassembled rather than streamed because a rendering
/// cannot exist until the whole emission is parsed - the answer is a
/// value plus a template over it, and neither is knowable a token at a
/// time. Streaming the raw emission alongside is a product decision
/// rather than something this owes.
async fn turn(
    client: &mut bridge::TlsClientHandle,
    text: String,
) -> QuestPluginResult<String> {
    client
        .send(bridge::ClientToServer::Turn(bridge::TurnRequest { text }))
        .await?;
    let mut emission = String::new();
    loop {
        let Some(batch) = client.recv().await else {
            snafu::whatever!("the daemon closed mid-turn");
        };
        for message in batch {
            match message {
                bridge::ServerToClient::TurnChunk(chunk) => emission.push_str(&chunk.text),
                bridge::ServerToClient::Turn(_) => return Ok(emission),
                bridge::ServerToClient::TurnFailed(failed) => {
                    snafu::whatever!("the turn did not run: {}", failed.message);
                }
                bridge::ServerToClient::Close => {
                    snafu::whatever!("the daemon closed mid-turn");
                }
                other => refuse_unexpected(other)?,
            }
        }
    }
}

/// Whatever the daemon said that no request asked for.
///
/// A shutdown notice ends the conversation, because reconnecting will not
/// help; a fault leaves the session open, so it is reported and the wait
/// continues. Anything else is a message this consumer does not service
/// and is passed over rather than treated as a fault of its own.
fn refuse_unexpected(message: bridge::ServerToClient) -> QuestPluginResult<()> {
    match message {
        bridge::ServerToClient::Notice(bridge::ServerNotice::Shutdown(notice)) => {
            snafu::whatever!("the daemon is going down: {}", notice.reason);
        }
        bridge::ServerToClient::Notice(bridge::ServerNotice::Fault(notice)) => {
            snafu::whatever!("the daemon reported a fault: {}", notice.message);
        }
        _ => Ok(()),
    }
}
