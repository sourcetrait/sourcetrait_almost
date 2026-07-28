//! The conversation with the daemon, driven to one answer.
use crate::*;

/// How long a polite close waits for the far end to answer it.
const GOODBYE: std::time::Duration = std::time::Duration::from_secs(2);

/// Ask one turn and drive it, asks included, to an answer.
///
/// THE TURN LOOP IS THE DAEMON'S. What runs here is the other half of a
/// sequence: when the model asks THIS side to run something, QuestHarness
/// runs it on its own engine and the value goes back as a continuation,
/// where the Thinkspace's conversation resumes.
pub(crate) fn ask(
    plugin: &QuestPlugin,
    request: bridge::InferRequest,
) -> QuestPluginResult<bridge::InferResponse> {
    plugin
        .runtime()
        .block_on(converse(plugin.harness(), request))
}

/// Connect, open, and turn until the model answers.
async fn converse(
    harness: harness::QuestHarness,
    request: bridge::InferRequest,
) -> QuestPluginResult<bridge::InferResponse> {
    let mut client = connect().await?;
    let outcome = drive(&harness, &mut client, request).await;
    // Closed on every path, including a failed one: a daemon reading a
    // dropped socket cannot tell a crashed client from a finished one.
    client.close(GOODBYE).await;
    outcome
}

/// The turn loop, once a session is open.
async fn drive(
    harness: &harness::QuestHarness,
    client: &mut bridge::TlsClientHandle,
    request: bridge::InferRequest,
) -> QuestPluginResult<bridge::InferResponse> {
    open(client).await?;
    let mut sending = request;
    loop {
        let response = turn(client, sending).await?;
        if response.insufficient {
            return Err(QuestPluginError::Insufficient);
        }
        let Some(form) = &response.nu else {
            return Ok(response);
        };
        let value = harness.run(form, &response.inputs)?;
        sending = bridge::InferRequest {
            output: Some(bridge::InferOutput::Nuon(bridge::InferNuonOutput(
                bridge::InferValue(value),
            ))),
            ..bridge::InferRequest::default()
        };
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

/// Send one request and wait for the turn to end.
///
/// The chunks are the model's RAW emission arriving as it is written, and
/// they are not the answer: an answer is a value plus a rendering over
/// it, and neither is knowable a token at a time. They are passed over
/// here, and stay on the wire for a caller that wants to watch.
async fn turn(
    client: &mut bridge::TlsClientHandle,
    request: bridge::InferRequest,
) -> QuestPluginResult<bridge::InferResponse> {
    client.send(bridge::ClientToServer::Turn(request)).await?;
    loop {
        let Some(batch) = client.recv().await else {
            snafu::whatever!("the daemon closed mid-turn");
        };
        for message in batch {
            match message {
                bridge::ServerToClient::TurnChunk(_) => {}
                bridge::ServerToClient::Turn(response) => return Ok(response),
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
