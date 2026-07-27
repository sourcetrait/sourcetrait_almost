//! Wire locks: both languages round-trip, and framing survives a
//! transport that hands over bytes whenever it feels like it.
use crate::all::{
    ChatOptions,
    EraInfo,
    FinishReason,
    TurnReport,
};
use crate::r::tokio::{
    BytesMut,
    Decoder,
    Encoder,
};
use crate::wire::{
    BitcodeCodec,
    CancelRequest,
    CancelResponse,
    ClientToServer,
    MAX_FRAME_BYTES,
    OpenRequest,
    OpenResponse,
    ResetRequest,
    ResetResponse,
    ServerFaultNotice,
    ServerNotice,
    ServerShutdownNotice,
    ServerToClient,
    TurnChunk,
    TurnRequest,
    TurnResponse,
};

fn client_messages() -> Vec<ClientToServer> {
    vec![
        ClientToServer::Open(OpenRequest {
            options: ChatOptions::default(),
        }),
        ClientToServer::Open(OpenRequest {
            options: ChatOptions {
                dir: Some(String::from("profiles")),
                config: Some(String::from("default")),
                settings: None,
                sample_len: Some(4096),
            },
        }),
        ClientToServer::Turn(TurnRequest {
            text: String::from("what is in this directory"),
        }),
        ClientToServer::Cancel(CancelRequest),
        ClientToServer::Reset(ResetRequest),
        ClientToServer::Close,
    ]
}

fn server_messages() -> Vec<ServerToClient> {
    vec![
        ServerToClient::Open(OpenResponse {
            info: EraInfo {
                era: String::from("two"),
                model: String::from("allenai/Olmo-Hybrid-Instruct-DPO-7B"),
            },
        }),
        ServerToClient::TurnChunk(TurnChunk {
            text: String::from("Hello"),
        }),
        ServerToClient::Turn(TurnResponse {
            report: TurnReport {
                finish: FinishReason::StopToken,
                prompt_token_count: 2106,
                generated_token_count: 128,
                prefill_seconds: 0.5103515625,
                decode_seconds: 2.328125,
            },
        }),
        ServerToClient::Cancel(CancelResponse { stopped: true }),
        ServerToClient::Cancel(CancelResponse { stopped: false }),
        ServerToClient::Reset(ResetResponse),
        ServerToClient::Notice(ServerNotice::Fault(ServerFaultNotice {
            message: String::from("a turn is already generating"),
        })),
        ServerToClient::Notice(ServerNotice::Shutdown(ServerShutdownNotice {
            reason: String::from("the operator stopped the daemon"),
        })),
        ServerToClient::Close,
    ]
}

#[test]
fn every_client_message_round_trips() {
    let mut codec = BitcodeCodec::<ClientToServer>::new();
    for message in client_messages() {
        let mut buffer = BytesMut::new();
        codec.encode(message.clone(), &mut buffer).expect("encodes");
        let back = codec.decode(&mut buffer).expect("decodes").expect("a whole frame");
        assert_eq!(back, message);
        assert!(buffer.is_empty(), "a frame consumes exactly its own bytes");
    }
}

#[test]
fn every_server_message_round_trips() {
    let mut codec = BitcodeCodec::<ServerToClient>::new();
    for message in server_messages() {
        let mut buffer = BytesMut::new();
        codec.encode(message.clone(), &mut buffer).expect("encodes");
        let back = codec.decode(&mut buffer).expect("decodes").expect("a whole frame");
        assert_eq!(back, message);
        assert!(buffer.is_empty(), "a frame consumes exactly its own bytes");
    }
}

#[test]
fn every_finish_reason_survives_the_wire() {
    let mut codec = BitcodeCodec::<ServerToClient>::new();
    for finish in [
        FinishReason::StopToken,
        FinishReason::SampleLen,
        FinishReason::Cancelled,
    ] {
        let message = ServerToClient::Turn(TurnResponse {
            report: TurnReport {
                finish,
                prompt_token_count: 1,
                generated_token_count: 1,
                prefill_seconds: 0.0,
                decode_seconds: 0.0,
            },
        });
        let mut buffer = BytesMut::new();
        codec.encode(message.clone(), &mut buffer).expect("encodes");
        assert_eq!(codec.decode(&mut buffer).expect("decodes"), Some(message));
    }
}

/// Seconds are read against a benchmark row, so a shifted last place
/// would be a silently wrong number rather than a visible failure.
#[test]
fn timings_round_trip_bit_for_bit() {
    let report = TurnReport {
        finish: FinishReason::SampleLen,
        prompt_token_count: 32152,
        generated_token_count: 4096,
        prefill_seconds: 6.949999809265137,
        decode_seconds: 95.34374809265137,
    };
    let mut codec = BitcodeCodec::<ServerToClient>::new();
    let mut buffer = BytesMut::new();
    codec
        .encode(
            ServerToClient::Turn(TurnResponse {
                report: report.clone(),
            }),
            &mut buffer,
        )
        .expect("encodes");
    let Some(ServerToClient::Turn(back)) = codec.decode(&mut buffer).expect("decodes") else {
        panic!("a turn response reads back as one");
    };
    assert_eq!(
        back.report.prefill_seconds.to_bits(),
        report.prefill_seconds.to_bits()
    );
    assert_eq!(
        back.report.decode_seconds.to_bits(),
        report.decode_seconds.to_bits()
    );
}

/// The reason this rides a length codec rather than anything of ours. A
/// socket hands over whatever arrived, so a frame split anywhere must
/// yield nothing rather than half a message, and complete when the rest
/// lands.
#[test]
fn a_frame_split_anywhere_yields_nothing_until_it_is_whole() {
    let message = ServerToClient::TurnChunk(TurnChunk {
        text: String::from("a chunk long enough to straddle a packet boundary"),
    });
    let mut whole = BytesMut::new();
    BitcodeCodec::<ServerToClient>::new()
        .encode(message.clone(), &mut whole)
        .expect("encodes");

    for split in 1..whole.len() {
        let mut codec = BitcodeCodec::<ServerToClient>::new();
        let mut buffer = BytesMut::from(&whole[..split]);
        assert_eq!(
            codec.decode(&mut buffer).expect("a partial frame is not an error"),
            None,
            "a frame split at {split} must not decode"
        );
        buffer.extend_from_slice(&whole[split..]);
        assert_eq!(
            codec.decode(&mut buffer).expect("decodes"),
            Some(message.clone()),
            "the rest arriving completes the frame split at {split}"
        );
    }
}

/// Several messages arriving in one read must not bleed into each other.
#[test]
fn several_frames_in_one_buffer_decode_in_order() {
    let mut codec = BitcodeCodec::<ServerToClient>::new();
    let messages = server_messages();
    let mut buffer = BytesMut::new();
    for message in &messages {
        codec.encode(message.clone(), &mut buffer).expect("encodes");
    }
    for message in &messages {
        assert_eq!(
            codec.decode(&mut buffer).expect("decodes"),
            Some(message.clone())
        );
    }
    assert_eq!(codec.decode(&mut buffer).expect("decodes"), None);
    assert!(buffer.is_empty());
}

/// A chunk is model output, so it carries whatever the model wrote. The
/// frame is delimited by its length, so none of it can end a frame early.
#[test]
fn a_chunk_carrying_hostile_text_is_inert() {
    let text = String::from("first\nsecond\r\n</>\n\0\u{1b}[31m \"quoted\" \\ tail");
    let message = ServerToClient::TurnChunk(TurnChunk { text: text.clone() });
    let mut codec = BitcodeCodec::<ServerToClient>::new();
    let mut buffer = BytesMut::new();
    codec.encode(message, &mut buffer).expect("encodes");

    let Some(ServerToClient::TurnChunk(back)) = codec.decode(&mut buffer).expect("decodes") else {
        panic!("a chunk reads back as a chunk");
    };
    assert_eq!(back.text, text, "arbitrary text survives byte for byte");
}

/// A well-framed payload that is not a message is a peer fault, so it
/// reports rather than panicking the connection task.
#[test]
fn a_frame_that_is_not_a_message_is_refused() {
    let mut buffer = BytesMut::new();
    BitcodeCodec::<ClientToServer>::new()
        .encode(ClientToServer::Close, &mut buffer)
        .expect("encodes");
    let last = buffer.len() - 1;
    buffer[last] ^= 0xff;

    let error = BitcodeCodec::<ClientToServer>::new()
        .decode(&mut buffer)
        .expect_err("a corrupted payload is refused");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn an_oversize_frame_is_refused_rather_than_buffered() {
    let message = ServerToClient::TurnChunk(TurnChunk {
        text: "x".repeat(MAX_FRAME_BYTES + 1),
    });
    let mut buffer = BytesMut::new();
    assert!(
        BitcodeCodec::<ServerToClient>::new()
            .encode(message, &mut buffer)
            .is_err()
    );
}
