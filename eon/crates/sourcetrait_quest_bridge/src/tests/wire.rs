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
    OpenRefusedResponse,
    ResetRequest,
    ResetResponse,
    ServerFaultNotice,
    ServerNotice,
    ServerShutdownNotice,
    ServerToClient,
    TurnChunk,
    TurnFailedResponse,
};
use crate::infer::{
    InferInput,
    InferNu,
    InferNuExecute,
    InferNuonInput,
    InferPass,
    InferRequest,
    InferResponse,
    InferText,
    InferValue,
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
        ClientToServer::Turn(InferRequest {
            text: Some(InferText(String::from("what is in this directory"))),
            ..InferRequest::default()
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
        ServerToClient::OpenRefused(OpenRefusedResponse {
            message: String::from("the checkpoint did not load"),
        }),
        ServerToClient::TurnChunk(TurnChunk {
            text: String::from("Hello"),
        }),
        ServerToClient::TurnFailed(TurnFailedResponse {
            message: String::from("the engine faulted mid-generation"),
        }),
        ServerToClient::Turn(InferResponse {
            report: Some(TurnReport {
                finish: FinishReason::StopToken,
                prompt_token_count: 2106,
                generated_token_count: 128,
                prefill_seconds: 0.5103515625,
                decode_seconds: 2.328125,
            }),
            ..InferResponse::default()
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
        let message = ServerToClient::Turn(InferResponse {
            report: Some(TurnReport {
                finish,
                prompt_token_count: 1,
                generated_token_count: 1,
                prefill_seconds: 0.0,
                decode_seconds: 0.0,
            }),
            ..InferResponse::default()
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
            ServerToClient::Turn(InferResponse {
                report: Some(report.clone()),
                ..InferResponse::default()
            }),
            &mut buffer,
        )
        .expect("encodes");
    let Some(ServerToClient::Turn(back)) = codec.decode(&mut buffer).expect("decodes") else {
        panic!("a turn response reads back as one");
    };
    let carried = back.report.as_ref().expect("a finished turn reports");
    assert_eq!(
        carried.prefill_seconds.to_bits(),
        report.prefill_seconds.to_bits()
    );
    assert_eq!(
        carried.decode_seconds.to_bits(),
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

/// The whole point of the rewrite: a typed value crosses AS A VALUE
/// rather than as a rendering of one, so no layer but ThinkHarness ever
/// sees a block.
#[test]
fn a_nu_value_crosses_the_wire_inside_a_request() {
    let value = InferValue::from_nuon("{name: foo, rows: [[a b]; [1 2]], took: 30sec}")
        .expect("fixture parses");
    let message = ClientToServer::Turn(InferRequest {
        inputs: vec![(InferPass::In, InferInput::Nuon(InferNuonInput(value)))],
        ..InferRequest::default()
    });

    let mut codec = BitcodeCodec::<ClientToServer>::new();
    let mut buffer = BytesMut::new();
    codec.encode(message.clone(), &mut buffer).expect("encodes");
    let back = codec.decode(&mut buffer).expect("decodes").expect("a whole frame");

    assert_eq!(back, message, "the typed literals survive the crossing");
}

/// An ask travels with what it consumes. An `InferNu` alone is a
/// function the caller has nothing to run on.
#[test]
fn an_ask_carries_its_bound_channels() {
    let value = InferValue::from_nuon("[a, b, c]").expect("fixture parses");
    let message = ServerToClient::Turn(InferResponse {
        nu: Some(InferNu::Execute(InferNuExecute(String::from(
            "def execute []: list<string> -> int { $in | length }",
        )))),
        inputs: vec![(InferPass::In, InferInput::Nuon(InferNuonInput(value)))],
        ..InferResponse::default()
    });

    let mut codec = BitcodeCodec::<ServerToClient>::new();
    let mut buffer = BytesMut::new();
    codec.encode(message, &mut buffer).expect("encodes");
    let Some(ServerToClient::Turn(back)) = codec.decode(&mut buffer).expect("decodes") else {
        panic!("a turn response reads back as one");
    };

    let ask = back.nu.expect("the ask travelled");
    assert!(!ask.is_think(), "only evaluate is a think turn");
    assert_eq!(ask.mode(), "execute");
    assert_eq!(back.inputs.len(), 1, "the ask carries what it consumes");
    assert_eq!(back.inputs[0].0, InferPass::In);
    assert_eq!(
        back.inputs[0].1.value().declared(),
        "list<string>",
        "and it carries the value's own derived type"
    );
}

/// A span is a byte offset into a file that existed in the sending
/// process, so it is meaningless to a receiver and must not travel.
/// NUON has nowhere to put one, which is why the newtype carries it.
#[test]
fn a_span_does_not_travel() {
    let span = nu_protocol::Span::new(4096, 4123);
    let carried = InferValue::new(nu_protocol::Value::string("hello", span));
    let nuon = carried.to_nuon().expect("renders");

    assert!(!nuon.contains("4096"), "a span offset reached the wire: {nuon}");
    assert!(!nuon.contains("4123"), "a span offset reached the wire: {nuon}");

    let back = InferValue::from_nuon(&nuon).expect("parses");
    assert_eq!(
        back.0.coerce_into_string().expect("a string"),
        "hello",
        "the value survives what the span does not"
    );
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
