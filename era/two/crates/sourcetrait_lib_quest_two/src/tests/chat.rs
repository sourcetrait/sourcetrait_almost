//! Chat-render locks. The fixture table below is the byte gate: each
//! `expected` was produced by the reference stack's own
//! `apply_chat_template` over the checkpoint's chat_template.jinja, so
//! a passing row proves our renderer equals transformers on that
//! branch. The structured `tools` argument is deliberately absent -
//! our data populates the `functions` / `function_calls` string
//! fields and the object branch is unreachable here.
use crate::*;

/// The tool-declaration JSON the reference fixtures carry verbatim in
/// a system message's `functions` STRING field.
const FUNCTIONS_JSON: &str = r#"[{"type": "function", "function": {"name": "weather.forecast", "description": "Fetch a forecast.", "parameters": {"type": "object", "properties": {"q": {"description": "Location.", "type": "string"}, "days": {"description": "Day count.", "type": "integer"}}}}}]"#;

/// The header the template injects when no system message is present.
const INJECTED_HEADER: &str = "<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n";

fn system_with_functions(content: &str) -> ChatMessage {
    ChatMessage {
        role: Some(ChatRole::System),
        content: Some(content.to_string()),
        functions: Some(FUNCTIONS_JSON.to_string()),
        function_calls: None,
    }
}

fn assistant_calls(content: Option<&str>, calls: &str) -> ChatMessage {
    ChatMessage {
        role: Some(ChatRole::Assistant),
        content: content.map(str::to_string),
        functions: None,
        function_calls: Some(calls.to_string()),
    }
}

fn tool(content: &str) -> ChatMessage {
    ChatMessage::new(ChatRole::Tool, content)
}

#[test]
fn chat_wrap_matches_template_default_rendering() {
    let expected = "<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
    assert_eq!(chat_wrap("Hi"), expected);
}

#[test]
fn chat_continue_closes_and_reopens_turns() {
    let expected = "<|im_end|>\n<|im_start|>user\nGo on<|im_end|>\n<|im_start|>assistant\n";
    assert_eq!(chat_continue("Go on"), expected);
}

#[test]
fn render_is_byte_exact_against_the_reference_fixtures() {
    let json = FUNCTIONS_JSON;
    let cases: Vec<(&str, Vec<ChatMessage>, bool, String)> = vec![
        (
            "single_turn",
            vec![ChatMessage::user("U"), ChatMessage::assistant("A")],
            false,
            format!("{INJECTED_HEADER}<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\nA<|endoftext|>"),
        ),
        (
            "multi_turn",
            vec![
                ChatMessage::user("U1"),
                ChatMessage::assistant("A1"),
                ChatMessage::user("U2"),
                ChatMessage::assistant("A2"),
            ],
            false,
            format!(
                "{INJECTED_HEADER}<|im_start|>user\nU1<|im_end|>\n\
                 <|im_start|>assistant\nA1<|im_end|>\n\
                 <|im_start|>user\nU2<|im_end|>\n\
                 <|im_start|>assistant\nA2<|endoftext|>"
            ),
        ),
        (
            "explicit_system",
            vec![
                ChatMessage::system("S"),
                ChatMessage::user("U"),
                ChatMessage::assistant("A"),
            ],
            false,
            String::from(
                "<|im_start|>system\nS<|im_end|>\n\
                 <|im_start|>user\nU<|im_end|>\n\
                 <|im_start|>assistant\nA<|endoftext|>",
            ),
        ),
        (
            "system_functions_string",
            vec![
                system_with_functions("S"),
                ChatMessage::user("U"),
                ChatMessage::assistant("A"),
            ],
            false,
            format!(
                "<|im_start|>system\nS <functions>{json}</functions><|im_end|>\n\
                 <|im_start|>user\nU<|im_end|>\n\
                 <|im_start|>assistant\nA<|endoftext|>"
            ),
        ),
        (
            "assistant_function_calls_string",
            vec![
                system_with_functions("S"),
                ChatMessage::user("U"),
                assistant_calls(None, r#"weather.forecast(q="Paris", days=5)"#),
            ],
            false,
            format!(
                "<|im_start|>system\nS <functions>{json}</functions><|im_end|>\n\
                 <|im_start|>user\nU<|im_end|>\n\
                 <|im_start|>assistant\n<function_calls>weather.forecast(q=\"Paris\", days=5)</function_calls><|endoftext|>"
            ),
        ),
        (
            "parallel_function_calls",
            vec![
                system_with_functions("S"),
                ChatMessage::user("U"),
                assistant_calls(
                    None,
                    "weather.forecast(q=\"Paris\", days=5)\nweather.forecast(q=\"Madrid\", days=5)",
                ),
            ],
            false,
            format!(
                "<|im_start|>system\nS <functions>{json}</functions><|im_end|>\n\
                 <|im_start|>user\nU<|im_end|>\n\
                 <|im_start|>assistant\n<function_calls>weather.forecast(q=\"Paris\", days=5)\nweather.forecast(q=\"Madrid\", days=5)</function_calls><|endoftext|>"
            ),
        ),
        (
            "full_tool_loop",
            vec![
                system_with_functions("S"),
                ChatMessage::user("U"),
                assistant_calls(None, r#"weather.forecast(q="Paris", days=5)"#),
                ChatMessage::environment(r#"{"results": {"maxtemp_c": 27}}"#),
                ChatMessage::assistant("Paris peaks at 27C."),
            ],
            false,
            format!(
                "<|im_start|>system\nS <functions>{json}</functions><|im_end|>\n\
                 <|im_start|>user\nU<|im_end|>\n\
                 <|im_start|>assistant\n<function_calls>weather.forecast(q=\"Paris\", days=5)</function_calls><|im_end|>\n\
                 <|im_start|>environment\n{{\"results\": {{\"maxtemp_c\": 27}}}}<|im_end|>\n\
                 <|im_start|>assistant\nParis peaks at 27C.<|endoftext|>"
            ),
        ),
        (
            "tool_role_alias",
            vec![ChatMessage::user("U"), tool("T"), ChatMessage::assistant("A")],
            false,
            format!(
                "{INJECTED_HEADER}<|im_start|>user\nU<|im_end|>\n\
                 <|im_start|>environment\nT<|im_end|>\n\
                 <|im_start|>assistant\nA<|endoftext|>"
            ),
        ),
        (
            "content_and_calls",
            vec![
                system_with_functions("S"),
                ChatMessage::user("U"),
                assistant_calls(Some("Thinking."), r#"weather.forecast(q="Paris", days=5)"#),
            ],
            false,
            format!(
                "<|im_start|>system\nS <functions>{json}</functions><|im_end|>\n\
                 <|im_start|>user\nU<|im_end|>\n\
                 <|im_start|>assistant\nThinking.<function_calls>weather.forecast(q=\"Paris\", days=5)</function_calls><|endoftext|>"
            ),
        ),
        (
            "empty_assistant_content",
            vec![ChatMessage::user("U"), ChatMessage::assistant("")],
            false,
            format!("{INJECTED_HEADER}<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"),
        ),
        (
            "gen_prompt_after_user",
            vec![ChatMessage::user("U")],
            true,
            format!("{INJECTED_HEADER}<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"),
        ),
        (
            "gen_prompt_after_environment",
            vec![
                ChatMessage::user("U"),
                assistant_calls(None, "f()"),
                ChatMessage::environment("R"),
            ],
            true,
            format!(
                "{INJECTED_HEADER}<|im_start|>user\nU<|im_end|>\n\
                 <|im_start|>assistant\n<function_calls>f()</function_calls><|im_end|>\n\
                 <|im_start|>environment\nR<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
        ),
    ];

    for (name, messages, generation_prompt, expected) in cases {
        let render = chat_render(&messages, generation_prompt);
        assert_eq!(render.text, expected, "fixture {name} diverged");
    }
}

#[test]
fn assistant_spans_cover_body_through_terminator() {
    let render = chat_render(
        &[
            ChatMessage::user("U1"),
            ChatMessage::assistant("A1"),
            ChatMessage::user("U2"),
            ChatMessage::assistant("A2"),
        ],
        false,
    );
    let slices: Vec<&str> = render
        .assistant_spans
        .iter()
        .map(|span| &render.text[span.start..span.end])
        .collect();
    // The interior turn stops at its own terminator; the newline that
    // follows opens the next turn and is deliberately outside.
    assert_eq!(slices, vec!["A1<|im_end|>", "A2<|endoftext|>"]);
}

#[test]
fn a_generation_prompt_produces_no_span() {
    let render = chat_render(&[ChatMessage::user("U")], true);
    assert!(render.assistant_spans.is_empty());
    assert!(render.text.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn spans_cover_calls_and_empty_bodies() {
    let render = chat_render(
        &[
            ChatMessage::user("U"),
            assistant_calls(Some("Thinking."), "f()"),
        ],
        false,
    );
    assert_eq!(
        &render.text[render.assistant_spans[0].start..render.assistant_spans[0].end],
        "Thinking.<function_calls>f()</function_calls><|endoftext|>"
    );

    let render = chat_render(&[ChatMessage::user("U"), ChatMessage::assistant("")], false);
    assert_eq!(
        &render.text[render.assistant_spans[0].start..render.assistant_spans[0].end],
        "<|endoftext|>"
    );
}

#[test]
fn roles_parse_and_tool_aliases_environment() {
    assert_eq!(ChatRole::parse("assistant").expect("known"), ChatRole::Assistant);
    assert_eq!(ChatRole::parse("tool").expect("known"), ChatRole::Tool);
    assert!(ChatRole::parse("narrator").is_err());
    let aliased = chat_render(&[tool("T")], false);
    assert!(aliased.text.contains("<|im_start|>environment\nT<|im_end|>\n"));
}

#[test]
fn encoded_spans_land_on_token_boundaries_and_decode_back() {
    let dir = model_dir(consts::DPO_MODEL_NAME).expect("dpo dir");
    let tokenizer = load_tokenizer(&dir).expect("dpo tokenizer loads");
    let render = chat_render(
        &[
            ChatMessage::user("What is the capital of France?"),
            ChatMessage::assistant("Paris."),
            ChatMessage::user("And of Spain?"),
            ChatMessage::assistant("Madrid."),
        ],
        false,
    );
    let encoded = encode_render(&tokenizer, &render).expect("spans align to token boundaries");
    assert_eq!(encoded.assistant_spans.len(), 2);

    // The whole render round-trips, and each token span decodes to
    // exactly the text its byte span covers.
    for (token_span, byte_span) in encoded.assistant_spans.iter().zip(&render.assistant_spans) {
        let ids = &encoded.ids[token_span.start..token_span.end];
        let decoded = tokenizer.decode(ids, false).expect("decodes");
        assert_eq!(decoded, render.text[byte_span.start..byte_span.end]);
    }

    // Both terminators carry training signal - the ai2 masking defect
    // dropped exactly this class.
    let first = &encoded.ids[encoded.assistant_spans[0].start..encoded.assistant_spans[0].end];
    let last = &encoded.ids[encoded.assistant_spans[1].start..encoded.assistant_spans[1].end];
    assert_eq!(*first.last().expect("interior span"), consts::TOKEN_IM_END);
    assert_eq!(*last.last().expect("final span"), consts::TOKEN_ENDOFTEXT);
}

/// The offload sequence a supervised example carries: the model asks,
/// the Thinkspace answers, the model finishes. Both assistant turns are
/// the model's own generation, so both train; the thought between them
/// is supplied and is context.
#[test]
fn both_assistant_turns_train_and_the_thought_between_them_does_not() {
    let messages = vec![
        ChatMessage {
            role: Some(ChatRole::User),
            content: Some(String::from("What is 5 + 5?")),
            ..Default::default()
        },
        ChatMessage {
            role: Some(ChatRole::Assistant),
            content: Some(String::from("ASK")),
            ..Default::default()
        },
        ChatMessage {
            role: Some(ChatRole::Thought),
            content: Some(String::from("TOLD")),
            ..Default::default()
        },
        ChatMessage {
            role: Some(ChatRole::Assistant),
            content: Some(String::from("10")),
            ..Default::default()
        },
    ];
    let rendered = chat_render(&messages, false);
    assert!(rendered.text.contains("<|im_start|>assistant\nASK"), "{}", rendered.text);
    assert!(rendered.text.contains("<|im_start|>thought\nTOLD"), "{}", rendered.text);

    let trained: Vec<&str> = rendered
        .assistant_spans
        .iter()
        .map(|span| &rendered.text[span.start..span.end])
        .collect();
    assert_eq!(trained.len(), 2, "the offload and the answer");
    assert!(trained[0].starts_with("ASK"), "got {trained:?}");
    assert!(trained[1].starts_with("10"), "got {trained:?}");
    assert!(
        !trained.iter().any(|span| span.contains("TOLD")),
        "a supplied thought is context rather than a target"
    );
}

#[test]
fn the_thought_role_parses_from_the_spelling_the_turn_emits() {
    assert_eq!(
        ChatRole::parse(questness::turn::THOUGHT_ROLE).expect("thought"),
        ChatRole::Thought
    );
}
