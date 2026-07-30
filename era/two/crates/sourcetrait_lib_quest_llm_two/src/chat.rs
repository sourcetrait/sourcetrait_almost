//! Chat rendering, byte-exact to the checkpoint's chat_template.jinja.
use crate::*;

/// The system text the template injects when none is supplied.
const INJECTED_SYSTEM: &str = "You are a helpful function-calling AI assistant. \
You do not currently have access to any functions. <functions></functions>";

/// The Thinkspace's own turn role, which never crosses the wire. A
/// reasoning FORM is the request; the Thinkspace answers on a thought,
/// and the model finishes inside the same turn.
pub const THOUGHT_ROLE: &str = "thought";

/// The byte-exact default rendering: one user turn, generation prompt.
pub fn chat_wrap(user_prompt: &str) -> String {
    chat_render(&[ChatMessage::user(user_prompt)], true).text
}

/// Continuation rendering for a context that ended mid assistant turn.
pub fn chat_continue(user_prompt: &str) -> String {
    let mut wrapped = String::new();
    wrapped.push_str("<|im_end|>\n");
    wrapped.push_str("<|im_start|>user\n");
    wrapped.push_str(user_prompt);
    wrapped.push_str("<|im_end|>\n");
    wrapped.push_str("<|im_start|>assistant\n");
    wrapped
}

/// Stop-token ids resolved by string, so tokenizer truth wins.
pub fn resolve_stop_ids(tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    consts::STOP_TOKENS
        .iter()
        .filter_map(|token| tokenizer.token_to_id(token))
        .collect()
}

/// A message's role; `Tool` is the template's alias for `Environment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    /// The Thinkspace's answer to a reasoning form; supplied, never
    /// generated, so it is context rather than a target.
    Thought,
    Environment,
    Tool,
}

impl ChatRole {
    /// The role token as the template spells it (`tool` renders as
    /// `environment`).
    fn rendered(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Thought => THOUGHT_ROLE,
            Self::Environment | Self::Tool => "environment",
        }
    }

    /// Parse the on-disk spelling; an unknown role raises.
    pub fn parse(role: &str) -> LibQuestResult<Self> {
        Ok(match role {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            THOUGHT_ROLE => Self::Thought,
            "environment" => Self::Environment,
            "tool" => Self::Tool,
            other => snafu::whatever!("unknown chat role {other:?}"),
        })
    }
}

/// One conversation turn; every field is optional as the template tests
/// presence.
#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    pub role: Option<ChatRole>,
    pub content: Option<String>,
    pub functions: Option<String>,
    pub function_calls: Option<String>,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: &str) -> Self {
        Self {
            role: Some(role),
            content: Some(content.to_string()),
            ..Self::default()
        }
    }

    pub fn system(content: &str) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: &str) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: &str) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    pub fn environment(content: &str) -> Self {
        Self::new(ChatRole::Environment, content)
    }

    fn is_role(&self, role: ChatRole) -> bool {
        self.role == Some(role)
    }
}

/// One assistant turn's byte range: the body through its terminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssistantSpan {
    pub start: usize,
    pub end: usize,
}

impl AssistantSpan {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// A render plus its assistant spans, in message order.
#[derive(Debug, Clone)]
pub struct ChatRender {
    pub text: String,
    pub assistant_spans: Vec<AssistantSpan>,
}

/// Render a message list byte-exactly, reporting each assistant span.
pub fn chat_render(messages: &[ChatMessage], add_generation_prompt: bool) -> ChatRender {
    let mut text = String::new();
    let mut assistant_spans = Vec::new();

    if !messages.iter().any(|message| message.is_role(ChatRole::System)) {
        text.push_str("<|im_start|>system\n");
        text.push_str(INJECTED_SYSTEM);
        text.push_str("<|im_end|>\n");
    }

    let last_index = messages.len().saturating_sub(1);
    for (index, message) in messages.iter().enumerate() {
        let last = index == last_index;
        let Some(role) = message.role else {
            continue;
        };
        match role {
            ChatRole::System => {
                text.push_str("<|im_start|>system\n");
                if let Some(content) = &message.content {
                    text.push_str(content);
                }
                if let Some(functions) = &message.functions {
                    text.push_str(" <functions>");
                    text.push_str(functions);
                    text.push_str("</functions>");
                }
                text.push_str("<|im_end|>\n");
            }
            ChatRole::User => {
                text.push_str("<|im_start|>user\n");
                if let Some(content) = &message.content {
                    text.push_str(content);
                }
                text.push_str("<|im_end|>\n");
            }
            // Every assistant turn is GENERATED, so every one carries a
            // span. An offload sequence has two - the model reaching for
            // the tool, and the answer it finishes with - and leaving the
            // first out of the loss would train everything about the
            // answer except the decision that produced it.
            ChatRole::Assistant => {
                text.push_str("<|im_start|>assistant\n");
                let start = text.len();
                if let Some(content) = &message.content {
                    text.push_str(content);
                }
                if let Some(calls) = &message.function_calls {
                    text.push_str("<function_calls>");
                    text.push_str(calls);
                    text.push_str("</function_calls>");
                }
                if last {
                    text.push_str(consts::EOS_TOKEN);
                    assistant_spans.push(AssistantSpan { start, end: text.len() });
                } else {
                    text.push_str("<|im_end|>");
                    assistant_spans.push(AssistantSpan { start, end: text.len() });
                    text.push('\n');
                }
            }
            // A thought is SUPPLIED, so it is context and never a span:
            // training it would train the model to predict its own tool's
            // output, which is the inference the offload exists to remove.
            ChatRole::Thought | ChatRole::Environment | ChatRole::Tool => {
                text.push_str("<|im_start|>");
                text.push_str(role.rendered());
                text.push('\n');
                if let Some(content) = &message.content {
                    text.push_str(content);
                }
                text.push_str("<|im_end|>\n");
            }
        }
        if last && add_generation_prompt {
            text.push_str("<|im_start|>assistant\n");
        }
    }

    ChatRender { text, assistant_spans }
}

/// One assistant turn as a half-open token range over the render's ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

/// A render encoded once, its spans carried to token positions.
#[derive(Debug, Clone)]
pub struct EncodedRender {
    pub ids: Vec<u32>,
    pub assistant_spans: Vec<TokenSpan>,
}

/// Encode a render and map its byte spans onto token positions.
pub fn encode_render(
    tokenizer: &tokenizers::Tokenizer,
    render: &ChatRender,
) -> LibQuestResult<EncodedRender> {
    let encoding = match tokenizer.encode(render.text.as_str(), false) {
        Ok(encoding) => encoding,
        Err(e) => snafu::whatever!("render encode failed: {e}"),
    };
    let ids = encoding.get_ids().to_vec();
    let offsets = encoding.get_offsets();

    let mut boundary: HashMap<usize, usize> = HashMap::with_capacity(offsets.len() + 1);
    for (index, (start, _)) in offsets.iter().enumerate() {
        boundary.entry(*start).or_insert(index);
    }
    boundary.insert(render.text.len(), ids.len());

    let mut assistant_spans = Vec::with_capacity(render.assistant_spans.len());
    for span in &render.assistant_spans {
        let (Some(start), Some(end)) =
            (boundary.get(&span.start).copied(), boundary.get(&span.end).copied())
        else {
            snafu::whatever!(
                "assistant span {}..{} does not align to token boundaries",
                span.start,
                span.end
            );
        };
        snafu::ensure_whatever!(
            start <= end,
            "assistant span {}..{} maps to a reversed token range {start}..{end}",
            span.start,
            span.end
        );
        assistant_spans.push(TokenSpan { start, end });
    }
    Ok(EncodedRender { ids, assistant_spans })
}
