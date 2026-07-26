//! Chat rendering, byte-exact to the checkpoint's chat_template.jinja
//! (byte-identical to era-one olmo3's - the one sanctioned code carry).
//!
//! Two surfaces. `chat_wrap` / `chat_continue` are the single-shape
//! inference helpers the engine and the battery rigs drive.
//! `chat_render` is the message-list form the trainer needs: it
//! returns the rendered text AND the byte span of every assistant
//! turn, so a training example's loss boundary is exact by
//! construction from ONE render.
//!
//! ## DEV
//! The span form exists because the alternative - re-rendering the
//! conversation prefix and taking its token count - is measurably
//! wrong against THIS template. The assistant turn renders
//! differently by position (eos when last, `<|im_end|>` plus a
//! newline when interior), so a prefix ending at an assistant turn
//! measures one token short, once per interior assistant turn. One
//! render plus spans has no positional invariant to maintain and
//! costs one pass instead of n+1.
//!
//! A span COVERS its turn's terminator (`<|im_end|>` or the eos
//! token) and STOPS there: the newline after an interior
//! `<|im_end|>` opens the next turn rather than closing this one, and
//! our harness re-renders the whole conversation each turn, so the
//! model is never asked to produce it. That exclusion is a decision,
//! not the inherited off-by-one.
//! ##
use crate::*;

/// The system text the template injects when the conversation
/// carries no system message of its own.
const INJECTED_SYSTEM: &str = "You are a helpful function-calling AI assistant. \
You do not currently have access to any functions. <functions></functions>";

/// Byte-exact default rendering (no system message, no tools, generation
/// prompt appended) of the checkpoint's chat_template.jinja.
pub fn chat_wrap(user_prompt: &str) -> String {
    chat_render(&[ChatMessage::user(user_prompt)], true).text
}

/// Continuation rendering for a standing context that ended mid
/// assistant turn (every post-decode save does - the stop token is
/// sampled but never consumed): close that turn, open a user turn with
/// the prompt, and open the next assistant turn.
pub fn chat_continue(user_prompt: &str) -> String {
    let mut wrapped = String::new();
    wrapped.push_str("<|im_end|>\n");
    wrapped.push_str("<|im_start|>user\n");
    wrapped.push_str(user_prompt);
    wrapped.push_str("<|im_end|>\n");
    wrapped.push_str("<|im_start|>assistant\n");
    wrapped
}

/// Stop-token ids resolved by string, so tokenizer truth wins over any
/// config-side id drift.
pub fn resolve_stop_ids(tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    consts::STOP_TOKENS
        .iter()
        .filter_map(|token| tokenizer.token_to_id(token))
        .collect()
}

/// A message's role. `Tool` is the template's own alias for
/// `Environment` and renders identically; it exists so foreign data
/// carrying that role round-trips without the caller rewriting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
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
            Self::Environment | Self::Tool => "environment",
        }
    }

    /// Parse the on-disk spelling; anything else is a hard error at
    /// the caller (training data must not carry a role we silently
    /// drop).
    pub fn parse(role: &str) -> LibQuestResult<Self> {
        Ok(match role {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "environment" => Self::Environment,
            "tool" => Self::Tool,
            other => snafu::whatever!("unknown chat role {other:?}"),
        })
    }
}

/// One conversation turn. `content` is optional because the template
/// tests presence rather than truthiness: absent renders nothing,
/// while an EMPTY string renders an empty body (both engines agree).
/// `functions` rides a system message and `function_calls` an
/// assistant one; both are the template's STRING paths, which is what
/// the checkpoint's own data populates.
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

/// One assistant turn's byte range in the rendered text: the body
/// (content plus any function-call block) through its terminator,
/// exclusive of the newline that opens the next turn.
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

/// A render plus the assistant spans within it, in message order. A
/// trailing generation prompt produces no span - there is no content
/// behind it yet.
#[derive(Debug, Clone)]
pub struct ChatRender {
    pub text: String,
    pub assistant_spans: Vec<AssistantSpan>,
}

/// Render a message list byte-exactly per the checkpoint template,
/// reporting each assistant turn's span. `add_generation_prompt`
/// appends the assistant opener after the final message, whatever its
/// role - the template's own behavior.
///
/// The structured `tools` argument is deliberately unrepresentable
/// here: the checkpoint's own data and our runtime both populate the
/// `functions` / `function_calls` STRING fields, so that branch is
/// dead for every path this crate takes.
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
            ChatRole::Environment | ChatRole::Tool => {
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

/// One assistant turn as a token range over the whole render's ids
/// (half-open, in token positions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

/// A render encoded once, with every assistant span carried through
/// to token positions.
#[derive(Debug, Clone)]
pub struct EncodedRender {
    pub ids: Vec<u32>,
    pub assistant_spans: Vec<TokenSpan>,
}

/// Encode a render and map its byte spans onto token positions.
///
/// A span boundary that does not coincide with a token boundary is a
/// HARD ERROR rather than a rounded one: the mask it feeds decides
/// which positions carry training signal, and a silently-shifted
/// boundary trains the wrong thing without any symptom. The whole
/// render encodes as ONE string, matching how the same text is
/// encoded at inference.
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

    // Byte offset -> token index, for every token START, plus the
    // end-of-text sentinel. A boundary must land in this map.
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
