//! The runtime config record: what we consume, what the model sees.
use crate::*;

/// Keys addressed to ThinkHarness, which never reach the model.
pub const THINK_HARNESS_KEYS: [&str; 2] = ["liquid", "conversation"];

/// The key whose presence makes the prompt a Liquid template.
pub const LIQUID_KEY: &str = "liquid";

/// The key that keeps the conversation standing past this turn.
pub const CONVERSATION_KEY: &str = "conversation";

/// The value that spells keeping it.
pub const CONVERSATION_KEEP: &str = "keep";

/// What happens to the conversation when this turn ends.
///
/// Teardown is the DEFAULT: a bare call is a fresh conversation, always,
/// and nothing has to be sent to get that. `{conversation: keep}` is the
/// opt-in that leaves the conversation standing for a following call
/// that also says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conversation {
    Teardown,
    Keep,
}

impl Conversation {
    /// The disposition a config record declares; absent means teardown.
    ///
    /// An unknown value is refused rather than defaulted: the caller
    /// composed the config, the caller is code, and a typo silently
    /// tearing a kept conversation down would be the worst reading.
    pub fn of(config: &nu::Value) -> HarnessQuestResult<Self> {
        let nu::Value::Record { val, .. } = config else {
            snafu::whatever!("a config is a record; got {}", config.get_type());
        };
        let Some(value) = val.get(CONVERSATION_KEY) else {
            return Ok(Self::Teardown);
        };
        match value {
            nu::Value::String { val, .. } if val == CONVERSATION_KEEP => Ok(Self::Keep),
            other => snafu::whatever!(
                "{CONVERSATION_KEY} is kept with `{CONVERSATION_KEEP}`; omit it for \
                 teardown; got {}",
                other.get_type()
            ),
        }
    }
}

/// A prepared turn: the prompt as sent, and the config as seen.
#[derive(Debug, Clone)]
pub struct PreparedTurn {
    pub prompt: String,
    /// The config record with the ThinkHarness keys removed.
    pub visible: nu::Value,
    /// Whether the prompt was rendered rather than sent verbatim.
    pub templated: bool,
    /// What happens to the conversation when this turn ends.
    pub conversation: Conversation,
}

/// Split a config record into the ThinkHarness half and the visible half.
pub fn split(config: &nu::Value) -> HarnessQuestResult<(nu::Record, nu::Record)> {
    let nu::Value::Record { val, .. } = config else {
        snafu::whatever!(
            "a config is a record; got {}",
            config.get_type()
        );
    };
    let mut think_harness = nu::Record::new();
    let mut visible = nu::Record::new();
    for (key, value) in val.iter() {
        if THINK_HARNESS_KEYS.contains(&key.as_str()) {
            think_harness.push(key.clone(), value.clone());
        } else {
            visible.push(key.clone(), value.clone());
        }
    }
    Ok((think_harness, visible))
}

/// Prepare one turn: render the prompt if asked, strip our own keys.
pub fn prepare(
    config: &nu::Value,
    prompt: &str,
    bindings: &[(String, nu::Value)],
) -> HarnessQuestResult<PreparedTurn> {
    let conversation = Conversation::of(config)?;
    let (think_harness, visible) = split(config)?;
    let templated = think_harness.get(LIQUID_KEY).is_some();
    let prompt = if templated {
        template::render(prompt, bindings)?
    } else {
        prompt.to_string()
    };
    Ok(PreparedTurn {
        prompt,
        visible: nu::Value::record(visible, nu::Span::unknown()),
        templated,
        conversation,
    })
}
