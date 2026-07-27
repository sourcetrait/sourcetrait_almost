//! The runtime config record: what we consume, what the model sees.
use crate::*;

/// Keys addressed to Questness, which never reach the model.
pub const QUESTNESS_KEYS: [&str; 1] = ["liquid"];

/// The key whose presence makes the prompt a Liquid template.
pub const LIQUID_KEY: &str = "liquid";

/// A prepared turn: the prompt as sent, and the config as seen.
#[derive(Debug, Clone)]
pub struct PreparedTurn {
    pub prompt: String,
    /// The config record with the Questness keys removed.
    pub visible: nu::Value,
    /// Whether the prompt was rendered rather than sent verbatim.
    pub templated: bool,
}

/// Split a config record into the Questness half and the visible half.
pub fn split(config: &nu::Value) -> LibQuestResult<(nu::Record, nu::Record)> {
    let nu::Value::Record { val, .. } = config else {
        snafu::whatever!(
            "a config is a record; got {}",
            config.get_type()
        );
    };
    let mut questness = nu::Record::new();
    let mut visible = nu::Record::new();
    for (key, value) in val.iter() {
        if QUESTNESS_KEYS.contains(&key.as_str()) {
            questness.push(key.clone(), value.clone());
        } else {
            visible.push(key.clone(), value.clone());
        }
    }
    Ok((questness, visible))
}

/// Prepare one turn: render the prompt if asked, strip our own keys.
pub fn prepare(
    config: &nu::Value,
    prompt: &str,
    bindings: &[(String, nu::Value)],
) -> LibQuestResult<PreparedTurn> {
    let (questness, visible) = split(config)?;
    let templated = questness.get(LIQUID_KEY).is_some();
    let prompt = if templated {
        template::render(prompt, bindings)?
    } else {
        prompt.to_string()
    };
    Ok(PreparedTurn {
        prompt,
        visible: nu::Value::record(visible, nu::Span::unknown()),
        templated,
    })
}
