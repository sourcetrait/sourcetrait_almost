//! The era-two entry: identity plus session factories over the
//! sourcetrait_lib_quest engine.
use crate::*;

/// Era two: the Olmo-Hybrid DPO checkpoint on the lib_quest engine.
pub struct BridgeTwo;

impl bridge::all::Era for BridgeTwo {
    /// The era's sole-checkpoint identity (the loaded model's actual
    /// coordinate rides the session's Ready event, where a config
    /// profile could differ).
    fn info(&self) -> bridge::all::EraInfo {
        bridge::all::EraInfo {
            era: String::from("two"),
            model: format!(
                "{}/{}",
                lib::consts::MODEL_AUTHOR,
                lib::consts::DPO_MODEL_NAME
            ),
        }
    }

    fn open_chat(
        &self,
        options: &bridge::all::ChatOptions,
    ) -> bridge::BridgeResult<bridge::all::ChatSession> {
        chat::open_chat(options)
    }
}
