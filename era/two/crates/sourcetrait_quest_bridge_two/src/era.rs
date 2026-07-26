//! The era-two entry: identity, and the session factory.
use crate::*;

/// Era two: the Olmo-Hybrid DPO checkpoint on the lib_quest engine.
pub struct BridgeTwo;

impl bridge::all::Era for BridgeTwo {
    /// The sole-checkpoint identity, from constants and not a load.
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
