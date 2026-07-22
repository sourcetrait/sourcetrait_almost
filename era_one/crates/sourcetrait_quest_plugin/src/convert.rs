use crate::*;

/// NUON text -> Value via the nuon crate's own parser (no engine
/// involved; from_nuon builds its own).
pub fn parse_nuon(text: &str) -> PluginResult<Value> {
    match nuon::from_nuon(text, None) {
        Ok(value) => Ok(value),
        Err(error) => NuonSnafu { message: error.to_string() }.fail(),
    }
}

/// Value -> NUON text (default style; non-serializable types reject
/// rather than stringify).
pub fn render_nuon(value: &Value) -> PluginResult<String> {
    let engine_state = nu_protocol::engine::EngineState::new();
    match nuon::to_nuon(&engine_state, value, nuon::ToNuonConfig::default()) {
        Ok(text) => Ok(text),
        Err(error) => NuonRenderSnafu { message: error.to_string() }.fail(),
    }
}
