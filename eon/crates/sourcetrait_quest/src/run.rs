//! Binary entry: serve the plugin, or say why it could not start.
use crate::*;

/// Serve until nushell stops asking.
///
/// A startup failure goes to STDERR and nowhere else. Stdout is the
/// msgpack protocol channel, so a diagnostic printed there is not a
/// message a reader ever sees - it is a corrupt frame.
pub fn run() {
    match QuestPlugin::new() {
        Ok(plugin) => nu_plugin::serve_plugin(&plugin, nu_plugin::MsgPackSerializer),
        Err(error) => {
            eprintln!("nu_plugin_quest: unable to start: {error}");
            std::process::exit(1);
        }
    }
}
