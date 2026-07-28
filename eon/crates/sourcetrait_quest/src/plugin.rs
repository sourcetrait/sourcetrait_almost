//! The plugin itself: what it holds across calls, and what it serves.
use crate::*;

/// The plugin process, and the two things it keeps between calls.
///
/// Stock plugin garbage collection tears the process down after an idle
/// timeout, so what is held here is held for a burst of calls rather than
/// forever. The runtime is built once; the harness is a world rather than
/// a session, so it carries nothing between calls and costs nothing to
/// hold.
pub(crate) struct QuestPlugin {
    runtime: tokio::runtime::Runtime,
    harness: harness::QuestHarness,
}

impl QuestPlugin {
    pub(crate) fn new() -> QuestPluginResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        Ok(Self {
            runtime,
            harness: harness::QuestHarness::default(),
        })
    }

    pub(crate) fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    /// This side's own nu engine, for what the model asks IT to run.
    ///
    /// Questness never reaches it. An ask comes back on a response, this
    /// runs it under its own confinement, and the value goes back on the
    /// next request.
    pub(crate) fn harness(&self) -> harness::QuestHarness {
        self.harness.clone()
    }
}

impl nu_plugin::Plugin for QuestPlugin {
    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").into()
    }

    fn commands(&self) -> Vec<Box<dyn nu_plugin::PluginCommand<Plugin = Self>>> {
        vec![Box::new(prompt::Prompt)]
    }
}
