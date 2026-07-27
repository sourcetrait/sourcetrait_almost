//! The plugin itself: what it holds across calls, and what it serves.
use crate::*;

/// The client harness the plugin services client-bound modes through.
type Harness = lib::bubble::BubbleHarness;

/// The plugin process, and the two things it keeps between calls.
///
/// Stock plugin garbage collection tears the process down after an idle
/// timeout, so what is held here is held for a burst of calls rather than
/// forever. Both members are worth that: a tokio runtime and the
/// evaluator's base engine are each built once and reused.
pub(crate) struct QuestPlugin {
    runtime: tokio::runtime::Runtime,
    questness: Mutex<lib::Questness<Harness>>,
}

impl QuestPlugin {
    pub(crate) fn new() -> QuestPluginResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        // Tool markers stay aliased because the routing they stand in for
        // is untrained: the channel probe reads zero of six on the marker
        // and the no-marker arm catches everything. Training retires this
        // rather than a code change.
        let questness = lib::Questness::new(
            Harness::default(),
            lib::channel::Aliasing::ToolMarkers,
        )?;
        Ok(Self {
            runtime,
            questness: Mutex::new(questness),
        })
    }

    pub(crate) fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    /// The turn owner, for the length of one call.
    ///
    /// A poisoned lock is recovered rather than propagated. One call
    /// panicking must not make every later call fail, and there is no
    /// invariant to protect: what is behind the lock is an engine and a
    /// world, both of which a fresh turn rebuilds its own state over.
    pub(crate) fn questness(&self) -> MutexGuard<'_, lib::Questness<Harness>> {
        match self.questness.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
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
