//! Who is connected: one Thinkspace, and the sessions live inside it.
use crate::*;

/// The live sessions, and the space they belong to.
///
/// Cloning shares the registry rather than copying it, so every accepted
/// connection admits against the same one.
#[derive(Clone)]
pub(crate) struct SessionManager {
    thinkspace: nom::ThinkspaceNom,
    log_root: Option<PathBuf>,
    live: std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<nom::SessionNom>>>,
    questness: QuestnessHandle,
}

/// A session's place in the registry, held for as long as it runs.
///
/// Deregistration is a DROP rather than a call, so a session that ends
/// early - a failed handshake, a dropped peer, a panicking task - cannot
/// leave itself behind in the registry. There is no path that forgets.
pub(crate) struct SessionTicket {
    nom: nom::SessionNom,
    live: std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<nom::SessionNom>>>,
}

impl SessionManager {
    /// The registry for one user's space, logging under `log_root`.
    pub(crate) fn for_user(
        username: &str,
        log_root: Option<PathBuf>,
        questness: impl bridge::all::Questness + 'static,
    ) -> Self {
        let owner: Box<dyn bridge::all::Questness> = Box::new(questness);
        Self {
            thinkspace: nom::ThinkspaceNom::of(username),
            log_root,
            live: std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new())),
            questness: std::sync::Arc::new(tokio::sync::Mutex::new(owner)),
        }
    }

    /// The space every session here belongs to.
    pub(crate) fn thinkspace(&self) -> &nom::ThinkspaceNom {
        &self.thinkspace
    }

    /// This space's turn owner. Every session shares the one Questness,
    /// which is what binds a continuation to the turn it continues.
    pub(crate) fn questness(&self) -> QuestnessHandle {
        std::sync::Arc::clone(&self.questness)
    }

    /// This session's log, addressed by space and then by session.
    ///
    /// None where there is no root or the directory will not open. A
    /// session that cannot be logged is still served, because refusing a
    /// connection over a log is the wrong trade - the log records the
    /// work rather than being part of it.
    pub(crate) fn log(&self, ticket: &SessionTicket) -> Option<log::SessionLog> {
        let root = self.log_root.as_ref()?;
        log::SessionLog::open(
            root,
            &[self.thinkspace().as_str(), ticket.nom().as_str()],
        )
        .ok()
    }

    /// Take a place in the registry, minting the nom that names it.
    pub(crate) fn admit(&self) -> SessionTicket {
        let nom = nom::SessionNom::fresh();
        self.locked().insert(nom.clone());
        SessionTicket {
            nom,
            live: std::sync::Arc::clone(&self.live),
        }
    }

    /// How many sessions are live.
    ///
    /// Read by the locks alone: the daemon has no verb that reports its
    /// own state yet, and inventing one to consume this would be a
    /// design decision rather than a wiring detail.
    #[allow(dead_code)]
    pub(crate) fn live(&self) -> usize {
        self.locked().len()
    }

    /// Every live session, for a listing rather than for control.
    #[allow(dead_code)]
    pub(crate) fn sessions(&self) -> Vec<nom::SessionNom> {
        self.locked().iter().cloned().collect()
    }

    /// A poisoned registry is recovered rather than propagated: one
    /// session's panic must not make the daemon refuse every later one.
    fn locked(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::BTreeSet<nom::SessionNom>> {
        match self.live.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl SessionTicket {
    pub(crate) fn nom(&self) -> &nom::SessionNom {
        &self.nom
    }
}

impl Drop for SessionTicket {
    fn drop(&mut self) {
        let mut live = match self.live.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        live.remove(&self.nom);
    }
}