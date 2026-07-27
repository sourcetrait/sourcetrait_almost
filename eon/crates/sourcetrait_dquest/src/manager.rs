//! Who is connected: one Thinkspace, and the sessions live inside it.
//!
//! Admitting and releasing are wired; OBSERVING is not, because its only
//! consumer is session logging, which is blocked on where `SessionLog`
//! should live (debt). The locks drive the observation surface, so the
//! allow covers what the daemon does not read yet rather than what
//! nothing exercises.
#![allow(dead_code)]
use crate::*;

/// The live sessions, and the space they belong to.
///
/// Cloning shares the registry rather than copying it, so every accepted
/// connection admits against the same one.
#[derive(Clone)]
pub(crate) struct SessionManager {
    thinkspace: nom::ThinkspaceNom,
    live: std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<nom::SessionNom>>>,
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
    /// The registry for one user's space.
    pub(crate) fn for_user(username: &str) -> Self {
        Self {
            thinkspace: nom::ThinkspaceNom::of(username),
            live: std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new())),
        }
    }

    /// The space every session here belongs to.
    pub(crate) fn thinkspace(&self) -> &nom::ThinkspaceNom {
        &self.thinkspace
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

    pub(crate) fn live(&self) -> usize {
        self.locked().len()
    }

    /// Every live session, for a listing rather than for control.
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