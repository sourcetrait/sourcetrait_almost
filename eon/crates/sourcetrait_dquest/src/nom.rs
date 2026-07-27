//! The noms: one per user's space, one per session inside it.
//!
//! Minted and compared today; READ as text only by the locks, since the
//! consumer that spells one into a path is session logging, which is
//! blocked on where `SessionLog` should live (debt).
#![allow(dead_code)]

/// The alphabet, in the order that makes a nom sort as its number does.
const BASE62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// A user's space, derived from their name rather than registered.
///
/// Stable across restarts, because the same user must find the same
/// space - which is why this hashes rather than counts.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ThinkspaceNom(String);

/// One session inside a space, minted fresh and never reused.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SessionNom(String);

impl ThinkspaceNom {
    /// The space a username names.
    ///
    /// An unrecognised user simply gets a space, since mutual TLS is the
    /// whole of the admission check - so this never fails, and there is
    /// no registration step for it to consult.
    pub(crate) fn of(username: &str) -> Self {
        let digest = <sha2::Sha256 as sha2::Digest>::digest(username.as_bytes());
        let mut head = [0u8; 8];
        head.copy_from_slice(&digest[..8]);
        Self(base62(u64::from_be_bytes(head)))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl SessionNom {
    pub(crate) fn fresh() -> Self {
        Self(base62(rand::random::<u64>()))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ThinkspaceNom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::fmt::Display for SessionNom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A number as base62 digits, most significant first.
///
/// The alphabet carries no separator, no dot and no path character, so a
/// nom is a plain path segment BY CONSTRUCTION rather than by being
/// checked afterwards - which is what lets one be joined to a log root
/// without a sanitiser standing between them.
fn base62(mut value: u64) -> String {
    if value == 0 {
        return String::from("0");
    }
    let mut digits = Vec::new();
    while value > 0 {
        digits.push(BASE62[(value % 62) as usize]);
        value /= 62;
    }
    digits.reverse();
    String::from_utf8(digits).expect("the alphabet is ascii")
}