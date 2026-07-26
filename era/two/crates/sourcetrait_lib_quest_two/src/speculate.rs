//! Lookup speculation: the gram index, the ladder, and the policy.
use crate::*;

/// Ladder levels, probed longest-first.
const LADDER: [usize; 3] = [4, 3, 2];

/// Widest ladder level; gram keys pad to this width.
const NGRAM_MAX: usize = 4;

/// Draft ceiling per verification forward, beside the pending token.
pub const MAX_DRAFT: usize = 16;

/// Consecutive zero-accept rounds at the floor ceiling before drafting
/// pauses.
const PAUSE_ZERO_STREAK: usize = 4;

/// Emitted-token rounds a drafting pause lasts.
const COOLDOWN_ROUNDS: usize = 64;

/// The two most recent continuation starts for one gram.
#[derive(Clone, Copy)]
struct GramSpots {
    latest: u32,
    previous: Option<u32>,
}

/// The prompt-lookup index over the committed context stream.
pub struct LookupIndex {
    map: HashMap<(u8, [u32; NGRAM_MAX]), GramSpots>,
    tokens: Vec<u32>,
}

/// The n-gram ending at `end`, padded to the key width.
fn gram_key(tokens: &[u32], end: usize, n: usize) -> (u8, [u32; NGRAM_MAX]) {
    let mut gram = [0u32; NGRAM_MAX];
    gram[..n].copy_from_slice(&tokens[end - n..end]);
    (n as u8, gram)
}

impl Default for LookupIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl LookupIndex {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            tokens: Vec::new(),
        }
    }

    /// Append accepted tokens, indexing each completed ladder gram.
    pub fn extend(&mut self, new_tokens: &[u32]) {
        for &token in new_tokens {
            self.tokens.push(token);
            let len = self.tokens.len();
            let continuation = len as u32;
            for &n in &LADDER {
                if len < n {
                    continue;
                }
                self.map
                    .entry(gram_key(&self.tokens, len, n))
                    .and_modify(|spots| {
                        spots.previous = Some(spots.latest);
                        spots.latest = continuation;
                    })
                    .or_insert(GramSpots {
                        latest: continuation,
                        previous: None,
                    });
            }
        }
    }

    /// Draft up to `limit` tokens from the longest matching gram.
    pub fn draft(&self, limit: usize) -> Option<(usize, Vec<u32>)> {
        let len = self.tokens.len();
        if limit == 0 {
            return None;
        }
        for &n in &LADDER {
            if len < n {
                continue;
            }
            let Some(spots) = self.map.get(&gram_key(&self.tokens, len, n)) else {
                continue;
            };
            let source = if (spots.latest as usize) < len {
                spots.latest as usize
            } else {
                match spots.previous {
                    Some(previous) => previous as usize,
                    None => continue,
                }
            };
            let end = (source + limit).min(len);
            if end <= source {
                continue;
            }
            return Some((n, self.tokens[source..end].to_vec()));
        }
        None
    }
}

/// The adaptive draft policy: a tracking ceiling plus a cold-streak
/// pause.
pub struct DraftPolicy {
    /// Tokens the next draft may carry; MAX_DRAFT when hot, 1 at the
    /// floor.
    ceiling: usize,
    /// Zero-accept rounds since the last accepted draft.
    zero_streak: usize,
    /// Emitted-token rounds left in a drafting pause.
    cooldown: usize,
}

impl Default for DraftPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl DraftPolicy {
    pub fn new() -> Self {
        Self {
            ceiling: MAX_DRAFT,
            zero_streak: 0,
            cooldown: 0,
        }
    }

    /// The per-round allowance, read before the index probe.
    pub fn probe_limit(&mut self) -> usize {
        if self.cooldown > 0 {
            self.cooldown -= 1;
            if self.cooldown == 0 {
                self.ceiling = 1;
                self.zero_streak = 0;
            }
            return 0;
        }
        self.ceiling
    }

    /// The n-seeded length for a matched draft.
    pub fn draft_limit(&self, matched_n: usize) -> usize {
        if matched_n >= 3 || self.ceiling == MAX_DRAFT {
            self.ceiling
        } else {
            self.ceiling.min(2)
        }
    }

    /// Record a verification round's acceptance count.
    pub fn record(&mut self, accepted: usize) {
        if accepted > 0 {
            self.zero_streak = 0;
            self.ceiling = (accepted * 2).clamp(2, MAX_DRAFT);
        } else {
            self.zero_streak += 1;
            self.ceiling = (self.ceiling / 2).max(1);
            if self.ceiling == 1 && self.zero_streak >= PAUSE_ZERO_STREAK {
                self.cooldown = COOLDOWN_ROUNDS;
                self.zero_streak = 0;
            }
        }
    }
}
