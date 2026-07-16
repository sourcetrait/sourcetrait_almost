use crate::*;

/// Ladder levels, probed longest-first: a longer matched context is a
/// higher-precision draft source; the 2-gram floor raises match rate on
/// the target workloads (schema-shaped, repair-loop, doc text) and its
/// precision cost rides the policy's n-seeded lengths.
const LADDER: [usize; 3] = [4, 3, 2];

/// Widest ladder level; gram keys pad to this width (the level rides in
/// the key, so padding cannot collide across levels).
const NGRAM_MAX: usize = 4;

/// Draft ceiling per verification forward (plus the pending token).
/// 16 beat 8 by +17% on echo-shaped decode (the tracking ceiling only
/// reaches it on deep accepts, so partial-accept workloads never pay
/// the wider span).
pub(crate) const MAX_DRAFT: usize = 16;

/// Consecutive zero-accept rounds at the floor ceiling before drafting
/// pauses.
const PAUSE_ZERO_STREAK: usize = 4;

/// Emitted-token rounds a drafting pause lasts.
const COOLDOWN_ROUNDS: usize = 64;

/// Two most recent continuation starts for one gram; the tail's own
/// insertion always occupies `latest`, so `previous` is what keeps an
/// earlier occurrence draftable.
#[derive(Clone, Copy)]
struct GramSpots {
    latest: u32,
    previous: Option<u32>,
}

/// E5a prompt-lookup index: the context stream (prompt + accepted
/// generation) with its ladder grams mapped to continuation positions,
/// maintained incrementally off the critical path. When the context
/// tail matches an earlier gram, the tokens that followed it become
/// draft candidates; greedy verification keeps logits exact.
pub(crate) struct LookupIndex {
    map: HashMap<(u8, [u32; NGRAM_MAX]), GramSpots>,
    tokens: Vec<u32>,
}

/// The n-gram ending at `end`, padded to the key width.
fn gram_key(tokens: &[u32], end: usize, n: usize) -> (u8, [u32; NGRAM_MAX]) {
    let mut gram = [0u32; NGRAM_MAX];
    gram[..n].copy_from_slice(&tokens[end - n..end]);
    (n as u8, gram)
}

impl LookupIndex {
    pub(crate) fn new() -> Self {
        Self {
            map: HashMap::new(),
            tokens: Vec::new(),
        }
    }

    /// Append accepted tokens, indexing each completed ladder gram to
    /// the position where its continuation starts.
    pub(crate) fn extend(&mut self, new_tokens: &[u32]) {
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

    /// Draft up to `limit` tokens continuing the current tail from the
    /// most recent EARLIER occurrence of the longest-matching ladder
    /// gram. Returns the matched level with the tokens; None when no
    /// level has an earlier occurrence or the limit is empty.
    pub(crate) fn draft(&self, limit: usize) -> Option<(usize, Vec<u32>)> {
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
            // The tail's own insertion is `latest` when the tail gram
            // is unique-so-far; an earlier occurrence lives in
            // `previous`.
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

/// E5a adaptive draft policy: the per-round ceiling grows on accepted
/// drafts and shrinks on rejected ones, and a sustained cold streak
/// pauses drafting entirely - novel output converges to plain-greedy
/// cost instead of paying the verification-forward tax on every false
/// hit.
pub(crate) struct DraftPolicy {
    /// Tokens the next draft may carry; MAX_DRAFT when hot, 1 at the
    /// floor.
    ceiling: usize,
    /// Zero-accept rounds since the last accepted draft.
    zero_streak: usize,
    /// Emitted-token rounds left in a drafting pause.
    cooldown: usize,
}

impl DraftPolicy {
    pub(crate) fn new() -> Self {
        Self {
            ceiling: MAX_DRAFT,
            zero_streak: 0,
            cooldown: 0,
        }
    }

    /// Per-round allowance, read before the index probe; 0 while
    /// paused. Each call is one prospective-draft round (ticks a pause
    /// down).
    pub(crate) fn probe_limit(&mut self) -> usize {
        if self.cooldown > 0 {
            self.cooldown -= 1;
            if self.cooldown == 0 {
                // Resume probing cheaply; accepts regrow the ceiling.
                self.ceiling = 1;
                self.zero_streak = 0;
            }
            return 0;
        }
        self.ceiling
    }

    /// n-seeded length for a matched draft: 3+ carries the full
    /// ceiling; the low-precision 2-gram floor drafts short until the
    /// run is hot.
    pub(crate) fn draft_limit(&self, matched_n: usize) -> usize {
        if matched_n >= 3 || self.ceiling == MAX_DRAFT {
            self.ceiling
        } else {
            self.ceiling.min(2)
        }
    }

    /// Record a verification round's acceptance count: track the
    /// ceiling toward twice the observed accept depth (echo-shaped
    /// runs climb to MAX_DRAFT, partial-accept workloads hover at
    /// their true depth instead of paying full-span rejections),
    /// halve on a full rejection, pause after a sustained cold streak
    /// at the floor.
    pub(crate) fn record(&mut self, accepted: usize) {
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
