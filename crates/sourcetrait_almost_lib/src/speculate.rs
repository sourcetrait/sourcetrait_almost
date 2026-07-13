use crate::*;

/// Tokens per lookup n-gram; 3 balances match rate against false drafts
/// on the target workloads (schema-shaped, repair-loop, doc text).
const NGRAM: usize = 3;

/// Draft ceiling per verification forward (plus the pending token).
pub(crate) const MAX_DRAFT: usize = 8;

/// Two most recent continuation starts for one n-gram; the tail's own
/// insertion always occupies `latest`, so `previous` is what keeps an
/// earlier occurrence draftable.
#[derive(Clone, Copy)]
struct GramSpots {
    latest: u32,
    previous: Option<u32>,
}

/// E5a prompt-lookup index: the context stream (prompt + accepted
/// generation) with its n-grams mapped to continuation positions,
/// maintained incrementally off the critical path. When the context
/// tail matches an earlier n-gram, the tokens that followed it become
/// draft candidates; greedy verification keeps logits exact.
pub(crate) struct LookupIndex {
    map: HashMap<[u32; NGRAM], GramSpots>,
    tokens: Vec<u32>,
}

impl LookupIndex {
    pub(crate) fn new() -> Self {
        Self {
            map: HashMap::new(),
            tokens: Vec::new(),
        }
    }

    /// Append accepted tokens, indexing each completed n-gram to the
    /// position where its continuation starts.
    pub(crate) fn extend(&mut self, new_tokens: &[u32]) {
        for &token in new_tokens {
            self.tokens.push(token);
            let len = self.tokens.len();
            if len >= NGRAM {
                let mut gram = [0u32; NGRAM];
                gram.copy_from_slice(&self.tokens[len - NGRAM..]);
                let continuation = len as u32;
                self.map
                    .entry(gram)
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

    /// Draft up to `budget.min(MAX_DRAFT)` tokens continuing the current
    /// tail, from the most recent EARLIER occurrence of the tail's
    /// n-gram. None when the tail is unseen or the budget is empty.
    pub(crate) fn draft(&self, budget: usize) -> Option<Vec<u32>> {
        let len = self.tokens.len();
        if len < NGRAM || budget == 0 {
            return None;
        }
        let mut gram = [0u32; NGRAM];
        gram.copy_from_slice(&self.tokens[len - NGRAM..]);
        let spots = self.map.get(&gram)?;
        // The tail's own insertion is `latest` when the tail gram is
        // unique-so-far; an earlier occurrence lives in `previous`.
        let source = if (spots.latest as usize) < len {
            spots.latest as usize
        } else {
            spots.previous? as usize
        };
        let end = (source + budget.min(MAX_DRAFT)).min(len);
        if end <= source {
            return None;
        }
        Some(self.tokens[source..end].to_vec())
    }
}
