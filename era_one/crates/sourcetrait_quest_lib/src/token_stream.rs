use crate::*;

/// Incremental utf8-safe detokenizer: emits the decoded suffix only once it
/// ends in ascii, so multi-byte sequences never split across writes.
pub(crate) struct TokenStream<'t> {
    tokenizer: &'t tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<'t> TokenStream<'t> {
    pub(crate) fn new(tokenizer: &'t tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub(crate) fn next_token(&mut self, token: u32) -> QuestResult<Option<String>> {
        let previous_text = self.decode(&self.tokens[self.prev_index..self.current_index])?;
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > previous_text.len() && text.chars().last().is_some_and(|c| c.is_ascii()) {
            let emitted = text.split_at(previous_text.len()).1.to_string();
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(emitted))
        } else {
            Ok(None)
        }
    }

    /// Flush whatever the ascii-boundary heuristic is still holding back.
    pub(crate) fn decode_rest(&self) -> QuestResult<Option<String>> {
        let previous_text = self.decode(&self.tokens[self.prev_index..self.current_index])?;
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > previous_text.len() {
            Ok(Some(text.split_at(previous_text.len()).1.to_string()))
        } else {
            Ok(None)
        }
    }

    fn decode(&self, ids: &[u32]) -> QuestResult<String> {
        match self.tokenizer.decode(ids, true) {
            Ok(text) => Ok(text),
            Err(error) => snafu::whatever!("detokenization failed: {error}"),
        }
    }
}
