//! The generation surface: pull-based tokens over the carried path.
use crate::*;

/// The prefill chunk fed per forward on the generate path.
pub(crate) const PREFILL_CHUNK: usize = 512;

/// Sampling posture and budget for one generation; no temperature is
/// greedy.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub sample_len: usize,
    /// Render through the chat template; false feeds text verbatim.
    pub chat: bool,
    /// Consume stop tokens and decode the full budget; never in chat.
    pub ignore_stops: bool,
    /// Lookup speculation; greedy-only, and never under a graph.
    pub speculate: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            temperature: Some(0.6),
            top_p: Some(0.95),
            seed: 299_792_458,
            sample_len: 32_768,
            chat: true,
            ignore_stops: false,
            speculate: false,
        }
    }
}

impl GenerateOptions {
    /// Greedy argmax with a tight budget: the battery and parity rig.
    pub fn greedy(sample_len: usize) -> Self {
        Self {
            temperature: None,
            top_p: None,
            sample_len,
            ..Self::default()
        }
    }
}

/// One produced token and the text it detokenized to, possibly empty.
pub struct GenerationStep {
    pub token_id: u32,
    pub chunk: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token was sampled, and not consumed into the caches.
    StopToken,
    /// The sample_len budget was spent.
    SampleLen,
}

/// The end-of-generation accounting `finish` returns.
pub struct GenerationReport {
    /// None when the caller stopped early by dropping the iterator.
    pub finish_reason: Option<FinishReason>,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    /// Speculation tallies; zero when it was off or never fired.
    pub drafted_token_count: usize,
    pub accepted_draft_token_count: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    /// Detokenizer tail not yet emitted through the steps.
    pub rest: String,
    /// The consumed-id trail, which is exactly the cache contents.
    pub context_ids: Vec<u32>,
}

/// The pull-based generation iterator over a borrowed model.
pub struct Generation<'a> {
    model: &'a mut OlmoHybrid,
    tokenizer: &'a tokenizers::Tokenizer,
    processor: r::sampling::LogitsProcessor,
    stop_ids: Vec<u32>,
    ignore_stops: bool,
    generated_ids: Vec<u32>,
    /// The consumed-id trail, growing in lockstep with the caches.
    context_ids: Vec<u32>,
    emitted_bytes: usize,
    pending_id: Option<u32>,
    /// Accepted tokens awaiting emission; their yields owe no forward.
    queued: VecDeque<u32>,
    speculate: bool,
    index: speculate::LookupIndex,
    policy: speculate::DraftPolicy,
    drafted_count: usize,
    accepted_draft_count: usize,
    sample_len: usize,
    finish_reason: Option<FinishReason>,
    failed: bool,
    prompt_token_count: usize,
    prefill_seconds: f64,
    decode_started: std::time::Instant,
}

impl OlmoHybrid {
    /// Clear, render, prefill chunked, and hand back the iterator.
    pub fn generate<'a>(
        &'a mut self,
        tokenizer: &'a tokenizers::Tokenizer,
        prompt: &str,
        options: &GenerateOptions,
    ) -> LibQuestResult<Generation<'a>> {
        self.clear_cache()?;
        let rendered = if options.chat {
            chat_wrap(prompt)
        } else {
            prompt.to_string()
        };
        self.start_generation(tokenizer, &rendered, Vec::new(), options)
    }

    /// Continue a RESTORED context, prefilling at its offset.
    pub fn generate_from<'a>(
        &'a mut self,
        tokenizer: &'a tokenizers::Tokenizer,
        restored: &RestoredContext,
        suffix: &str,
        options: &GenerateOptions,
    ) -> LibQuestResult<Generation<'a>> {
        snafu::ensure_whatever!(
            self.context_len() == restored.context_len && restored.context_len > 0,
            "the live cache ({}) does not hold the restored context ({})",
            self.context_len(),
            restored.context_len
        );
        let rendered = if options.chat {
            chat_continue(suffix)
        } else {
            suffix.to_string()
        };
        self.start_generation(tokenizer, &rendered, restored.context_ids.clone(), options)
    }

    /// The shared core: encode, prefill at the current offset, sample.
    fn start_generation<'a>(
        &'a mut self,
        tokenizer: &'a tokenizers::Tokenizer,
        rendered: &str,
        carried_trail: Vec<u32>,
        options: &GenerateOptions,
    ) -> LibQuestResult<Generation<'a>> {
        let encoding = match tokenizer.encode(rendered, false) {
            Ok(encoding) => encoding,
            Err(e) => snafu::whatever!("prompt encode failed: {e}"),
        };
        let prompt_ids = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            snafu::whatever!("empty prompt after rendering");
        }

        if options.speculate {
            snafu::ensure_whatever!(
                options.temperature.is_none(),
                "speculation is greedy-only (sampling verification is a later track)"
            );
        }
        let sampling = match (options.temperature, options.top_p) {
            (None, _) => r::sampling::Sampling::ArgMax,
            (Some(temperature), Some(p)) => r::sampling::Sampling::TopP { p, temperature },
            (Some(temperature), None) => r::sampling::Sampling::All { temperature },
        };
        let mut context_ids = carried_trail;
        context_ids.extend_from_slice(&prompt_ids);
        let mut generation = Generation {
            processor: r::sampling::LogitsProcessor::from_sampling(options.seed, sampling),
            stop_ids: resolve_stop_ids(tokenizer),
            ignore_stops: options.ignore_stops,
            generated_ids: Vec::new(),
            context_ids,
            emitted_bytes: 0,
            pending_id: None,
            queued: VecDeque::new(),
            speculate: options.speculate,
            index: speculate::LookupIndex::new(),
            policy: speculate::DraftPolicy::new(),
            drafted_count: 0,
            accepted_draft_count: 0,
            sample_len: options.sample_len,
            finish_reason: None,
            failed: false,
            prompt_token_count: prompt_ids.len(),
            prefill_seconds: 0.0,
            decode_started: std::time::Instant::now(),
            model: self,
            tokenizer,
        };
        if generation.speculate {
            generation.index.extend(&generation.context_ids);
        }

        let prefill_started = std::time::Instant::now();
        let context_before = generation.model.context_len();
        generation
            .model
            .reserve_for_generation(context_before + prompt_ids.len())?;
        let mut last_logits = None;
        let mut start = 0;
        while start < prompt_ids.len() {
            let len = PREFILL_CHUNK.min(prompt_ids.len() - start);
            let chunk = candle_core::Tensor::from_vec(
                prompt_ids[start..start + len].to_vec(),
                len,
                generation.model.device(),
            )?;
            if start + len == prompt_ids.len() {
                last_logits = Some(generation.model.forward_chunk_last(&chunk)?);
            } else {
                generation.model.forward_chunk_carry(&chunk)?;
            }
            start += len;
        }
        let last_logits = last_logits.expect("non-empty prompt");
        if let Some(evict) = generation.model.settings().eviction.clone() {
            generation.model.evict_post_prefill(&evict)?;
        }
        if generation.model.settings().graph && !generation.speculate {
            let expected_total =
                context_before + generation.prompt_token_count + generation.sample_len;
            generation.model.arm_graph_decode(expected_total)?;
        }
        generation.pending_id = Some(generation.sample(&last_logits)?);
        generation.prefill_seconds = prefill_started.elapsed().as_secs_f64();
        generation.decode_started = std::time::Instant::now();
        Ok(generation)
    }
}

impl Generation<'_> {
    fn sample(&mut self, logits_row: &candle_core::Tensor) -> LibQuestResult<u32> {
        let row = logits_row
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .to_device(&candle_core::Device::Cpu)?;
        Ok(self.processor.sample(&row)?)
    }

    /// The incremental detokenizer, as a diff of full decodes.
    fn emit_chunk(&mut self) -> LibQuestResult<String> {
        let full = match self.tokenizer.decode(&self.generated_ids, true) {
            Ok(text) => text,
            Err(e) => snafu::whatever!("detokenize failed: {e}"),
        };
        if full.len() <= self.emitted_bytes {
            return Ok(String::new());
        }
        let chunk = full[self.emitted_bytes..].to_string();
        if chunk.ends_with('\u{fffd}') {
            return Ok(String::new());
        }
        self.emitted_bytes = full.len();
        Ok(chunk)
    }

    fn step(&mut self) -> LibQuestResult<Option<GenerationStep>> {
        if self.finish_reason.is_some() || self.failed {
            return Ok(None);
        }
        let (token_id, from_queue) = match self.queued.pop_front() {
            Some(token_id) => (token_id, true),
            None => match self.pending_id.take() {
                Some(token_id) => (token_id, false),
                None => return Ok(None),
            },
        };
        if !self.ignore_stops && self.stop_ids.contains(&token_id) {
            self.queued.clear();
            self.finish_reason = Some(FinishReason::StopToken);
            return Ok(None);
        }
        self.generated_ids.push(token_id);
        let chunk = self.emit_chunk()?;
        if self.generated_ids.len() >= self.sample_len {
            self.queued.clear();
            self.finish_reason = Some(FinishReason::SampleLen);
            return Ok(Some(GenerationStep { token_id, chunk }));
        }
        if from_queue {
            return Ok(Some(GenerationStep { token_id, chunk }));
        }
        if let Some(evict) = self.model.settings().eviction.clone()
            && self.model.context_len() >= evict.decode_cap + evict::OVERFLOW_SLACK
        {
            self.model.evict_overflow(&evict)?;
        }
        if self.speculate {
            self.stage_or_speculate(token_id)?;
            return Ok(Some(GenerationStep { token_id, chunk }));
        }
        let logits = if self.model.graph_armed() {
            self.model.graph_disarm_when_full()?;
            if self.model.graph_armed() {
                self.model.graph_decode_step(token_id)?
            } else {
                let step_ids =
                    candle_core::Tensor::from_vec(vec![token_id], 1, self.model.device())?;
                self.model.forward_chunk(&step_ids)?
            }
        } else {
            let step_ids =
                candle_core::Tensor::from_vec(vec![token_id], 1, self.model.device())?;
            self.model.forward_chunk(&step_ids)?
        };
        self.context_ids.push(token_id);
        self.pending_id = Some(self.sample(&logits)?);
        Ok(Some(GenerationStep { token_id, chunk }))
    }

    /// One speculation round: index, probe, verify or step plainly.
    fn stage_or_speculate(&mut self, token_id: u32) -> LibQuestResult<()> {
        self.index.extend(&[token_id]);
        let budget = self.sample_len.saturating_sub(self.generated_ids.len());
        let draft = if budget > 0 {
            let probe = self.policy.probe_limit();
            self.index
                .draft(probe.min(budget))
                .map(|(matched_n, mut tokens)| {
                    tokens.truncate(self.policy.draft_limit(matched_n).min(budget));
                    tokens
                })
                .filter(|tokens| !tokens.is_empty())
        } else {
            None
        };
        let Some(draft) = draft else {
            let step_ids =
                candle_core::Tensor::from_vec(vec![token_id], 1, self.model.device())?;
            let logits = self.model.forward_chunk(&step_ids)?;
            self.context_ids.push(token_id);
            self.pending_id = Some(self.sample(&logits)?);
            return Ok(());
        };

        let span = 1 + draft.len();
        let mark = self.model.spec_mark()?;
        let mut block: Vec<u32> = Vec::with_capacity(span);
        block.push(token_id);
        block.extend_from_slice(&draft);
        let block_ids =
            candle_core::Tensor::from_vec(block, span, self.model.device())?;
        let logits = self
            .model
            .forward_chunk(&block_ids)?
            .to_dtype(candle_core::DType::F32)?
            .to_device(&candle_core::Device::Cpu)?;
        self.drafted_count += draft.len();

        let mut greedy_next: Vec<u32> = Vec::with_capacity(span);
        for row_index in 0..span {
            let row = logits.narrow(0, row_index, 1)?.squeeze(0)?;
            greedy_next.push(self.processor.sample(&row)?);
        }
        let mut accepted: Vec<u32> = vec![token_id];
        for (draft_index, &draft_token) in draft.iter().enumerate() {
            if greedy_next[draft_index] == draft_token {
                accepted.push(draft_token);
            } else {
                break;
            }
        }
        let consumed = accepted.len();
        self.accepted_draft_count += consumed - 1;
        self.policy.record(consumed - 1);
        let bonus = greedy_next[consumed - 1];

        if consumed < span {
            self.model.spec_accept(&mark, consumed)?;
        } else {
            self.model.spec_release();
        }
        self.context_ids.extend_from_slice(&accepted);
        self.index.extend(&accepted[1..]);

        let mut queued: Vec<u32> = accepted[1..].to_vec();
        let mut pending = bonus;
        if !self.ignore_stops
            && let Some(stop_at) = queued.iter().position(|t| self.stop_ids.contains(t))
        {
            pending = queued[stop_at];
            queued.truncate(stop_at);
        }
        self.queued = queued.into();
        self.pending_id = Some(pending);
        Ok(())
    }

    /// Close the generation: timing, the tail, and the accounting.
    pub fn finish(self) -> GenerationReport {
        let decode_seconds = if self.generated_ids.is_empty() {
            0.0
        } else {
            self.decode_started.elapsed().as_secs_f64()
        };
        let rest = self
            .tokenizer
            .decode(&self.generated_ids, true)
            .map(|full| full[self.emitted_bytes.min(full.len())..].to_string())
            .unwrap_or_default();
        GenerationReport {
            finish_reason: self.finish_reason,
            prompt_token_count: self.prompt_token_count,
            generated_token_count: self.generated_ids.len(),
            drafted_token_count: self.drafted_count,
            accepted_draft_token_count: self.accepted_draft_count,
            prefill_seconds: self.prefill_seconds,
            decode_seconds,
            rest,
            context_ids: self.context_ids,
        }
    }
}

impl Iterator for Generation<'_> {
    type Item = LibQuestResult<GenerationStep>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.step() {
            Ok(Some(step)) => Some(Ok(step)),
            Ok(None) => None,
            Err(e) => {
                self.failed = true;
                Some(Err(e))
            }
        }
    }
}
