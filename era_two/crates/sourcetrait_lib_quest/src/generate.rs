//! The generation surface: pull-based token production over the
//! carried forward path (era-one's settled API shape, era-two
//! implementation).
//!
//! The caller drives the loop: `Model::generate` prefills chunked and
//! samples the first token; each `Generation::next` yields one
//! `GenerationStep` (per-step errors ride the Item and fuse the
//! iterator); `finish` flushes the detokenizer tail and reports.
//! Dropping the iterator early is a clean stop. A sampled stop token
//! is never consumed into the caches, so a saved context stays
//! transcript-complete mid assistant turn.
use crate::*;

/// The prefill chunk fed per forward on the generate path.
pub(crate) const PREFILL_CHUNK: usize = 512;

/// Sampling posture + budget for one generation. The defaults are the
/// checkpoint card's recommended posture (temperature 0.6, top_p
/// 0.95) with the model-card decode budget; None temperature = greedy.
/// NoPE leaves the hybrid without a position ceiling, so sample_len
/// is a plain budget - VRAM is the real bound (the era-one 65536
/// clamp deliberately does not carry).
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub sample_len: usize,
    /// Render the prompt through the chat template; false feeds the
    /// text verbatim (the parity/battery rig).
    pub chat: bool,
    /// Consume stop tokens like any other id and decode the full
    /// budget - the bench posture (the incumbent rows force their
    /// decode length the same way). Never a chat behavior.
    pub ignore_stops: bool,
    /// SpeculationPort lookup speculation: draft continuations from
    /// earlier context occurrences, verify in one batched carried
    /// forward. Greedy-only (verification rides the same argmax
    /// sampler - token-exact vs plain greedy at f32); refuses
    /// eviction-armed settings and keeps the classic decode path
    /// (graphs never arm on a speculative run).
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
    /// Greedy argmax with a tight budget - the battery/parity rig.
    pub fn greedy(sample_len: usize) -> Self {
        Self {
            temperature: None,
            top_p: None,
            sample_len,
            ..Self::default()
        }
    }
}

/// One produced token and the text it detokenized to (possibly empty
/// while a multi-token grapheme is pending).
pub struct GenerationStep {
    pub token_id: u32,
    pub chunk: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token was sampled (and not consumed into the caches).
    StopToken,
    /// The sample_len budget was spent.
    SampleLen,
}

/// The end-of-generation accounting `finish` returns.
pub struct GenerationReport {
    /// None when the caller stopped early (dropped before an end).
    pub finish_reason: Option<FinishReason>,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    /// SpeculationPort tallies (zero when speculation was off or
    /// never fired).
    pub drafted_token_count: usize,
    pub accepted_draft_token_count: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    /// Detokenizer tail not yet emitted through the steps.
    pub rest: String,
    /// The consumed-id trail: prompt ids + consumed decode ids =
    /// exactly the cache contents (what a snapshot save wants). A
    /// stop-token end is transcript-complete (the stop is sampled,
    /// never consumed); a sample_len end leaves the final emitted
    /// token out of the KV.
    pub context_ids: Vec<u32>,
}

/// The pull-based generation iterator, borrowing the model (whose
/// caches hold the growing context) and the tokenizer.
pub struct Generation<'a> {
    model: &'a mut OlmoHybrid,
    tokenizer: &'a tokenizers::Tokenizer,
    processor: r::sampling::LogitsProcessor,
    stop_ids: Vec<u32>,
    ignore_stops: bool,
    generated_ids: Vec<u32>,
    /// The consumed-id trail (prompt + consumed decode ids); grows in
    /// lockstep with the caches.
    context_ids: Vec<u32>,
    emitted_bytes: usize,
    pending_id: Option<u32>,
    /// SpeculationPort: verification-accepted tokens awaiting
    /// emission - already consumed into the caches and the trail, so
    /// their yields owe no forward.
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
    /// Clear the caches, render + encode the prompt, prefill it
    /// chunked, sample the first token, and hand back the iterator.
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

    /// PrefixSnapshots ContinueSurface: continue a RESTORED context.
    /// The suffix renders as the next chat turn (chat_continue: close
    /// the open assistant turn, user turn, assistant opener) unless
    /// options.chat is false (verbatim suffix - the parity/battery
    /// rig). It prefills chunked AT THE RESTORED OFFSET, then the
    /// normal post-prefill sequence runs (KvEviction compaction when
    /// restored + suffix exceeds the cap - the whole-store re-score
    /// rebuilt the scores during the suffix prefill - and the graph
    /// arm). The same Generation iterator comes back; its trail is
    /// seeded restored.context_ids + suffix.
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

    /// The shared generation core: encode `rendered`, prefill it
    /// chunked at the CURRENT offset (zero for a fresh generate, the
    /// restored length for a continuation), run the post-prefill
    /// sequence, and sample the first token. `carried_trail` seeds
    /// the consumed-id trail (the restored trail for continuations,
    /// empty for fresh).
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
            snafu::ensure_whatever!(
                self.settings().eviction.is_none(),
                "speculation under eviction is not supported yet (run eviction-off)"
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
            // The whole committed context (a carried trail included)
            // seeds the index; the emitted tokens join per round.
            generation.index.extend(&generation.context_ids);
        }

        let prefill_started = std::time::Instant::now();
        // The known-length pre-reserve: the whole run's KV capacity
        // in one allocation, ahead of the first chunk (the carried
        // context included on a continuation).
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
            // PrefillLogitsSkip: intermediate chunks advance the
            // caches without computing logits; only the final chunk
            // pays the norm + head, over its last row alone.
            if start + len == prompt_ids.len() {
                last_logits = Some(generation.model.forward_chunk_last(&chunk)?);
            } else {
                generation.model.forward_chunk_carry(&chunk)?;
            }
            start += len;
        }
        let last_logits = last_logits.expect("non-empty prompt");
        // KvEviction: the one-shot post-prefill compaction precedes
        // graph arming, so graphs capture against the compacted store.
        if let Some(evict) = generation.model.settings().eviction.clone() {
            generation.model.evict_post_prefill(&evict)?;
        }
        // Arm the staged graph decode after prefill (settings.graph;
        // cuda builds only - Model::new already rejected the rest).
        // Speculation keeps the classic path silently (the era-one
        // scoping): variable-length verify chunks would churn buckets.
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

    /// The incremental detokenizer: the skip-special decode of all
    /// generated ids, minus what earlier steps already emitted. Stays
    /// prefix-consistent with the final full decode by construction.
    fn emit_chunk(&mut self) -> LibQuestResult<String> {
        let full = match self.tokenizer.decode(&self.generated_ids, true) {
            Ok(text) => text,
            Err(e) => snafu::whatever!("detokenize failed: {e}"),
        };
        if full.len() <= self.emitted_bytes {
            return Ok(String::new());
        }
        let chunk = full[self.emitted_bytes..].to_string();
        // Hold back a chunk that ends mid-replacement-character (a
        // pending multi-token grapheme decodes as U+FFFD until its
        // continuation arrives).
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
            // SpeculationPort: a verification-accepted token - already
            // consumed into the caches and the trail; no forward owed.
            return Ok(Some(GenerationStep { token_id, chunk }));
        }
        if self.speculate {
            self.stage_or_speculate(token_id)?;
            return Ok(Some(GenerationStep { token_id, chunk }));
        }
        // KvEviction overflow epoch: decode outgrew the cap by the
        // slack - re-compact between steps (uncaptured host work; an
        // armed graph keeps replaying at the unchanged bucket).
        if let Some(evict) = self.model.settings().eviction.clone()
            && self.model.context_len() >= evict.decode_cap + evict::OVERFLOW_SLACK
        {
            self.model.evict_overflow(&evict)?;
        }
        let logits = if self.model.graph_armed() {
            // A decode outrunning the armed capacity falls to the
            // classic path (the capacity epoch).
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
        // The token is consumed into the caches now - the trail
        // advances in lockstep.
        self.context_ids.push(token_id);
        self.pending_id = Some(self.sample(&logits)?);
        Ok(Some(GenerationStep { token_id, chunk }))
    }

    /// One SpeculationPort round for the emitted token: index it,
    /// probe the ladder, and either verify [token ++ draft] in one
    /// batched carried forward or fall to the plain single step. On a
    /// partial accept the caches roll back to the mark and the
    /// accepted rows re-advance logits-free (the GDN caches are
    /// cumulative - the shadow restore + replay replaces era-one's
    /// exact ring rewind); a full accept keeps the caches as-is.
    /// Token-exact vs plain greedy: every row rides the same argmax
    /// sampler.
    fn stage_or_speculate(&mut self, token_id: u32) -> LibQuestResult<()> {
        // The emitted token is committed context NOW - it must join
        // the index before drafting, or every draft continues the
        // pre-token tail and competes with the token itself.
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
            // Plain single step (classic path; speculation never arms
            // graphs).
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
            // Partial accept: restore the mark and re-advance the
            // accepted rows logits-free (the bonus row is already in
            // hand from the verification logits).
            self.model.spec_rollback(&mark)?;
            let replay = candle_core::Tensor::from_vec(
                accepted.clone(),
                consumed,
                self.model.device(),
            )?;
            self.model.forward_chunk_carry(&replay)?;
        }
        self.context_ids.extend_from_slice(&accepted);
        // token_id is already indexed; the verified continuation
        // joins now.
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

    /// Close the generation: decode timing, the detokenizer tail, and
    /// the accounting.
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
