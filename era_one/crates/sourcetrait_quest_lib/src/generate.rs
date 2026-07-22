use crate::*;

/// Sampling and length knobs for one generation run.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub greedy: bool,
    pub temperature: f64,
    pub top_p: f64,
    /// Decode budget (tokens to yield); generation clamps it to the
    /// position ceiling for the run's context, so a deep session stops
    /// (SampleLen) at the window edge instead of erroring mid-reply.
    pub sample_len: usize,
    pub seed: u64,
    /// E5a prompt-lookup speculation: draft from earlier context
    /// occurrences, verify in one batched forward. Greedy only (the
    /// verification path reuses the same argmax sampler, so output is
    /// token-exact vs non-speculative greedy); incompatible with
    /// dump_logits.
    pub speculate: bool,
    /// When set, capture a parity dump - "logits" (rows, vocab) f32 over
    /// every fed position of [prompt ++ fed], plus prompt/fed/generated
    /// ids - written by finish(). No repeat penalty exists anywhere: logits
    /// stay unmodified so parity comparisons stay clean.
    pub dump_logits: Option<PathBuf>,
}

/// Why a finished generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A configured stop token was sampled (it is never yielded).
    StopToken,
    /// sample_len tokens were yielded.
    SampleLen,
}

/// One decode step's yield: the sampled token and whatever text became
/// printable with it (None while a multi-byte sequence is still open).
#[derive(Debug, Clone)]
pub struct GenerationStep {
    pub token_id: u32,
    pub chunk: Option<String>,
}

/// Returned by finish(): timing, counts, and the detokenizer tail the
/// ascii-boundary heuristic was still holding. finish_reason is None when
/// the run was stopped early or a step failed.
#[derive(Debug, Clone)]
pub struct GenerationReport {
    pub finish_reason: Option<FinishReason>,
    /// Full context ahead of decode: restored tokens plus this run's
    /// prefill.
    pub prompt_token_count: usize,
    /// Tokens THIS run prefilled (the whole prompt, or just the suffix
    /// on a generate_from continuation).
    pub prefilled_token_count: usize,
    pub generated_token_count: usize,
    /// Speculation tallies (zero when speculate was off or never fired).
    pub drafted_token_count: usize,
    pub accepted_draft_token_count: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub rest: Option<String>,
    /// The id trail whose KV the caches hold after this generation
    /// (context plus consumed decode) - exactly what snapshot_caches
    /// wants. A sample_len stop leaves the final EMITTED token out (it
    /// was never forwarded); a stop-token end is transcript-complete.
    pub context_ids: Vec<u32>,
}

/// Pull-based generation borrowing the model: each next() performs one
/// decode step and yields the token + decoded chunk, per-step errors ride
/// in the Item and fuse the iterator, dropping the iterator is early stop,
/// and finish() writes any armed dump and returns the report. The caller
/// drives the loop.
pub struct Generation<'m, 't> {
    model: &'m mut Model,
    stream: TokenStream<'t>,
    processor: r::candle::LogitsProcessor,
    options: GenerateOptions,
    stop_ids: Vec<u32>,
    prompt_ids: Vec<u32>,
    prefilled_count: usize,
    fed_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    dump_rows: Vec<candle_core::Tensor>,
    pending: Option<u32>,
    /// Speculation-verified tokens awaiting emission; already consumed
    /// into the caches and counted in offset.
    queued: VecDeque<u32>,
    index: LookupIndex,
    policy: DraftPolicy,
    drafted_count: usize,
    accepted_draft_count: usize,
    offset: usize,
    finish_reason: Option<FinishReason>,
    failed: bool,
    prefill_seconds: f64,
    decode_seconds: f64,
}

/// Tokenize `text`, labeled for the error message; errors when it
/// tokenizes to nothing.
fn encode_ids(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    what: &str,
) -> QuestResult<Vec<u32>> {
    let encoding = match tokenizer.encode(text, true) {
        Ok(encoding) => encoding,
        Err(error) => snafu::whatever!("{what} tokenization failed: {error}"),
    };
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    snafu::ensure_whatever!(!ids.is_empty(), "the {what} tokenized to zero tokens");
    Ok(ids)
}

impl Model {
    /// Tokenize, prefill (from a cleared cache), and sample the first
    /// token; the returned iterator then decodes one token per next().
    pub fn generate<'m, 't>(
        &'m mut self,
        tokenizer: &'t tokenizers::Tokenizer,
        text: &str,
        options: &GenerateOptions,
    ) -> QuestResult<Generation<'m, 't>> {
        let prompt_ids = encode_ids(tokenizer, text, "prompt")?;
        self.clear_kv_cache();
        self.start_generation(tokenizer, prompt_ids, 0, options)
    }

    /// E2: continue from a restored standing context - tokenize
    /// `suffix_text`, prefill it at the restored offset, and decode
    /// exactly as generate() does. The caches must hold the restored
    /// context (restore_caches puts them there); parity dumps are
    /// whole-context instruments and are refused here.
    pub fn generate_from<'m, 't>(
        &'m mut self,
        tokenizer: &'t tokenizers::Tokenizer,
        restored: &RestoredContext,
        suffix_text: &str,
        options: &GenerateOptions,
    ) -> QuestResult<Generation<'m, 't>> {
        snafu::ensure_whatever!(
            options.dump_logits.is_none(),
            "parity dumps run from a cleared cache; a restored context does not combine with --dump-logits"
        );
        let cached = self.max_full_cache_len();
        snafu::ensure_whatever!(
            cached == restored.context_len,
            "caches hold {cached} entries but the restored context says {}; restore first",
            restored.context_len
        );
        let suffix_ids = encode_ids(tokenizer, suffix_text, "suffix")?;
        let mut context_ids = restored.context_ids.clone();
        context_ids.extend_from_slice(&suffix_ids);
        self.start_generation(tokenizer, context_ids, restored.context_len, options)
    }

    /// The shared generation core: prefill context_ids[prefilled..] in
    /// chunks over whatever the caches already hold, run the stage-2
    /// compaction, sample the first token, arm graph decode when
    /// configured, and hand back the pull iterator.
    fn start_generation<'m, 't>(
        &'m mut self,
        tokenizer: &'t tokenizers::Tokenizer,
        context_ids: Vec<u32>,
        prefilled: usize,
        options: &GenerateOptions,
    ) -> QuestResult<Generation<'m, 't>> {
        snafu::ensure_whatever!(
            prefilled < context_ids.len(),
            "nothing to prefill (the context carries no new tokens)"
        );
        // The budget clamps to the position ceiling: a deep context
        // stops at the window edge (SampleLen) instead of erroring
        // mid-reply on the forward assert.
        let mut options = options.clone();
        options.sample_len = options
            .sample_len
            .min(self.max_position_embeddings().saturating_sub(context_ids.len()));
        let stop_ids = resolve_stop_ids(tokenizer);
        let dumping = options.dump_logits.is_some();
        if options.speculate {
            snafu::ensure_whatever!(
                options.greedy,
                "speculation is greedy-only (sampling verification is a later track)"
            );
            snafu::ensure_whatever!(
                !dumping,
                "speculation and --dump-logits do not combine; parity dumps run non-speculative"
            );
            snafu::ensure_whatever!(
                self.settings().eviction.is_none(),
                "speculation under eviction is not supported yet (run eviction-off)"
            );
        }
        let mut index = LookupIndex::new();
        if options.speculate {
            index.extend(&context_ids);
        }

        let sampling = if options.greedy {
            r::candle::Sampling::ArgMax
        } else {
            r::candle::Sampling::TopP {
                p: options.top_p,
                temperature: options.temperature,
            }
        };
        let mut processor = r::candle::LogitsProcessor::from_sampling(options.seed, sampling);
        let mut dump_rows: Vec<candle_core::Tensor> = Vec::new();

        // A run that will not arm the graph path (speculation keeps the
        // classic path; cpu and graph-off likewise) must not carry
        // captured graphs into classic append-growth - the regrow frees
        // buffers those graphs reference (the reserve-time flush below
        // covers arming runs).
        let will_arm = self.settings().graph && self.device().is_cuda() && !options.speculate;
        if !will_arm {
            self.flush_captured_graphs();
        }
        // The context length is known here: fine-grain reserve so the
        // context tail does not pay FULL_CACHE_STEP rounding; the margin
        // covers a short decode and longer decodes grow coarsely as before.
        self.reserve_full_caches(context_ids.len() + consts::RESERVE_DECODE_MARGIN)?;
        // Chunked prefill of the un-prefilled tail: logit-exact vs a
        // single forward. The eager chunk bounds the attention transient;
        // the flash chunk trades throughput against the 32K peak (consts).
        let prefill_chunk = if self.flash_enabled() {
            consts::PREFILL_CHUNK_FLASH
        } else {
            consts::PREFILL_CHUNK_EAGER
        };
        let prefill_start = Instant::now();
        let mut last_logits: Option<candle_core::Tensor> = None;
        let mut position = prefilled;
        while position < context_ids.len() {
            let end = (position + prefill_chunk).min(context_ids.len());
            let input =
                candle_core::Tensor::new(&context_ids[position..end], self.device())?.unsqueeze(0)?;
            let chunk_logits = if dumping {
                let all = self.forward_all(&input, position)?.squeeze(0)?;
                dump_rows.push(all.to_device(&candle_core::Device::Cpu)?.to_dtype(candle_core::DType::F32)?);
                all.narrow(0, end - position - 1, 1)?
            } else {
                self.forward(&input, position)?.squeeze(0)?
            };
            last_logits = Some(chunk_logits);
            position = end;
        }
        let logits = match last_logits {
            Some(logits) => logits.squeeze(0)?.to_dtype(candle_core::DType::F32)?,
            None => snafu::whatever!("prefill produced no logits"),
        };
        // A3 stage 2: the question-informed compaction to the decode cap,
        // now that scoring has seen the final chunk.
        self.compact_full_caches()?;
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();
        let first = processor.sample(&logits)?;
        // E4: arm the staged graph-mode decode (phase A runs the same
        // op sequence uncaptured). Speculation keeps the classic path
        // (v1 scoping); cpu decode is classic everywhere.
        if self.settings().graph && self.device().is_cuda() && !options.speculate {
            self.arm_graph_decode(context_ids.len() + options.sample_len + 1)?;
        }

        let offset = context_ids.len();
        let prefilled_count = offset - prefilled;
        Ok(Generation {
            model: self,
            stream: TokenStream::new(tokenizer),
            processor,
            options: options.clone(),
            stop_ids,
            prompt_ids: context_ids,
            prefilled_count,
            fed_ids: Vec::new(),
            generated_ids: Vec::new(),
            dump_rows,
            pending: Some(first),
            queued: VecDeque::new(),
            index,
            policy: DraftPolicy::new(),
            drafted_count: 0,
            accepted_draft_count: 0,
            offset,
            finish_reason: None,
            failed: false,
            prefill_seconds,
            decode_seconds: 0.0,
        })
    }
}

impl Generation<'_, '_> {
    /// Flush the detokenizer tail, write the dump when one is armed (and
    /// no step failed), and report. Valid after exhaustion or an early
    /// stop alike.
    pub fn finish(mut self) -> QuestResult<GenerationReport> {
        let rest = self.stream.decode_rest()?;
        if let Some(path) = &self.options.dump_logits
            && !self.failed
        {
            write_dump(
                path,
                &self.prompt_ids,
                &self.fed_ids,
                &self.generated_ids,
                std::mem::take(&mut self.dump_rows),
            )?;
        }
        let mut context_ids = self.prompt_ids.clone();
        context_ids.extend_from_slice(&self.fed_ids);
        Ok(GenerationReport {
            finish_reason: self.finish_reason,
            prompt_token_count: self.prompt_ids.len(),
            prefilled_token_count: self.prefilled_count,
            generated_token_count: self.generated_ids.len(),
            drafted_token_count: self.drafted_count,
            accepted_draft_token_count: self.accepted_draft_count,
            prefill_seconds: self.prefill_seconds,
            decode_seconds: self.decode_seconds,
            rest,
            context_ids,
        })
    }

    /// The decode work of one next(): forward the yielded token, stage the
    /// next sample. An armed graph stage routes through the staged decode
    /// step (E4); the classic path is unchanged. A decode that outruns
    /// the armed capacity disarms first (the capacity epoch) and
    /// continues classic.
    fn stage_next(&mut self, token: u32) -> QuestResult<u32> {
        self.model.graph_disarm_when_full()?;
        if self.model.graph_armed() {
            let step_logits = self.model.graph_decode_step(token, self.offset)?;
            self.fed_ids.push(token);
            if self.options.dump_logits.is_some() {
                self.dump_rows
                    .push(step_logits.to_device(&candle_core::Device::Cpu)?);
            }
            let step_logits = step_logits.squeeze(0)?;
            self.offset += 1;
            return Ok(self.processor.sample(&step_logits)?);
        }
        let input = candle_core::Tensor::new(&[token], self.model.device())?.unsqueeze(0)?;
        let step_logits = self.model.forward(&input, self.offset)?;
        self.fed_ids.push(token);
        if self.options.dump_logits.is_some() {
            self.dump_rows.push(
                step_logits
                    .squeeze(0)?
                    .to_device(&candle_core::Device::Cpu)?
                    .to_dtype(candle_core::DType::F32)?,
            );
        }
        let step_logits = step_logits.squeeze(0)?.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
        self.offset += 1;
        Ok(self.processor.sample(&step_logits)?)
    }

    /// One decode-or-verify round for the emitted token: with a lookup
    /// hit, verify [token ++ draft] in one batched forward, roll the
    /// caches back to the accepted prefix, queue the verified
    /// continuation, and stage the model's own next token (the
    /// correction/bonus row) - token-exact vs plain greedy because every
    /// row goes through the same argmax sampler. Without a hit, plain
    /// stage_next.
    fn stage_or_speculate(&mut self, token: u32) -> QuestResult<()> {
        // The emitted token is committed context NOW - it must be in the
        // index before drafting, or every draft continues the pre-token
        // tail and competes with the token itself (an off-by-one that
        // rejects everything).
        if self.options.speculate {
            self.index.extend(&[token]);
        }
        let budget = self
            .options
            .sample_len
            .saturating_sub(self.generated_ids.len());
        // The policy shapes the round: its ceiling caps the probe, the
        // matched ladder level seeds the final length (a 2-gram hit
        // drafts short until the run is hot), and a paused policy
        // skips drafting entirely.
        let draft = if self.options.speculate && budget > 0 {
            let probe = self.policy.probe_limit();
            self.index
                .draft(probe.min(budget))
                .map(|(matched_n, mut tokens)| {
                    tokens.truncate(self.policy.draft_limit(matched_n).min(budget));
                    tokens
                })
        } else {
            None
        };
        let Some(draft) = draft else {
            let next_token = self.stage_next(token)?;
            self.pending = Some(next_token);
            return Ok(());
        };

        let span = 1 + draft.len();
        let mark = self.model.cache_mark(self.offset, span)?;
        let mut block: Vec<u32> = Vec::with_capacity(span);
        block.push(token);
        block.extend_from_slice(&draft);
        let input = candle_core::Tensor::new(block.as_slice(), self.model.device())?.unsqueeze(0)?;
        let logits = self
            .model
            .forward_all(&input, self.offset)?
            .squeeze(0)?
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?;
        self.drafted_count += draft.len();

        let mut greedy_next: Vec<u32> = Vec::with_capacity(span);
        for row_index in 0..span {
            let row = logits.narrow(0, row_index, 1)?.squeeze(0)?;
            greedy_next.push(self.processor.sample(&row)?);
        }

        let mut accepted: Vec<u32> = vec![token];
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

        self.model.cache_rollback(&mark, consumed)?;
        self.offset += consumed;
        self.fed_ids.extend(&accepted);
        // token is already indexed; the verified continuation joins now.
        self.index.extend(&accepted[1..]);

        let mut queued: Vec<u32> = accepted[1..].to_vec();
        let mut pending = bonus;
        if let Some(stop_at) = queued.iter().position(|t| self.stop_ids.contains(t)) {
            pending = queued[stop_at];
            queued.truncate(stop_at);
        }
        self.queued = queued.into();
        self.pending = Some(pending);
        Ok(())
    }
}

impl Iterator for Generation<'_, '_> {
    type Item = QuestResult<GenerationStep>;

    fn next(&mut self) -> Option<Self::Item> {
        let (token, from_queue) = match self.queued.pop_front() {
            Some(queued_token) => (queued_token, true),
            None => match self.pending {
                Some(pending_token) => (pending_token, false),
                None => return None,
            },
        };
        if self.stop_ids.contains(&token) {
            self.pending = None;
            self.queued.clear();
            self.finish_reason = Some(FinishReason::StopToken);
            return None;
        }
        let step_start = Instant::now();
        self.generated_ids.push(token);
        let chunk = match self.stream.next_token(token) {
            Ok(chunk) => chunk,
            Err(error) => {
                self.pending = None;
                self.queued.clear();
                self.failed = true;
                return Some(Err(error));
            }
        };
        if self.generated_ids.len() >= self.options.sample_len {
            self.pending = None;
            self.queued.clear();
            self.finish_reason = Some(FinishReason::SampleLen);
            return Some(Ok(GenerationStep { token_id: token, chunk }));
        }
        if from_queue {
            // Verified continuation: already consumed into the caches,
            // no forward owed on this call.
            return Some(Ok(GenerationStep { token_id: token, chunk }));
        }
        let staged = self.stage_or_speculate(token);
        self.decode_seconds += step_start.elapsed().as_secs_f64();
        match staged {
            Ok(()) => Some(Ok(GenerationStep { token_id: token, chunk })),
            Err(error) => {
                self.pending = None;
                self.queued.clear();
                self.failed = true;
                Some(Err(error))
            }
        }
    }
}

/// Dump format shared with compat and heat: logits rows cover positions of
/// [prompt_ids ++ fed_ids]; a stateless replayer forwards exactly those ids.
fn write_dump(
    path: &Path,
    prompt_ids: &[u32],
    fed_ids: &[u32],
    generated_ids: &[u32],
    rows: Vec<candle_core::Tensor>,
) -> QuestResult<()> {
    let cpu = candle_core::Device::Cpu;
    let logits = candle_core::Tensor::cat(&rows, 0)?;
    let mut tensors = HashMap::new();
    tensors.insert(String::from("logits"), logits);
    tensors.insert(
        String::from("prompt_ids"),
        candle_core::Tensor::new(prompt_ids, &cpu)?,
    );
    tensors.insert(String::from("fed_ids"), candle_core::Tensor::new(fed_ids, &cpu)?);
    tensors.insert(
        String::from("generated_ids"),
        candle_core::Tensor::new(generated_ids, &cpu)?,
    );
    candle_core::safetensors::save(&tensors, path)?;
    Ok(())
}
