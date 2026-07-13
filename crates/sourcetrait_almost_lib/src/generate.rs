use crate::*;

/// Sampling and length knobs for one generation run.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub greedy: bool,
    pub temperature: f64,
    pub top_p: f64,
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
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    /// Speculation tallies (zero when speculate was off or never fired).
    pub drafted_token_count: usize,
    pub accepted_draft_token_count: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub rest: Option<String>,
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
    fed_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    dump_rows: Vec<candle_core::Tensor>,
    pending: Option<u32>,
    /// Speculation-verified tokens awaiting emission; already consumed
    /// into the caches and counted in offset.
    queued: VecDeque<u32>,
    index: LookupIndex,
    drafted_count: usize,
    accepted_draft_count: usize,
    offset: usize,
    finish_reason: Option<FinishReason>,
    failed: bool,
    prefill_seconds: f64,
    decode_seconds: f64,
}

impl Model {
    /// Tokenize, prefill (from a cleared cache), and sample the first
    /// token; the returned iterator then decodes one token per next().
    pub fn generate<'m, 't>(
        &'m mut self,
        tokenizer: &'t tokenizers::Tokenizer,
        text: &str,
        options: &GenerateOptions,
    ) -> AlmostResult<Generation<'m, 't>> {
        let encoding = match tokenizer.encode(text, true) {
            Ok(encoding) => encoding,
            Err(error) => snafu::whatever!("prompt tokenization failed: {error}"),
        };
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        snafu::ensure_whatever!(!prompt_ids.is_empty(), "the prompt tokenized to zero tokens");
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
            index.extend(&prompt_ids);
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

        self.clear_kv_cache();
        // The prompt length is known here: fine-grain reserve so the
        // context tail does not pay FULL_CACHE_STEP rounding; the margin
        // covers a short decode and longer decodes grow coarsely as before.
        self.reserve_full_caches(prompt_ids.len() + consts::RESERVE_DECODE_MARGIN)?;
        // Chunked prefill: logit-exact vs a single forward. The eager
        // chunk bounds the attention transient; the flash chunk trades
        // throughput against the 32K peak (consts).
        let prefill_chunk = if self.flash_enabled() {
            consts::PREFILL_CHUNK_FLASH
        } else {
            consts::PREFILL_CHUNK_EAGER
        };
        let prefill_start = Instant::now();
        let mut last_logits: Option<candle_core::Tensor> = None;
        let mut chunk_start = 0usize;
        while chunk_start < prompt_ids.len() {
            let chunk_end = (chunk_start + prefill_chunk).min(prompt_ids.len());
            let input =
                candle_core::Tensor::new(&prompt_ids[chunk_start..chunk_end], self.device())?.unsqueeze(0)?;
            let chunk_logits = if dumping {
                let all = self.forward_all(&input, chunk_start)?.squeeze(0)?;
                dump_rows.push(all.to_device(&candle_core::Device::Cpu)?.to_dtype(candle_core::DType::F32)?);
                all.narrow(0, chunk_end - chunk_start - 1, 1)?
            } else {
                self.forward(&input, chunk_start)?.squeeze(0)?
            };
            last_logits = Some(chunk_logits);
            chunk_start = chunk_end;
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

        let offset = prompt_ids.len();
        Ok(Generation {
            model: self,
            stream: TokenStream::new(tokenizer),
            processor,
            options: options.clone(),
            stop_ids,
            prompt_ids,
            fed_ids: Vec::new(),
            generated_ids: Vec::new(),
            dump_rows,
            pending: Some(first),
            queued: VecDeque::new(),
            index,
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
    pub fn finish(mut self) -> AlmostResult<GenerationReport> {
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
        Ok(GenerationReport {
            finish_reason: self.finish_reason,
            prompt_token_count: self.prompt_ids.len(),
            generated_token_count: self.generated_ids.len(),
            drafted_token_count: self.drafted_count,
            accepted_draft_token_count: self.accepted_draft_count,
            prefill_seconds: self.prefill_seconds,
            decode_seconds: self.decode_seconds,
            rest,
        })
    }

    /// The decode work of one next(): forward the yielded token, stage the
    /// next sample.
    fn stage_next(&mut self, token: u32) -> AlmostResult<u32> {
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
    fn stage_or_speculate(&mut self, token: u32) -> AlmostResult<()> {
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
        let draft = if self.options.speculate && budget > 0 {
            self.index.draft(budget)
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
    type Item = AlmostResult<GenerationStep>;

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
) -> AlmostResult<()> {
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
