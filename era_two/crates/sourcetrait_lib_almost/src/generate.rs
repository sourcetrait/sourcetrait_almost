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
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    /// Detokenizer tail not yet emitted through the steps.
    pub rest: String,
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
    emitted_bytes: usize,
    pending_id: Option<u32>,
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
    ) -> LibAlmostResult<Generation<'a>> {
        self.clear_cache()?;
        let rendered = if options.chat {
            chat_wrap(prompt)
        } else {
            prompt.to_string()
        };
        let encoding = match tokenizer.encode(rendered, false) {
            Ok(encoding) => encoding,
            Err(e) => snafu::whatever!("prompt encode failed: {e}"),
        };
        let prompt_ids = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            snafu::whatever!("empty prompt after rendering");
        }

        let sampling = match (options.temperature, options.top_p) {
            (None, _) => r::sampling::Sampling::ArgMax,
            (Some(temperature), Some(p)) => r::sampling::Sampling::TopP { p, temperature },
            (Some(temperature), None) => r::sampling::Sampling::All { temperature },
        };
        let mut generation = Generation {
            processor: r::sampling::LogitsProcessor::from_sampling(options.seed, sampling),
            stop_ids: resolve_stop_ids(tokenizer),
            ignore_stops: options.ignore_stops,
            generated_ids: Vec::new(),
            emitted_bytes: 0,
            pending_id: None,
            sample_len: options.sample_len,
            finish_reason: None,
            failed: false,
            prompt_token_count: prompt_ids.len(),
            prefill_seconds: 0.0,
            decode_started: std::time::Instant::now(),
            model: self,
            tokenizer,
        };

        let prefill_started = std::time::Instant::now();
        let mut last_logits = None;
        let mut start = 0;
        while start < prompt_ids.len() {
            let len = PREFILL_CHUNK.min(prompt_ids.len() - start);
            let chunk = candle_core::Tensor::from_vec(
                prompt_ids[start..start + len].to_vec(),
                len,
                generation.model.device(),
            )?;
            let logits = generation.model.forward_chunk(&chunk)?;
            last_logits = Some(logits.narrow(0, len - 1, 1)?);
            start += len;
        }
        let last_logits = last_logits.expect("non-empty prompt");
        // Arm the staged graph decode after prefill (settings.graph;
        // cuda builds only - Model::new already rejected the rest).
        if generation.model.settings().graph {
            let expected_total = generation.prompt_token_count + generation.sample_len;
            generation.model.arm_graph_decode(expected_total)?;
        }
        generation.pending_id = Some(generation.sample(&last_logits)?);
        generation.prefill_seconds = prefill_started.elapsed().as_secs_f64();
        generation.decode_started = std::time::Instant::now();
        Ok(generation)
    }
}

impl Generation<'_> {
    fn sample(&mut self, logits_row: &candle_core::Tensor) -> LibAlmostResult<u32> {
        let row = logits_row
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .to_device(&candle_core::Device::Cpu)?;
        Ok(self.processor.sample(&row)?)
    }

    /// The incremental detokenizer: the skip-special decode of all
    /// generated ids, minus what earlier steps already emitted. Stays
    /// prefix-consistent with the final full decode by construction.
    fn emit_chunk(&mut self) -> LibAlmostResult<String> {
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

    fn step(&mut self) -> LibAlmostResult<Option<GenerationStep>> {
        if self.finish_reason.is_some() || self.failed {
            return Ok(None);
        }
        let Some(token_id) = self.pending_id.take() else {
            return Ok(None);
        };
        if !self.ignore_stops && self.stop_ids.contains(&token_id) {
            self.finish_reason = Some(FinishReason::StopToken);
            return Ok(None);
        }
        self.generated_ids.push(token_id);
        let chunk = self.emit_chunk()?;
        if self.generated_ids.len() >= self.sample_len {
            self.finish_reason = Some(FinishReason::SampleLen);
            return Ok(Some(GenerationStep { token_id, chunk }));
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
        self.pending_id = Some(self.sample(&logits)?);
        Ok(Some(GenerationStep { token_id, chunk }))
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
            prefill_seconds: self.prefill_seconds,
            decode_seconds,
            rest,
        }
    }
}

impl Iterator for Generation<'_> {
    type Item = LibAlmostResult<GenerationStep>;

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
