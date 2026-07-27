//! The era engine: the loaded model, behind the container's trait.
use crate::*;

/// The model, its tokenizer, and the conversation so far.
pub(crate) struct EraEngine {
    model: lib::OlmoHybrid,
    tokenizer: tokenizers::Tokenizer,
    coordinate: String,
    trail: Vec<u32>,
}

impl EraEngine {
    /// Load from the standing default profile.
    ///
    /// Called on the container's own thread through the factory, which is
    /// what makes a not-`Send` model legal here at all.
    pub(crate) fn load() -> Result<Self, String> {
        let config = lib::LibConfig::load_from_dir(None::<&str>, None::<&str>).map_err(message_of)?;
        let mut settings = lib::LibSettings::load_from_dir(None::<&str>, None::<&str>).map_err(message_of)?;
        // Captured graphs bake buffer addresses, and a daemon restores
        // and clears across turns for the life of the process.
        settings.graph = false;

        #[cfg(feature = "cuda")]
        let (device, dtype) = (
            candle_core::Device::new_cuda(0).map_err(message_of)?,
            candle_core::DType::BF16,
        );
        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

        let model_dir = config.model_dir();
        let checkpoint = lib::load_config(&model_dir).map_err(message_of)?;
        let tokenizer = lib::load_tokenizer(&model_dir).map_err(message_of)?;
        lib::verify_token_map(&tokenizer).map_err(message_of)?;
        let weights = lib::load_weights(&config, dtype, &device).map_err(message_of)?;
        let model = lib::OlmoHybrid::new(&checkpoint, settings, weights).map_err(message_of)?;

        Ok(Self {
            model,
            tokenizer,
            coordinate: config.model,
            trail: Vec::new(),
        })
    }
}

impl container::Engine for EraEngine {
    /// Report what is loaded.
    ///
    /// THE OPTIONS ARE NOT CONSULTED, and that follows from the singleton
    /// rather than being an oversight. One container owns one model for
    /// the service's life, so a session cannot ask for a different
    /// checkpoint or a different settings profile - what it can do is
    /// learn which one it got, which is what the answer carries.
    fn open(
        &mut self,
        _options: &bridge::all::ChatOptions,
    ) -> Result<bridge::all::EraInfo, String> {
        Ok(bridge::all::EraInfo {
            era: String::from("two"),
            model: self.coordinate.clone(),
        })
    }

    fn turn(
        &mut self,
        text: &str,
        chunk: &mut dyn FnMut(bridge::TurnChunk) -> bool,
    ) -> Result<bridge::all::TurnReport, String> {
        let options = self.model.settings().generation.clone();
        let started = if self.trail.is_empty() {
            self.model.generate(&self.tokenizer, text, &options)
        } else {
            let restored = lib::RestoredContext {
                context_len: self.model.context_len(),
                context_ids: self.trail.clone(),
            };
            self.model
                .generate_from(&self.tokenizer, &restored, text, &options)
        };
        let mut generation = started.map_err(message_of)?;

        let mut listening = true;
        for step in &mut generation {
            let step = step.map_err(message_of)?;
            if step.chunk.is_empty() {
                continue;
            }
            if !chunk(bridge::TurnChunk { text: step.chunk }) {
                listening = false;
                break;
            }
        }

        let report = generation.finish();
        if listening && !report.rest.is_empty() {
            chunk(bridge::TurnChunk {
                text: report.rest.clone(),
            });
        }
        self.trail = report.context_ids;

        Ok(bridge::all::TurnReport {
            finish: match report.finish_reason {
                Some(lib::FinishReason::StopToken) => bridge::all::FinishReason::StopToken,
                Some(lib::FinishReason::SampleLen) => bridge::all::FinishReason::SampleLen,
                None => bridge::all::FinishReason::Cancelled,
            },
            prompt_token_count: report.prompt_token_count,
            generated_token_count: report.generated_token_count,
            prefill_seconds: report.prefill_seconds,
            decode_seconds: report.decode_seconds,
        })
    }

    fn reset(&mut self) -> Result<(), String> {
        self.trail.clear();
        self.model.clear_cache().map_err(message_of)
    }
}

/// Library errors cross as text, because the container's trait is the
/// seam between an era and a daemon that must not know which era.
fn message_of(error: lib::LibQuestError) -> String {
    error.to_string()
}