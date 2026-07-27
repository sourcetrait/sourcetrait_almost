//! Era two behind the bridge API: the engine, and the entry that builds
//! it.
use crate::*;

/// Era two: the Olmo-Hybrid DPO checkpoint on the lib_quest engine.
///
/// This is the whole of what eon names. Everything era-specific below it
/// is reached through the API's traits, so a consumer generic over `Era`
/// has no path into the library at all.
pub struct BridgeTwo;

impl bridge::all::Era for BridgeTwo {
    type Engine = TwoEngine;

    /// From constants rather than a load, so a consumer can say what it
    /// is about to open before paying for the model.
    fn info() -> bridge::all::EraInfo {
        bridge::all::EraInfo {
            era: String::from("two"),
            model: format!(
                "{}/{}",
                lib::consts::MODEL_AUTHOR,
                lib::consts::DPO_MODEL_NAME
            ),
        }
    }

    fn engine() -> Result<Self::Engine, String> {
        TwoEngine::load()
    }
}

/// The model, its tokenizer, and the conversation so far.
pub struct TwoEngine {
    model: lib::OlmoHybrid,
    tokenizer: tokenizers::Tokenizer,
    coordinate: String,
    trail: Vec<u32>,
}

impl TwoEngine {
    /// Load from the standing default profile.
    ///
    /// Called through `Era::engine` on the thread that will own the
    /// result, which is what makes a not-`Send` model legal here.
    fn load() -> Result<Self, String> {
        let config =
            lib::LibConfig::load_from_dir(None::<&str>, None::<&str>).map_err(message_of)?;
        let mut settings =
            lib::LibSettings::load_from_dir(None::<&str>, None::<&str>).map_err(message_of)?;
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

impl bridge::all::Engine for TwoEngine {
    /// Report what is loaded.
    ///
    /// THE OPTIONS ARE NOT CONSULTED, and that follows from one container
    /// owning one model for the service's life rather than from anything
    /// unfinished here. A session cannot ask for a different checkpoint
    /// or settings profile; what it can do is learn which one it got.
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

/// Library errors cross as TEXT, and that is the seam doing its job: an
/// eon consumer must not have to name an era's error type to report a
/// failure, or the API would leak the library it exists to hide.
fn message_of(error: lib::LibQuestError) -> String {
    error.to_string()
}
