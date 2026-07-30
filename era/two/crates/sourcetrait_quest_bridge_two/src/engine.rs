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
    type Questness = TwoQuestness;

    /// From constants rather than a load, so a consumer can say what it
    /// is about to open before paying for the model.
    fn info() -> bridge::all::EraInfo {
        bridge::all::EraInfo {
            era: String::from("two"),
            model: format!(
                "{}/{}",
                llm::consts::MODEL_AUTHOR,
                llm::consts::DPO_MODEL_NAME
            ),
        }
    }

    fn engine(options: &bridge::all::ChatOptions) -> Result<Self::Engine, String> {
        TwoEngine::load(options)
    }

    fn questness() -> Result<Self::Questness, String> {
        TwoQuestness::open()
    }
}

/// Era two's Questness behind the API's trait.
///
/// It adds nothing the library does not already provide: the whole of it
/// is the conversion between the `Infer*` forms and what the library's
/// own turn speaks.
pub struct TwoQuestness {
    inner: harness::Questness,
    /// The last assembled request's conversation disposition.
    keep: bool,
}

impl TwoQuestness {
    fn open() -> Result<Self, String> {
        // The boundary aliases the trained tool-marker attractor until
        // channel-marker routing is trained, which 10_Channels measures
        // as dead at zero of six without it.
        let inner = harness::Questness::new(harness::channel::Aliasing::ToolMarkers).map_err(message_of)?;
        Ok(Self { inner, keep: false })
    }
}

impl bridge::all::Questness for TwoQuestness {
    /// A request carrying an `output` is a CONTINUATION: the caller ran
    /// an ask and this is its result, so the conversation resumes rather
    /// than starting a fresh turn. Every request's own config decides
    /// the disposition its turn ends with; a continuation carrying none
    /// ends with the default teardown.
    fn assemble(&mut self, request: &bridge::InferRequest) -> Result<String, String> {
        let config = match &request.config {
            Some(config) => config.0.0.clone(),
            None => harness::nu::Value::record(harness::nu::Record::new(), harness::nu::Span::unknown()),
        };
        self.keep = matches!(
            harness::Conversation::of(&config).map_err(message_of)?,
            harness::Conversation::Keep
        );
        if let Some(output) = &request.output {
            return self.inner.resume(&output.value().0).map_err(message_of);
        }
        let bindings = request
            .inputs
            .iter()
            .map(|(pass, input)| harness::Binding::new(pass.spelling(), input.value().0.clone()))
            .collect();
        let assembled = self
            .inner
            .assemble(&harness::Request {
                config,
                prompt: request.text.as_ref().map(|text| text.0.clone()).unwrap_or_default(),
                bindings,
            })
            .map_err(message_of)?;
        Ok(assembled.text)
    }

    fn keeps_conversation(&self) -> bool {
        self.keep
    }

    fn step(
        &mut self,
        emission: &str,
        insufficient: bool,
    ) -> Result<bridge::all::Step, String> {
        Ok(match self.inner.step(emission, insufficient).map_err(message_of)? {
            harness::Step::Continue(text) => bridge::all::Step::Continue(text),
            harness::Step::Insufficient => bridge::all::Step::Insufficient,
            harness::Step::Ask { form, bindings } => bridge::all::Step::Ask {
                form,
                inputs: bindings
                    .iter()
                    .map(bound)
                    .collect::<Result<Vec<_>, String>>()?,
            },
            harness::Step::Answered(answer) => bridge::all::Step::Answered {
                output: answer.value.map(|value| {
                    bridge::InferOutput::Nuon(bridge::InferNuonOutput(bridge::InferValue(value)))
                }),
                text: match answer.rendered.is_empty() {
                    true => None,
                    false => Some(bridge::InferText(answer.rendered)),
                },
                config: answer
                    .config
                    .map(|value| bridge::InferConfig(bridge::InferValue(value))),
            },
            harness::Step::Repair(envelope) => bridge::all::Step::Repair(
                envelope
                    .errors
                    .iter()
                    .map(|row| bridge::all::InferDiagnostic {
                        kind: row.kind.clone(),
                        source: row.source.clone(),
                        message: row.message.clone(),
                    })
                    .collect(),
            ),
        })
    }
}

/// One of the library's bindings as the pass-tagged input it is.
fn bound(
    binding: &harness::Binding,
) -> Result<(bridge::InferPass, bridge::InferInput), String> {
    let pass = bridge::InferPass::parse(&binding.pass).map_err(|error| error.to_string())?;
    let carried = bridge::InferValue(binding.value.clone());
    Ok((pass, bridge::InferInput::Nuon(bridge::InferNuonInput(carried))))
}

/// The model, its tokenizer, and the conversation so far.
pub struct TwoEngine {
    model: llm::OlmoHybrid,
    tokenizer: tokenizers::Tokenizer,
    coordinate: String,
    trail: Vec<u32>,
}

impl TwoEngine {
    /// Load the posture the consumer's options name.
    ///
    /// Called through `Era::engine` on the thread that will own the
    /// result, which is what makes a not-`Send` model legal here. A
    /// consumer naming its own `dir` reads profiles from its own root,
    /// so it picks a posture - which adapter, above all - without
    /// writing into the suite's shared default and redefining what an
    /// un-tokened run means for every other consumer.
    fn load(options: &bridge::all::ChatOptions) -> Result<Self, String> {
        let config =
            llm::LibConfig::load_from_dir(options.dir.as_ref(), options.config.as_ref())
                .map_err(message_of)?;
        let mut settings =
            llm::LibSettings::load_from_dir(options.dir.as_ref(), options.settings.as_ref())
                .map_err(message_of)?;
        // Captured graphs bake buffer addresses, and a daemon restores
        // and clears across turns for the life of the process.
        settings.graph = false;

        // The candle error is lifted into the library's own before it is
        // flattened, because `message_of` speaks one error type and this
        // is the only call here that does not already produce it.
        #[cfg(feature = "cuda")]
        let (device, dtype) = (
            candle_core::Device::new_cuda(0)
                .map_err(message_of)?,
            candle_core::DType::BF16,
        );
        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

        let model_dir = config.model_dir();
        let checkpoint = llm::load_config(&model_dir).map_err(message_of)?;
        let tokenizer = llm::load_tokenizer(&model_dir).map_err(message_of)?;
        llm::verify_token_map(&tokenizer).map_err(message_of)?;
        let weights = llm::load_weights(&config, dtype, &device).map_err(message_of)?;
        let model = llm::OlmoHybrid::new(&checkpoint, settings, weights).map_err(message_of)?;

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
            let restored = llm::RestoredContext {
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
                Some(llm::FinishReason::StopToken) => bridge::all::FinishReason::StopToken,
                Some(llm::FinishReason::SampleLen) => bridge::all::FinishReason::SampleLen,
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
fn message_of<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}
