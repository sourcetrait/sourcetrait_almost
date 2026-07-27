//! Questness itself: it owns the turn and arbitrates its sub-turns.
use crate::*;

/// What one emission moved the turn to.
#[derive(Debug, Clone)]
pub enum Step {
    /// A sub-turn ran; send this back to the model and read it again.
    Continue(String),
    Answered(turn::Answer),
    /// The model said it cannot produce conforming output.
    Insufficient,
    /// The emission did not conform; the envelope is the feedback.
    Repair(Envelope),
}

/// The turn's owner: it renders, reads, and arbitrates.
///
/// Generic over the harness rather than boxing it, because one instance
/// talks to one harness for its life.
pub struct Questness<H: harness::ClientHarness> {
    evaluator: QuestnessEvaluator,
    harness: H,
    aliasing: sourcetrait_quest_core::channel::Aliasing,
}

impl<H: harness::ClientHarness> Questness<H> {
    pub fn new(
        harness: H,
        aliasing: sourcetrait_quest_core::channel::Aliasing,
    ) -> QuestHarnessResult<Self> {
        Ok(Self {
            evaluator: QuestnessEvaluator::new()?,
            harness,
            aliasing,
        })
    }

    /// The engine that runs the one mode staying inside.
    pub fn evaluator(&self) -> &QuestnessEvaluator {
        &self.evaluator
    }

    /// The text a turn shows the model.
    pub fn assemble(&self, request: &turn::Request) -> QuestHarnessResult<turn::Assembled> {
        turn::assemble(request)
    }

    /// Read one emission and take the turn to its next state.
    pub fn step(&mut self, emission: &str, insufficient: bool) -> QuestHarnessResult<Step> {
        let outcome = turn::interpret(&self.evaluator, emission, self.aliasing, insufficient)?;
        match outcome {
            turn::Outcome::Answered(answer) => Ok(Step::Answered(answer)),
            turn::Outcome::Insufficient => Ok(Step::Insufficient),
            turn::Outcome::Repair(envelope) => Ok(Step::Repair(envelope)),
            turn::Outcome::SubTurn(sub) => self.serve(&sub),
        }
    }

    /// Run a sub-turn wherever it belongs, and render its result back.
    fn serve(&mut self, sub: &turn::SubTurn) -> QuestHarnessResult<Step> {
        let response = match sub.destination {
            turn::Destination::Inside => {
                match turn::run_inside(&self.evaluator, sub) {
                    Ok(value) => harness::HarnessResponse::value(value),
                    Err(error) => {
                        harness::HarnessResponse::failed("questness::evaluate", &error.to_string())
                    }
                }
            }
            turn::Destination::Client => self.harness.serve(&request_for(sub))?,
        };
        match response.as_block() {
            Ok(block) => Ok(Step::Continue(
                sourcetrait_quest_core::channel::render_block(&block),
            )),
            Err(diagnostic) => {
                let mut envelope = Envelope::default();
                envelope.errors.push(diagnostic);
                Ok(Step::Repair(envelope))
            }
        }
    }
}

/// A sub-turn as the request that crosses to a client harness.
pub fn request_for(sub: &turn::SubTurn) -> harness::HarnessRequest {
    harness::HarnessRequest {
        mode: sub.contract.mode.clone(),
        source: sub.source.clone(),
        output: sub.contract.output.to_string(),
        bindings: sub
            .bindings
            .iter()
            .map(|binding| harness::RequestBinding {
                pass: binding.pass.clone(),
                value: harness::QuestNuValue::new(binding.value.clone()),
            })
            .collect(),
    }
}
