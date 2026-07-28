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
    aliasing: channel::Aliasing,
    log: Option<session::SessionLog>,
}

impl<H: harness::ClientHarness> Questness<H> {
    pub fn new(
        harness: H,
        aliasing: channel::Aliasing,
    ) -> LibQuestResult<Self> {
        Ok(Self {
            evaluator: QuestnessEvaluator::new()?,
            harness,
            aliasing,
            log: None,
        })
    }

    /// Record this session's low-level text to the given log.
    pub fn logging_to(mut self, log: session::SessionLog) -> Self {
        self.log = Some(log);
        self
    }

    /// The text a turn shows the model.
    pub fn assemble(&self, request: &turn::Request) -> LibQuestResult<turn::Assembled> {
        let assembled = turn::assemble(request)?;
        self.record("assembled", &assembled.text);
        Ok(assembled)
    }

    /// Read one emission and take the turn to its next state.
    pub fn step(&mut self, emission: &str, insufficient: bool) -> LibQuestResult<Step> {
        self.record("emission", emission);
        let outcome = turn::interpret(&self.evaluator, emission, self.aliasing, insufficient)?;
        let step = match outcome {
            turn::Outcome::Answered(answer) => Step::Answered(answer),
            turn::Outcome::Insufficient => Step::Insufficient,
            turn::Outcome::Repair(envelope) => Step::Repair(envelope),
            turn::Outcome::SubTurn(sub) => self.serve(&sub)?,
        };
        match &step {
            Step::Continue(text) => self.record("continue", text),
            Step::Answered(answer) => self.record("answered", &answer.rendered),
            Step::Insufficient => self.record("insufficient", ""),
            Step::Repair(envelope) => {
                let rows: Vec<String> = envelope
                    .errors
                    .iter()
                    .map(|row| format!("{}: {}", row.kind, row.message))
                    .collect();
                self.record("repair", &rows.join("\n"));
            }
        }
        Ok(step)
    }

    /// Append to the session log, if one is attached.
    ///
    /// A logging failure never fails a turn: the log is a record of the
    /// work rather than part of it.
    fn record(&self, label: &str, body: &str) {
        if let Some(log) = &self.log {
            let _ = log.append(label, body);
        }
    }

    /// Run a sub-turn wherever it belongs, and render its result back.
    fn serve(&mut self, sub: &turn::SubTurn) -> LibQuestResult<Step> {
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
                channel::render_block(&block),
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
pub(crate) fn request_for(sub: &turn::SubTurn) -> harness::HarnessRequest {
    harness::HarnessRequest {
        mode: sub.contract.head.clone(),
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
