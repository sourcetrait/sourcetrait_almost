//! Questness itself: it owns the turn and arbitrates what it asks for.
use crate::*;

/// What one emission moved the turn to.
#[derive(Debug, Clone)]
pub enum Step {
    /// A think turn ran here; send this back to the model and read it
    /// again. The only state that keeps the turn going.
    Continue(String),
    Answered(turn::Answer),
    /// The model asked the CALLER's own engine to run something. The
    /// turn pauses: nothing here runs it, and the caller sends the
    /// result back on the next request.
    Ask {
        form: bridge::InferNu,
        bindings: Vec<turn::Binding>,
    },
    /// The model said it cannot produce conforming output.
    Insufficient,
    /// The emission did not conform; the envelope is the feedback.
    Repair(Envelope),
}

/// The turn's owner: it renders, reads, and arbitrates.
///
/// It holds no harness and calls nothing out. The one thing it runs is a
/// think turn, on the Thinkspace's own evaluator.
pub struct Questness {
    evaluator: QuestnessEvaluator,
    aliasing: channel::Aliasing,
    log: Option<session::SessionLog>,
}

impl Questness {
    pub fn new(aliasing: channel::Aliasing) -> LibQuestResult<Self> {
        Ok(Self {
            evaluator: QuestnessEvaluator::new()?,
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
    ///
    /// `thinking` says whether this emission may reach a reasoning mode.
    pub fn step(
        &mut self,
        emission: &str,
        insufficient: bool,
        thinking: bool,
    ) -> LibQuestResult<Step> {
        self.record("emission", emission);
        let outcome = turn::interpret(
            &self.evaluator,
            emission,
            self.aliasing,
            insufficient,
            thinking,
        )?;
        let step = match outcome {
            turn::Outcome::Answered(answer) => Step::Answered(answer),
            turn::Outcome::Insufficient => Step::Insufficient,
            turn::Outcome::Repair(envelope) => Step::Repair(envelope),
            turn::Outcome::Think {
                form,
                signature,
                bindings,
            } => self.think(&form, &signature, &bindings)?,
            turn::Outcome::Repl { source } => self.repl(&source)?,
            turn::Outcome::Ask { form, bindings } => Step::Ask { form, bindings },
        };
        match &step {
            Step::Continue(text) => self.record("thought", text),
            Step::Answered(answer) => self.record("answered", &answer.rendered),
            Step::Ask { form, .. } => self.record("ask", form.source()),
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

    /// Render an ask's result back into the conversation.
    ///
    /// The caller ran what the model asked for on its own engine, and
    /// this is how the answer re-enters: the same thought turn a think's
    /// own answer takes, so the model sees one shape either way.
    pub fn resume(&mut self, value: &nu::Value) -> LibQuestResult<String> {
        let text = turn::thought_turn(&output_block(value)?);
        self.record("resumed", &text);
        Ok(text)
    }

    /// Run a think turn and render its answer back as a thought.
    ///
    /// A failure becomes a repair envelope rather than a block, because
    /// in this grammar `<output>` means a VALUE and the model should not
    /// have to tell a result from a report of a non-result.
    fn think(
        &mut self,
        form: &bridge::InferNu,
        signature: &NuSignature,
        bindings: &[turn::Binding],
    ) -> LibQuestResult<Step> {
        match turn::run_think(&self.evaluator, form, signature, bindings) {
            Ok(value) => Ok(Step::Continue(turn::thought_turn(&output_block(&value)?))),
            Err(error) => {
                let mut envelope = Envelope::default();
                envelope.error(
                    "questness::evaluate",
                    Some(Tag::Nu.name()),
                    &error.to_string(),
                );
                Ok(Step::Repair(envelope))
            }
        }
    }

    /// Run a repl expression and render its result back as a thought.
    fn repl(&mut self, source: &str) -> LibQuestResult<Step> {
        match self.evaluator.evaluate(source, None) {
            Ok(value) => {
                let rendered = nu::to_nuon_text(&value)?;
                Ok(Step::Continue(turn::thought_turn(&repl_block(&rendered))))
            }
            Err(error) => {
                let mut envelope = Envelope::default();
                envelope.error(
                    "questness::repl",
                    Some(Tag::Nu.name()),
                    &error.to_string(),
                );
                Ok(Step::Repair(envelope))
            }
        }
    }
}

/// A value as the `<output>` block the model reads it from, escaped so
/// no string's newline can forge a closer.
fn output_block(value: &nu::Value) -> LibQuestResult<Block> {
    let declared = channel::Descriptor::nuon(value.get_type()).render();
    let payload = channel::escape_content(&nu::to_nuon_text(value)?);
    Ok(Block::new(Tag::Output, &declared, &payload))
}

/// A repl result as the `<input>` block the model reads it from.
fn repl_block(rendered: &str) -> Block {
    Block::new(
        Tag::Input,
        &channel::Descriptor::text().render(),
        &channel::escape_content(rendered),
    )
}
