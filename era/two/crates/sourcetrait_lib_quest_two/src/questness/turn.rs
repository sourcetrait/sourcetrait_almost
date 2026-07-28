//! The turn: assemble what the model sees, and read what it emits.
use crate::*;

/// The Thinkspace's own turn, which never crosses the wire. The model
/// asks with a think, the Thinkspace's evaluator answers with a thought,
/// and the model finishes inside the same turn.
/// Routing is off the FORM today, since an evaluate is always a think
/// turn, so nothing yet reads a think wrapper out of an emission. The
/// name stands because the pair is the design.
#[allow(dead_code)]
pub const THINK_ROLE: &str = "think";
pub const THOUGHT_ROLE: &str = "thought";

/// The `<nu>` mode carrying a bare expression rather than a typed def.
pub const REPL_MODE: &str = "repl";

/// A `<pass>` binding and the value travelling on it.
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    /// The channel as nushell spells it: `$in` or `$args`.
    pub pass: String,
    pub value: nu::Value,
}

impl Binding {
    pub fn new(pass: &str, value: nu::Value) -> Self {
        Self {
            pass: pass.trim().to_string(),
            value,
        }
    }
}

/// One turn's inputs, before anything is rendered.
#[derive(Debug, Clone)]
pub struct Request {
    /// The caller's config record, Questness keys included.
    pub config: nu::Value,
    pub prompt: String,
    pub bindings: Vec<Binding>,
}

/// What the model is shown, and what was stripped on the way.
#[derive(Debug, Clone)]
pub struct Assembled {
    pub text: String,
    /// The config as the model sees it, Questness keys removed.
    pub visible: nu::Value,
    pub templated: bool,
}

/// A finished answer: the value, its prose, and any emitted config.
///
/// The config is REPORTED rather than judged here. Whether the model was
/// entitled to send one is the response shape's question, so this layer
/// says what arrived and `questness::shape` decides what it is worth.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Answer {
    pub value: Option<nu::Value>,
    pub declared: Option<nu::Type>,
    pub rendered: String,
    pub config: Option<nu::Value>,
}

/// What the emission asks for next.
#[derive(Debug, Clone)]
pub enum Outcome {
    Answered(Answer),
    /// An `evaluate` form, which is ALWAYS a think turn: the
    /// Thinkspace's own evaluator answers it and nothing leaves.
    Think {
        form: bridge::InferNu,
        signature: NuSignature,
        bindings: Vec<Binding>,
    },
    /// A bare expression whose rendering comes back as text.
    Repl {
        source: String,
    },
    /// Every other form. Questness never runs one - it rides back to the
    /// caller with what it consumes, and the turn pauses there.
    Ask {
        form: bridge::InferNu,
        bindings: Vec<Binding>,
    },
    /// The model said it cannot produce conforming output.
    Insufficient,
    /// The emission did not conform; the envelope is the feedback.
    Repair(Envelope),
}

/// Assemble the text one turn shows the model.
pub fn assemble(request: &Request) -> LibQuestResult<Assembled> {
    let bindings: Vec<(String, nu::Value)> = request
        .bindings
        .iter()
        .map(|binding| (binding.pass.clone(), binding.value.clone()))
        .collect();
    let prepared = questness::config::prepare(
        &request.config,
        &request.prompt,
        &bindings,
    )?;

    let mut blocks: Vec<Block> = Vec::new();
    if !visible_is_empty(&prepared.visible) {
        blocks.push(Block::new(Tag::Config, "", &nuon_payload(&prepared.visible)?));
    }
    for binding in &request.bindings {
        let declared = channel::Descriptor::nuon(binding.value.get_type()).render();
        blocks.push(Block::new(Tag::Input, &declared, &nuon_payload(&binding.value)?));
        blocks.push(Block::new(Tag::Pass, "", &binding.pass));
    }

    let mut text = String::new();
    if !blocks.is_empty() {
        text.push_str(&channel::render_blocks(&blocks));
        text.push('\n');
    }
    text.push_str(&prepared.prompt);

    Ok(Assembled {
        text,
        visible: prepared.visible,
        templated: prepared.templated,
    })
}

/// Read a decoded emission and decide what the turn owes next.
///
/// `insufficient` comes from the caller because the insufficiency tag is
/// special and a skip-special decode strips it, so it is detectable by
/// token id at the engine layer and never by scanning this text.
/// `thinking` says whether this emission may reach a reasoning mode.
pub fn interpret(
    evaluator: &QuestnessEvaluator,
    text: &str,
    aliasing: channel::Aliasing,
    insufficient: bool,
    thinking: bool,
) -> LibQuestResult<Outcome> {
    if insufficient {
        return Ok(Outcome::Insufficient);
    }
    let blocks = match channel::parse_blocks(text, aliasing) {
        Ok(blocks) => blocks,
        Err(error) => {
            let mut envelope = Envelope::default();
            envelope.error("channel::parse", None, &error.to_string());
            return Ok(Outcome::Repair(envelope));
        }
    };
    match blocks.iter().find(|block| block.tag == Tag::Nu) {
        Some(nu_block) => sub_turn(evaluator, &blocks, nu_block, thinking),
        None => answer(&blocks),
    }
}

/// A reasoning mode emitted where reasoning is not permitted.
fn denied(mode: &str) -> Outcome {
    let mut envelope = Envelope::default();
    envelope.error(
        "channel::not_thinking",
        Some(Tag::Nu.name()),
        &format!("`{mode}` is a reasoning mode and runs only in a think turn"),
    );
    Outcome::Repair(envelope)
}

/// Run an `evaluate` sub-turn here and hand its value back.
///
/// A `<nu>` body only DECLARES its def, so the call is appended: the
/// mode is the def's name, `$args` renders as a NUON literal at the call
/// site, and `$in` rides the pipeline.
pub fn run_think(
    evaluator: &QuestnessEvaluator,
    form: &bridge::InferNu,
    signature: &NuSignature,
    bindings: &[Binding],
) -> LibQuestResult<nu::Value> {
    snafu::ensure_whatever!(
        form.is_think(),
        "`{}` is not a think turn; only evaluate runs here",
        form.mode()
    );
    let mut call = String::new();
    if signature.takes_pipeline() {
        call.push_str("$in | ");
    }
    call.push_str(form.mode());
    if signature.args.is_some() {
        let Some(args) = binding_of(bindings, "$args") else {
            snafu::whatever!(
                "`{}` declares an args positional that nothing bound",
                form.mode()
            );
        };
        call.push(' ');
        call.push_str(&nu::to_nuon_text(args)?);
    }
    let pipeline = if signature.takes_pipeline() {
        binding_of(bindings, "$in").cloned()
    } else {
        None
    };
    evaluator.evaluate(&format!("{}\n{call}", form.source()), pipeline)
}

/// The thought turn a think's answer comes back on.
///
/// Questness-internal by construction: this never crosses the wire, so
/// nothing outside the conversation with the model ever sees a role.
pub fn thought_turn(block: &Block) -> String {
    format!(
        "<|im_start|>{THOUGHT_ROLE}\n{}\n<|im_end|>\n",
        channel::render_block(block)
    )
}

/// The value a channel carries, if anything bound it.
fn binding_of<'a>(bindings: &'a [Binding], pass: &str) -> Option<&'a nu::Value> {
    bindings
        .iter()
        .find(|binding| binding.pass == pass)
        .map(|binding| &binding.value)
}

/// The sub-turn path: read the contract, check it, bind the channels.
fn sub_turn(
    evaluator: &QuestnessEvaluator,
    blocks: &[Block],
    nu_block: &Block,
    thinking: bool,
) -> LibQuestResult<Outcome> {
    // `repl` is read before the parser sees anything, because it carries
    // a bare expression rather than a def and there is no signature to
    // diff out of the declaration set.
    if nu_block.header.trim() == REPL_MODE {
        return Ok(if thinking {
            Outcome::Repl {
                source: nu_block.content.clone(),
            }
        } else {
            denied(REPL_MODE)
        });
    }
    let source = nu_source(nu_block);
    let signature = match evaluator.signature_of(&source) {
        Ok(signature) => signature,
        Err(error) => {
            let mut envelope = Envelope::default();
            envelope.error("channel::nu_parse", Some(Tag::Nu.name()), &error.to_string());
            return Ok(Outcome::Repair(envelope));
        }
    };
    let mut envelope = check_agreements(blocks, &signature);
    let bindings = decode_bindings(blocks, &mut envelope);
    if !envelope.is_clean() {
        return Ok(Outcome::Repair(envelope));
    }
    let form = match questness::contract::form_of(&signature, &source) {
        Ok(form) => form,
        Err(error) => {
            envelope.error("channel::nu_form", Some(Tag::Nu.name()), &error.to_string());
            return Ok(Outcome::Repair(envelope));
        }
    };
    if form.is_think() && !thinking {
        return Ok(denied(form.mode()));
    }
    Ok(if form.is_think() {
        Outcome::Think {
            form,
            signature,
            bindings,
        }
    } else {
        Outcome::Ask { form, bindings }
    })
}

/// The answer path: the typed value, then the prose that renders it.
fn answer(blocks: &[Block]) -> LibQuestResult<Outcome> {
    let mut envelope = Envelope::default();
    let output = blocks.iter().find(|block| block.tag == Tag::Output);
    let mut declared = None;
    let mut value = None;

    if let Some(block) = output
        && !block.header.is_empty()
    {
        match typed_payload(&block.header, &block.content) {
            Ok((parsed_type, parsed_value)) => {
                declared = Some(parsed_type);
                value = Some(parsed_value);
            }
            Err(diagnostic) => envelope.errors.push(diagnostic),
        }
    }

    let rendered = match blocks.iter().find(|block| block.tag == Tag::Liquid) {
        Some(block) => {
            let bindings = liquid_bindings(blocks, value.as_ref());
            match template::render(&block.content, &bindings) {
                Ok(text) => text,
                Err(error) => {
                    envelope.error(
                        "channel::liquid_render",
                        Some(Tag::Liquid.name()),
                        &error.to_string(),
                    );
                    String::new()
                }
            }
        }
        None => output.map(|block| block.content.clone()).unwrap_or_default(),
    };

    let config = emitted_config(blocks, &mut envelope);

    if !envelope.is_clean() {
        return Ok(Outcome::Repair(envelope));
    }
    Ok(Outcome::Answered(Answer {
        value,
        declared,
        rendered,
        config,
    }))
}

/// The config record the emission carried, if it carried one.
///
/// A `<config>` travels outbound as the caller's curation, so one coming
/// BACK is the model addressing the harness. Reading it here is what
/// gives the shape something to police; a block left unparsed would be
/// indistinguishable from one never sent.
fn emitted_config(blocks: &[Block], envelope: &mut Envelope) -> Option<nu::Value> {
    let block = blocks.iter().find(|block| block.tag == Tag::Config)?;
    let text = channel::unescape_content(&block.content);
    match nu::from_nuon_text(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            envelope.error("channel::nuon", Some(Tag::Config.name()), &error.to_string());
            None
        }
    }
}

/// Read a block's descriptor and content into a value it can carry.
fn typed_payload(
    header: &str,
    content: &str,
) -> Result<(nu::Type, nu::Value), channel::Diagnostic> {
    let fault = |kind: &str, error: LibQuestError| {
        channel::Diagnostic::new(kind, Some(Tag::Output.name()), &error.to_string())
    };
    let span = nu::Span::unknown();
    let descriptor = channel::Descriptor::parse(header)
        .map_err(|error| fault("channel::descriptor", error))?;
    let text = channel::unescape_content(content);
    match descriptor.declared {
        channel::Declared::Conforms(declared) => {
            let value = nu::from_nuon_text(&text)
                .map_err(|error| fault("channel::nuon", error))?;
            nu::conform(&value, &declared)
                .map_err(|error| fault("channel::conformance", error))?;
            Ok((declared, value))
        }
        // A typedef is carried as a string, so what is checked is that
        // the content parses AS a type - its own contract - rather than
        // that some value conforms to it.
        channel::Declared::Typedef => {
            let typedef = text.trim().to_string();
            nu::parse_typedef(&typedef)
                .map_err(|error| fault("channel::typedef", error))?;
            Ok((nu::Type::String, nu::Value::string(typedef, span)))
        }
        channel::Declared::Untyped => {
            Ok((nu::Type::String, nu::Value::string(text, span)))
        }
    }
}

/// Decode every bound block's NUON into the value its channel carries.
fn decode_bindings(blocks: &[Block], envelope: &mut Envelope) -> Vec<Binding> {
    let mut bindings = Vec::new();
    for (pass, block) in bound_blocks(blocks) {
        let Some(block) = block else { continue };
        let text = channel::unescape_content(&block.content);
        match nu::from_nuon_text(&text) {
            Ok(value) => bindings.push(Binding::new(&pass, value)),
            Err(error) => envelope.error("channel::nuon", Some(&pass), &error.to_string()),
        }
    }
    bindings
}

/// What a `<liquid>` block addresses: its own binding, or the output.
fn liquid_bindings(blocks: &[Block], value: Option<&nu::Value>) -> Vec<(String, nu::Value)> {
    let mut bindings: Vec<(String, nu::Value)> = Vec::new();
    for (pass, block) in bound_blocks(blocks) {
        let Some(block) = block else { continue };
        if block.tag != Tag::Output {
            continue;
        }
        if let Some(value) = value {
            bindings.push((pass, value.clone()));
        }
    }
    if bindings.is_empty()
        && let Some(value) = value
    {
        bindings.push((String::from("$in"), value.clone()));
    }
    bindings
}

/// A `<nu>` block's source: its header carries the def's opening line.
fn nu_source(block: &Block) -> String {
    if block.header.is_empty() {
        return block.content.clone();
    }
    if block.content.is_empty() {
        return block.header.clone();
    }
    format!("{}\n{}", block.header, block.content)
}

/// A NUON payload, escaped so no string's newline can forge a closer.
fn nuon_payload(value: &nu::Value) -> LibQuestResult<String> {
    Ok(channel::escape_content(&nu::to_nuon_text(value)?))
}

/// Whether the visible config carries nothing worth sending.
fn visible_is_empty(visible: &nu::Value) -> bool {
    match visible {
        nu::Value::Record { val, .. } => val.is_empty(),
        _ => false,
    }
}

/// The `<pass>` pairing, without re-reporting what the caller already
/// collected.
///
/// The pairing rule lives in `contract`, which is the module that owns
/// the agreements, so this reads it there rather than restating an
/// index rule that would then drift.
fn bound_blocks(blocks: &[Block]) -> Vec<(String, Option<&Block>)> {
    let mut discard = Envelope::default();
    questness::contract::pass_bindings(blocks, &mut discard)
}
