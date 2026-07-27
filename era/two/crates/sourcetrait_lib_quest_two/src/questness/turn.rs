//! The turn: assemble what the model sees, and read what it emits.
use crate::*;

/// The one `<nu>` mode that round-trips inside Questness.
pub const INSIDE_MODE: &str = "evaluate";

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

/// A finished answer: the value produced, and the prose carrying it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Answer {
    pub value: Option<nu::Value>,
    pub declared: Option<nu::Type>,
    pub rendered: String,
}

/// Where a `<nu>` mode runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Destination {
    /// `evaluate`, which round-trips here with no client involved.
    Inside,
    /// Every other mode, which leaves for the client harness.
    Client,
}

/// A sub-turn the model asked for, ready to dispatch.
#[derive(Debug, Clone)]
pub struct SubTurn {
    pub contract: NuContract,
    pub destination: Destination,
    pub source: String,
    pub bindings: Vec<Binding>,
}

/// What the emission asks for next.
#[derive(Debug, Clone)]
pub enum Outcome {
    Answered(Answer),
    SubTurn(SubTurn),
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
    let prepared = sourcetrait_quest_core::config::prepare(
        &request.config,
        &request.prompt,
        &bindings,
    )?;

    let mut blocks: Vec<Block> = Vec::new();
    if !visible_is_empty(&prepared.visible) {
        blocks.push(Block::new(Tag::Config, "", &nuon_payload(&prepared.visible)?));
    }
    for binding in &request.bindings {
        let declared = binding.value.get_type().to_string();
        blocks.push(Block::new(Tag::Input, &declared, &nuon_payload(&binding.value)?));
        blocks.push(Block::new(Tag::Pass, "", &binding.pass));
    }

    let mut text = String::new();
    if !blocks.is_empty() {
        text.push_str(&sourcetrait_quest_core::channel::render_blocks(&blocks));
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
pub fn interpret(
    evaluator: &QuestnessEvaluator,
    text: &str,
    aliasing: sourcetrait_quest_core::channel::Aliasing,
    insufficient: bool,
) -> LibQuestResult<Outcome> {
    if insufficient {
        return Ok(Outcome::Insufficient);
    }
    let blocks = match sourcetrait_quest_core::channel::parse_blocks(text, aliasing) {
        Ok(blocks) => blocks,
        Err(error) => {
            let mut envelope = Envelope::default();
            envelope.error("channel::parse", None, &error.to_string());
            return Ok(Outcome::Repair(envelope));
        }
    };
    match blocks.iter().find(|block| block.tag == Tag::Nu) {
        Some(nu_block) => sub_turn(evaluator, &blocks, nu_block),
        None => answer(&blocks),
    }
}

/// Run an `evaluate` sub-turn here and hand its value back.
///
/// A `<nu>` body only DECLARES its def, so the call is appended: the
/// mode is the def's name, `$args` renders as a NUON literal at the call
/// site, and `$in` rides the pipeline.
pub fn run_inside(
    evaluator: &QuestnessEvaluator,
    sub: &SubTurn,
) -> LibQuestResult<nu::Value> {
    if sub.destination != Destination::Inside {
        snafu::whatever!(
            "`{}` leaves for the client rather than running here",
            sub.contract.mode
        );
    }
    let mut call = String::new();
    if sub.contract.takes_pipeline() {
        call.push_str("$in | ");
    }
    call.push_str(&sub.contract.mode);
    if sub.contract.args.is_some() {
        let Some(args) = binding(sub, "$args") else {
            snafu::whatever!("`{}` declares an args positional that nothing bound", sub.contract.mode);
        };
        call.push(' ');
        call.push_str(&nu::to_nuon_text(args)?);
    }
    let pipeline = if sub.contract.takes_pipeline() {
        binding(sub, "$in").cloned()
    } else {
        None
    };
    evaluator.evaluate(&format!("{}\n{call}", sub.source), pipeline)
}

/// The value a channel carries into this sub-turn, if anything bound it.
fn binding<'a>(sub: &'a SubTurn, pass: &str) -> Option<&'a nu::Value> {
    sub.bindings
        .iter()
        .find(|binding| binding.pass == pass)
        .map(|binding| &binding.value)
}

/// The sub-turn path: read the contract, check it, bind the channels.
fn sub_turn(
    evaluator: &QuestnessEvaluator,
    blocks: &[Block],
    nu_block: &Block,
) -> LibQuestResult<Outcome> {
    let source = nu_source(nu_block);
    let contract = match evaluator.contract(&source) {
        Ok(contract) => contract,
        Err(error) => {
            let mut envelope = Envelope::default();
            envelope.error("channel::nu_parse", Some(Tag::Nu.name()), &error.to_string());
            return Ok(Outcome::Repair(envelope));
        }
    };
    let mut envelope = check_agreements(blocks, &contract);
    let bindings = decode_bindings(blocks, &mut envelope);
    if !envelope.is_clean() {
        return Ok(Outcome::Repair(envelope));
    }
    let destination = if contract.mode == INSIDE_MODE {
        Destination::Inside
    } else {
        Destination::Client
    };
    Ok(Outcome::SubTurn(SubTurn {
        contract,
        destination,
        source,
        bindings,
    }))
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
            match sourcetrait_quest_core::template::render(&block.content, &bindings) {
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

    if !envelope.is_clean() {
        return Ok(Outcome::Repair(envelope));
    }
    Ok(Outcome::Answered(Answer {
        value,
        declared,
        rendered,
    }))
}

/// Parse a block's declared type and its NUON, then conform one to the
/// other.
fn typed_payload(
    header: &str,
    content: &str,
) -> Result<(nu::Type, nu::Value), sourcetrait_quest_core::channel::Diagnostic> {
    let declared = nu::parse_typedef(header).map_err(|error| {
        sourcetrait_quest_core::channel::Diagnostic::new(
            "channel::typedef",
            Some(Tag::Output.name()),
            &error.to_string(),
        )
    })?;
    let text = sourcetrait_quest_core::channel::unescape_content(content);
    let value = nu::from_nuon_text(&text).map_err(|error| {
        sourcetrait_quest_core::channel::Diagnostic::new(
            "channel::nuon",
            Some(Tag::Output.name()),
            &error.to_string(),
        )
    })?;
    nu::conform(&value, &declared).map_err(|error| {
        sourcetrait_quest_core::channel::Diagnostic::new(
            "channel::conformance",
            Some(Tag::Output.name()),
            &error.to_string(),
        )
    })?;
    Ok((declared, value))
}

/// Decode every bound block's NUON into the value its channel carries.
fn decode_bindings(blocks: &[Block], envelope: &mut Envelope) -> Vec<Binding> {
    let mut bindings = Vec::new();
    for (pass, block) in bound_blocks(blocks) {
        let Some(block) = block else { continue };
        let text = sourcetrait_quest_core::channel::unescape_content(&block.content);
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
    Ok(sourcetrait_quest_core::channel::escape_content(&nu::to_nuon_text(value)?))
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
