//! `quest`: the one verb, with capability composed onto it.
use crate::*;

/// The flag naming a file to read the prompt from.
pub(crate) const FILE_FLAG: &str = "file";

/// The flag carrying the runtime config record.
pub(crate) const CONFIG_FLAG: &str = "config";

/// The piped record's field carrying a prompt inline.
pub(crate) const PROMPT_FIELD: &str = "prompt";

/// The piped record's field naming a file holding one.
pub(crate) const FILE_PROMPT_FIELD: &str = "fprompt";

pub(crate) struct Prompt;

/// What one call resolved to: the prompt, and what it binds.
#[derive(Debug, PartialEq)]
pub(crate) struct Asked {
    pub(crate) prompt: String,
    pub(crate) bindings: Vec<(bridge::InferPass, bridge::InferInput)>,
}

impl nu_plugin::SimplePluginCommand for Prompt {
    type Plugin = QuestPlugin;

    fn name(&self) -> &str {
        "quest"
    }

    fn description(&self) -> &str {
        "Ask the quest model, answering in the shape the config declares."
    }

    fn extra_description(&self) -> &str {
        "The prompt is a Liquid template, given as the positional, through \
         --file, or as a prompt or fprompt field of a piped record. Piped \
         data self-describes: its own derived type is what the model is \
         shown, so there is no type to declare.\n\n\
         What comes back is decided by the config's shape key. Declaring \
         nothing gets prose, which is the model's own trained shape; \
         {shape: {response: [output]}} gets the typed value itself, and \
         several members get a record keyed by member."
    }

    fn search_terms(&self) -> Vec<&str> {
        vec!["ask", "llm", "model", "prompt"]
    }

    fn signature(&self) -> nu_protocol::Signature {
        nu_protocol::Signature::build(nu_plugin::PluginCommand::name(self))
            .input_output_type(nu_protocol::Type::Any, nu_protocol::Type::Any)
            .optional(
                "prompt",
                nu_protocol::SyntaxShape::String,
                "The prompt, as a Liquid template.",
            )
            .named(
                FILE_FLAG,
                nu_protocol::SyntaxShape::Filepath,
                "Read the prompt from a file instead.",
                Some('f'),
            )
            .named(
                CONFIG_FLAG,
                nu_protocol::SyntaxShape::Record(vec![].into()),
                "The runtime config record, carrying shape, env and the rest.",
                Some('c'),
            )
            .category(nu_protocol::Category::Custom(String::from("quest")))
    }

    /// AN INTERRUPT IS ONLY OBSERVABLE BY ASKING FOR IT. Ctrl-C reaches
    /// the engine rather than this process, so a plugin that never
    /// consults the interface never learns one happened - and this call
    /// then sits in a blocking runtime until the daemon answers, with no
    /// way to abandon a turn from the terminal.
    fn run(
        &self,
        plugin: &QuestPlugin,
        engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: &nu_protocol::Value,
    ) -> Result<nu_protocol::Value, nu_protocol::LabeledError> {
        let interrupted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let raised = std::sync::Arc::clone(&interrupted);
        // The guard is RAII: held for the call, unregistered on return.
        let _guard = engine.register_signal_handler(Box::new(move |action| {
            if matches!(action, nu_protocol::SignalAction::Interrupt) {
                raised.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }))?;
        Ok(answer(plugin, call, input, &interrupted)?)
    }
}

/// One call, from its arguments to the value it returns.
fn answer(
    plugin: &QuestPlugin,
    call: &nu_plugin::EvaluatedCall,
    input: &nu_protocol::Value,
    interrupted: &std::sync::atomic::AtomicBool,
) -> QuestPluginResult<nu_protocol::Value> {
    let config = call
        .get_flag_value(CONFIG_FLAG)
        .unwrap_or_else(|| nu_protocol::Value::record(lib::nu::Record::new(), call.head));
    // Read BEFORE the turn runs, so a malformed shape is refused without
    // having spent a generation on it.
    let shape = lib::Shape::of(&config)?;
    let asked = asked(call, input)?;
    let request = bridge::InferRequest {
        config: Some(bridge::InferConfig(bridge::InferValue(config))),
        inputs: asked.bindings,
        text: Some(bridge::InferText(asked.prompt)),
        output: None,
    };
    shaped(
        &shape,
        &converse::ask(plugin, request, interrupted)?,
        call.head,
    )
}

/// The value a caller gets back, decided by the shape it declared.
///
/// One declared member comes back BARE so a pipeline works; several come
/// back as a record keyed by member spelling, in declaration order. A
/// single-key record would force a `get` at every call site.
fn shaped(
    shape: &lib::Shape,
    response: &bridge::InferResponse,
    span: lib::nu::Span,
) -> QuestPluginResult<nu_protocol::Value> {
    let mut carried: Vec<(&'static str, nu_protocol::Value)> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for member in &shape.response {
        let value = match member {
            lib::ShapeMember::Text => Some(nu_protocol::Value::string(
                response
                    .text
                    .as_ref()
                    .map(|text| text.0.clone())
                    .unwrap_or_default(),
                span,
            )),
            lib::ShapeMember::Output => response.output.as_ref().map(|output| output.value().0.clone()),
            lib::ShapeMember::Config => response.config.as_ref().map(|config| config.0.0.clone()),
        };
        match value {
            Some(value) => carried.push((member.spelling(), value)),
            // Collected rather than bailed on, so one answer reports
            // every member it owes rather than the first.
            None => missing.push(format!(
                "shape::missing at {member}: the response shape declares {member}, \
                 so the answer owes one",
                member = member.spelling()
            )),
        }
    }
    if !missing.is_empty() {
        return Err(QuestPluginError::rows(missing));
    }
    match carried.len() {
        1 => Ok(carried.remove(0).1),
        _ => {
            let mut record = lib::nu::Record::new();
            for (member, value) in carried {
                record.push(member.to_string(), value);
            }
            Ok(nu_protocol::Value::record(record, span))
        }
    }
}

/// The prompt this call asks, and the binding that rides with it.
///
/// Four sources, and exactly one of them may answer. Silently preferring
/// one over another would make two spellings of the same call mean
/// different things depending on which the caller remembered.
pub(crate) fn asked(
    call: &nu_plugin::EvaluatedCall,
    input: &nu_protocol::Value,
) -> QuestPluginResult<Asked> {
    let piped = piped_record(input);
    let mut found: Vec<(&str, String)> = Vec::new();
    if let Some(text) = call.opt::<String>(0).ok().flatten() {
        found.push(("the positional", text));
    }
    if let Some(path) = call.get_flag::<String>(FILE_FLAG).ok().flatten() {
        found.push((FILE_FLAG, fs::read_to_string(path)?));
    }
    if let Some(text) = piped.as_ref().and_then(|record| field(record, PROMPT_FIELD)) {
        found.push((PROMPT_FIELD, text));
    }
    if let Some(path) = piped
        .as_ref()
        .and_then(|record| field(record, FILE_PROMPT_FIELD))
    {
        found.push((FILE_PROMPT_FIELD, fs::read_to_string(path)?));
    }

    let prompt = match found.len() {
        1 => found.remove(0).1,
        0 => snafu::whatever!(
            "a prompt is the positional, --{FILE_FLAG}, or a {PROMPT_FIELD} or \
             {FILE_PROMPT_FIELD} field of the piped record"
        ),
        _ => snafu::whatever!(
            "a prompt comes from one source; got {}",
            found
                .iter()
                .map(|(origin, _)| *origin)
                .collect::<Vec<&str>>()
                .join(", ")
        ),
    };
    Ok(Asked {
        prompt,
        bindings: bindings(input, piped),
    })
}

/// The piped value as a record, where it is one.
fn piped_record(input: &nu_protocol::Value) -> Option<lib::nu::Record> {
    match input {
        nu_protocol::Value::Record { val, .. } => Some(val.as_ref().clone()),
        _ => None,
    }
}

/// A record field's text, where it carries one.
fn field(record: &lib::nu::Record, key: &str) -> Option<String> {
    match record.get(key)? {
        nu_protocol::Value::String { val, .. } => Some(val.clone()),
        _ => None,
    }
}

/// What the piped value binds, once the prompt fields are consumed.
///
/// The prompt-bearing fields are STRIPPED rather than left in place. A
/// caller piping `{prompt: ..., rows: ...}` said the first as an
/// instruction and the second as data, and leaving the instruction inside
/// the data shows the model its own prompt twice.
fn bindings(
    input: &nu_protocol::Value,
    piped: Option<lib::nu::Record>,
) -> Vec<(bridge::InferPass, bridge::InferInput)> {
    let span = input.span();
    let bound = match piped {
        Some(mut record) => {
            record
                .retain(|key, _| key != PROMPT_FIELD && key != FILE_PROMPT_FIELD);
            if record.is_empty() {
                return Vec::new();
            }
            nu_protocol::Value::record(record, span)
        }
        None if matches!(input, nu_protocol::Value::Nothing { .. }) => return Vec::new(),
        None => input.clone(),
    };
    vec![(
        bridge::InferPass::In,
        bridge::InferInput::Nuon(bridge::InferNuonInput(bridge::InferValue(bound))),
    )]
}
