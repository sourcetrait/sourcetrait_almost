//! The `<nu>` form: matching a def to a form, and the agreements around
//! it.
use crate::*;

/// The positional name `<pass>$args</>` binds to.
pub const ARGS_POSITIONAL: &str = "args";

/// What a def's signature says, before it is matched to a form.
///
/// Internal to Questness. The form a body turns out to be is an
/// `InferNu`; this is only what the parser read on the way there.
#[derive(Debug, Clone)]
pub struct NuSignature {
    /// The def's own head, which names the form it claims to be.
    pub head: String,
    /// The infix input type; `nothing` means no pipeline input.
    pub input: nu::Type,
    /// Read by whoever runs the form. A think turn is checked by the
    /// evaluator; an ask is the caller's, and the caller re-reads the
    /// signature off the source it was handed.
    #[allow(dead_code)]
    pub output: nu::Type,
    /// The `args` positional's type, when the def declares one.
    pub args: Option<nu::Type>,
    /// Whether the def carries `--env`, which IS interact's contract.
    pub env: bool,
}

impl NuSignature {
    /// Whether the def takes pipeline input at all.
    pub fn takes_pipeline(&self) -> bool {
        self.input != nu::Type::Nothing
    }
}

impl questness::evaluate::QuestnessEvaluator {
    /// Read a `<nu>` body's signature back out of the parser.
    ///
    /// The head is discovered by diffing the declaration set rather than
    /// by scanning the source, so a def spelled across lines or carrying
    /// flags is found exactly as a plain one is.
    pub fn signature_of(&self, source: &str) -> LibQuestResult<NuSignature> {
        let before = self.decl_names();
        let (head, signature, env) = self.parse_fresh_decl(source, &before)?;
        let (input, output) = match signature.input_output_types.first() {
            Some(pair) => pair.clone(),
            None => (nu::Type::Nothing, nu::Type::Any),
        };
        let args = signature
            .required_positional
            .iter()
            .find(|positional| positional.name == ARGS_POSITIONAL)
            .map(|positional| positional.shape.to_type());
        Ok(NuSignature {
            head,
            input,
            output,
            args,
            env,
        })
    }

    /// Parse a `<nu>` body, check it, and match it to a form.
    ///
    /// Three questions, each with its own answer: does it parse, is the
    /// form one we accept, and does its signature match that form's
    /// prototype.
    ///
    /// Reached by its locks until the turn is collapsed onto it, which
    /// is what retires `Destination` and the client-harness seam.
    /// The turn holds a signature already and calls `form_of` directly,
    /// so this is the surface for anyone who does not.
    #[allow(dead_code)]
    pub fn infer_nu(&self, source: &str) -> LibQuestResult<bridge::InferNu> {
        let signature = self.signature_of(source)?;
        form_of(&signature, source)
    }
}

/// Match a read signature to the form it declares.
///
/// Separate from `infer_nu` because the turn already holds a signature
/// by the time it gets here: the agreements are checked against it
/// first, and reading it twice would parse the body twice.
pub fn form_of(signature: &NuSignature, source: &str) -> LibQuestResult<bridge::InferNu> {
    let body = source.to_string();
    match signature.head.as_str() {
        "evaluate" => {
            check_prototype(signature, false)?;
            Ok(bridge::InferNu::Evaluate(bridge::InferNuEvaluate(body)))
        }
        "execute" => {
            check_prototype(signature, false)?;
            Ok(bridge::InferNu::Execute(bridge::InferNuExecute(body)))
        }
        "call" => {
            check_prototype(signature, false)?;
            Ok(bridge::InferNu::Call(bridge::InferNuCall(body)))
        }
        "interact" => {
            check_prototype(signature, true)?;
            Ok(bridge::InferNu::Interact(bridge::InferNuInteract(body)))
        }
        other => snafu::whatever!(
            "`{other}` is not a form; a nu block declares evaluate, execute, \
             call or interact"
        ),
    }
}

/// The one prototype fact the design states: `--env` is interact's
/// contract, since env and `cd` changes surviving into the caller is
/// what interact MEANS. A form that carries it and should not, or does
/// not and should, is not that form however it is spelled.
fn check_prototype(signature: &NuSignature, wants_env: bool) -> LibQuestResult<()> {
    match (wants_env, signature.env) {
        (true, false) => snafu::whatever!(
            "`{}` declares no --env, which IS the interact contract",
            signature.head
        ),
        (false, true) => snafu::whatever!(
            "`{}` carries --env, which only interact declares",
            signature.head
        ),
        _ => Ok(()),
    }
}

/// Collect every agreement failure rather than bailing at the first.
pub fn check_agreements(blocks: &[Block], signature: &NuSignature) -> Envelope {
    let mut envelope = Envelope::default();
    let bindings = pass_bindings(blocks, &mut envelope);

    let in_bound = bindings.iter().find(|(binding, _)| binding == "$in");
    let args_bound = bindings.iter().find(|(binding, _)| binding == "$args");

    match (signature.takes_pipeline(), in_bound) {
        (true, None) => envelope.error(
            "channel::missing_pass",
            Some("$in"),
            &format!(
                "`{}` declares a {} input, so a $in binding is required",
                signature.head, signature.input
            ),
        ),
        (false, Some(_)) => envelope.error(
            "channel::unbound_pass",
            Some("$in"),
            &format!("`{}` declares no pipeline input", signature.head),
        ),
        _ => {}
    }

    match (&signature.args, args_bound) {
        (Some(_), None) => envelope.error(
            "channel::missing_pass",
            Some("$args"),
            &format!(
                "`{}` declares an `{ARGS_POSITIONAL}` positional, so an $args \
                 binding is required",
                signature.head
            ),
        ),
        (None, Some(_)) => envelope.error(
            "channel::unbound_pass",
            Some("$args"),
            &format!(
                "`{}` declares no `{ARGS_POSITIONAL}` positional",
                signature.head
            ),
        ),
        _ => {}
    }

    for (binding, bound) in &bindings {
        let channel = match binding.as_str() {
            "$in" => Some(&signature.input),
            "$args" => signature.args.as_ref(),
            _ => None,
        };
        let (Some(channel), Some(bound)) = (channel, bound) else {
            continue;
        };
        let declared = &bound.header;
        if declared.is_empty() {
            continue;
        }
        let parsed = match nu::parse_typedef(declared) {
            Ok(parsed) => parsed,
            Err(error) => {
                envelope.error("channel::typedef", Some(binding), &error.to_string());
                continue;
            }
        };
        if !parsed.is_subtype_of(channel) {
            envelope.error(
                "channel::type_disagreement",
                Some(binding),
                &format!(
                    "the block declares {parsed} where `{}` declares {channel}",
                    signature.head
                ),
            );
        }
    }
    envelope
}

/// Pair each `<pass>` with the block it binds, in emission order.
pub fn pass_bindings<'a>(
    blocks: &'a [Block],
    envelope: &mut Envelope,
) -> Vec<(String, Option<&'a Block>)> {
    let mut bindings: Vec<(String, Option<&'a Block>)> = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        if block.tag != Tag::Pass {
            continue;
        }
        let binding = block.content.trim().to_string();
        if binding != "$in" && binding != "$args" {
            envelope.error(
                "channel::unknown_binding",
                None,
                &format!("a pass binds $in or $args, not {binding:?}"),
            );
            continue;
        }
        if bindings.iter().any(|(known, _)| *known == binding) {
            envelope.error(
                "channel::duplicate_binding",
                Some(&binding),
                &format!("{binding} is bound more than once"),
            );
            continue;
        }
        let bound = index
            .checked_sub(1)
            .and_then(|previous| blocks.get(previous))
            .filter(|previous| previous.tag != Tag::Pass);
        match bound {
            Some(previous) => bindings.push((binding, Some(previous))),
            None => {
                envelope.error(
                    "channel::unbound_pass",
                    Some(&binding),
                    &format!("{binding} follows no block to bind"),
                );
            }
        }
    }
    bindings
}
