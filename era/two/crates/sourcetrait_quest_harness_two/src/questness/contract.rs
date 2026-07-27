//! The def's contract, and the agreements checkable before it runs.
use crate::*;

/// The positional name `<pass>$args</>` binds to.
pub const ARGS_POSITIONAL: &str = "args";

/// What a `<nu>` block's def declares, read back from its signature.
#[derive(Debug, Clone)]
pub struct NuContract {
    /// The def's name, which IS the mode.
    pub mode: String,
    /// The infix input type; `nothing` means no pipeline input.
    pub input: nu::Type,
    pub output: nu::Type,
    /// The `args` positional's type, when the def declares one.
    pub args: Option<nu::Type>,
}

impl NuContract {
    /// Whether the def takes pipeline input at all.
    pub fn takes_pipeline(&self) -> bool {
        self.input != nu::Type::Nothing
    }
}

impl questness::evaluate::QuestnessEvaluator {
    /// Parse a `<nu>` body and read its def's contract back out.
    ///
    /// The mode is discovered by diffing the declaration set rather
    /// than by scanning the source, so a def named anything at all is
    /// found and the answer comes from the parser instead of a regex.
    pub fn contract(&self, source: &str) -> QuestHarnessResult<NuContract> {
        let before = self.decl_names();
        let (fresh, signature) = self.parse_fresh_decl(source, &before)?;
        let (input, output) = match signature.input_output_types.first() {
            Some(pair) => pair.clone(),
            None => (nu::Type::Nothing, nu::Type::Any),
        };
        let args = signature
            .required_positional
            .iter()
            .find(|positional| positional.name == ARGS_POSITIONAL)
            .map(|positional| positional.shape.to_type());
        Ok(NuContract {
            mode: fresh,
            input,
            output,
            args,
        })
    }
}

/// Collect every agreement failure rather than bailing at the first.
pub fn check_agreements(blocks: &[Block], contract: &NuContract) -> Envelope {
    let mut envelope = Envelope::default();
    let bindings = pass_bindings(blocks, &mut envelope);

    let in_bound = bindings.iter().find(|(binding, _)| binding == "$in");
    let args_bound = bindings.iter().find(|(binding, _)| binding == "$args");

    match (contract.takes_pipeline(), in_bound) {
        (true, None) => envelope.error(
            "channel::missing_pass",
            Some("$in"),
            &format!(
                "`{}` declares a {} input, so a $in binding is required",
                contract.mode, contract.input
            ),
        ),
        (false, Some(_)) => envelope.error(
            "channel::unbound_pass",
            Some("$in"),
            &format!("`{}` declares no pipeline input", contract.mode),
        ),
        _ => {}
    }

    match (&contract.args, args_bound) {
        (Some(_), None) => envelope.error(
            "channel::missing_pass",
            Some("$args"),
            &format!(
                "`{}` declares an `{ARGS_POSITIONAL}` positional, so an $args \
                 binding is required",
                contract.mode
            ),
        ),
        (None, Some(_)) => envelope.error(
            "channel::unbound_pass",
            Some("$args"),
            &format!("`{}` declares no `{ARGS_POSITIONAL}` positional", contract.mode),
        ),
        _ => {}
    }

    for (binding, bound) in &bindings {
        let channel = match binding.as_str() {
            "$in" => Some(&contract.input),
            "$args" => contract.args.as_ref(),
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
                    contract.mode
                ),
            );
        }
    }
    envelope
}

/// Pair each `<pass>` with the block it binds, in emission order.
pub(crate) fn pass_bindings<'a>(
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
