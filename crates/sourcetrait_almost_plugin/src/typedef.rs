use crate::*;

/// Parse a typedef string (nu 0.114's native Type grammar, unions
/// included) with the real parser. The public surface has no bare
/// type-parse entry (parse_shape_specs is crate-private upstream), so
/// the typedef rides a synthetic closure signature - `{|x: <t>| null}`
/// parses decl-free on a bare engine state and its positional carries
/// the shape.
pub fn parse_typedef(typedef: &str) -> PluginResult<Type> {
    let mut engine_state = r::nu::EngineState::new();
    // The parser expects $env.PWD on the engine state (the nuon crate
    // performs the same seeding for from_nuon).
    engine_state.add_env_var(
        String::from("PWD"),
        Value::string("", nu_protocol::Span::unknown()),
    );
    let mut working_set = r::nu::StateWorkingSet::new(&engine_state);
    let source = format!("{{|x: {typedef}| null}}");
    let block = r::nu::parse(&mut working_set, None, source.as_bytes(), false);
    if let Some(error) = working_set.parse_errors.first() {
        return TypedefSnafu { message: error.to_string() }.fail();
    }
    let closure_id = block
        .pipelines
        .first()
        .and_then(|pipeline| pipeline.elements.first())
        .and_then(|element| match &element.expr.expr {
            r::nu::Expr::Closure(block_id) => Some(*block_id),
            _ => None,
        });
    let Some(closure_id) = closure_id else {
        return TypedefSnafu {
            message: String::from("typedef did not parse as a signature annotation"),
        }
        .fail();
    };
    let signature = &working_set.get_block(closure_id).signature;
    let Some(positional) = signature.required_positional.first() else {
        return TypedefSnafu {
            message: String::from("typedef did not bind a typed positional"),
        }
        .fail();
    };
    Ok(positional.shape.to_type())
}

/// The wire rendering of a Type (Display is the typedef form).
pub fn render_typedef(ty: &Type) -> String {
    ty.to_string()
}

/// Deep type of a value (records recurse; a uniform list of records
/// derives as table<...>) - the input-def derivation for piped data.
pub fn derive_type(value: &Value) -> Type {
    value.get_type()
}

/// Value-informed deep conformance via nu's own subtype machinery:
/// oneof accepts any member, empty lists bind any list/table, records
/// are OPEN (extra fields pass - the native-trust posture).
pub fn conforms(value: &Value, ty: &Type) -> bool {
    value.is_subtype_of(ty)
}
