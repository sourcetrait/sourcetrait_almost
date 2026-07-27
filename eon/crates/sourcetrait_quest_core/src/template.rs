//! Liquid rendering, and the nu value bridge that feeds it.
use crate::*;

/// The channel name a `<pass>` binding carries into a template.
///
/// The sigil is nushell's, and Liquid's grammar rejects it, so the
/// channel keeps its name and loses the `$`.
pub fn binding_name(pass: &str) -> &str {
    pass.trim().strip_prefix('$').unwrap_or(pass.trim())
}

/// Render a template against its bound channels.
pub fn render(source: &str, bindings: &[(String, nu::Value)]) -> QuestCoreResult<String> {
    let parser = match liquid::ParserBuilder::with_stdlib().build() {
        Ok(parser) => parser,
        Err(error) => snafu::whatever!("the liquid parser did not build: {error}"),
    };
    let template = match parser.parse(source) {
        Ok(template) => template,
        Err(error) => snafu::whatever!("template does not parse: {error}"),
    };
    let mut globals = liquid::Object::new();
    for (pass, value) in bindings {
        let key = liquid::model::KString::from_string(binding_name(pass).to_string());
        globals.insert(key, to_liquid(value)?);
    }
    match template.render(&globals) {
        Ok(text) => Ok(text),
        Err(error) => snafu::whatever!("template does not render: {error}"),
    }
}

/// A nu value as liquid data; anything exotic rides its NUON form.
pub fn to_liquid(value: &nu::Value) -> QuestCoreResult<liquid::model::Value> {
    Ok(match value {
        nu::Value::Nothing { .. } => liquid::model::Value::Nil,
        nu::Value::Bool { val, .. } => liquid::model::Value::scalar(*val),
        nu::Value::Int { val, .. } => liquid::model::Value::scalar(*val),
        nu::Value::Float { val, .. } => liquid::model::Value::scalar(*val),
        nu::Value::String { val, .. } => liquid::model::Value::scalar(val.clone()),
        nu::Value::List { vals, .. } => {
            let mut out = Vec::with_capacity(vals.len());
            for item in vals {
                out.push(to_liquid(item)?);
            }
            liquid::model::Value::Array(out)
        }
        nu::Value::Record { val, .. } => {
            let mut object = liquid::Object::new();
            for (key, item) in val.iter() {
                let key = liquid::model::KString::from_string(key.clone());
                object.insert(key, to_liquid(item)?);
            }
            liquid::model::Value::Object(object)
        }
        other => liquid::model::Value::scalar(nu::to_nuon_text(other)?),
    })
}
