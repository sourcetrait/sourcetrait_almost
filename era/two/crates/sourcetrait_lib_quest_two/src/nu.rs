//! The NUON data-format core: the nu type and value surface, and its IO.
use crate::*;

pub use nu_protocol::{
    CompareTypes,
    Record,
    ShellError,
    Span,
    Type,
    Value,
    record,
};

/// Parse a nu typedef string into a Type, oneof included.
pub fn parse_typedef(typedef: &str) -> LibQuestResult<Type> {
    let mut engine_state = r::nu::EngineState::new();
    engine_state.add_env_var(String::from("PWD"), Value::string("", Span::unknown()));
    let mut working_set = r::nu::StateWorkingSet::new(&engine_state);
    let source = format!("{{|x: {typedef}| null}}");
    let block = r::nu::parse(&mut working_set, None, source.as_bytes(), false);
    if let Some(error) = working_set.parse_errors.first() {
        snafu::whatever!("typedef {typedef:?} does not parse: {error}");
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
        snafu::whatever!("typedef {typedef:?} did not parse as a signature annotation");
    };
    let signature = &working_set.get_block(closure_id).signature;
    let Some(positional) = signature.required_positional.first() else {
        snafu::whatever!("typedef {typedef:?} did not bind a typed positional");
    };
    Ok(positional.shape.to_type())
}

/// NUON text -> Value (the nuon crate's own parser; no engine).
pub fn from_nuon_text(text: &str) -> LibQuestResult<Value> {
    match nuon::from_nuon(text, None) {
        Ok(value) => Ok(value),
        Err(error) => snafu::whatever!("nuon parse failed: {error}"),
    }
}

/// Value -> NUON text, compact and single-line.
pub fn to_nuon_text(value: &Value) -> LibQuestResult<String> {
    render_nuon(value, nuon::ToStyle::Default)
}

/// Value -> CONDENSED NUON, what `to nuon --raw` emits.
pub fn to_nuon_condensed(value: &Value) -> LibQuestResult<String> {
    render_nuon(value, nuon::ToStyle::Raw)
}

/// Value -> PRETTY NUON, two-space indented as `to nuon --pretty`.
pub fn to_nuon_pretty(value: &Value) -> LibQuestResult<String> {
    render_nuon(value, nuon::ToStyle::Spaces(2))
}

fn render_nuon(value: &Value, style: nuon::ToStyle) -> LibQuestResult<String> {
    let engine_state = r::nu::EngineState::new();
    let config = nuon::ToNuonConfig::default().style(style);
    match nuon::to_nuon(&engine_state, value, config) {
        Ok(text) => Ok(text),
        Err(error) => snafu::whatever!("nuon render failed: {error}"),
    }
}

/// Deep value-vs-typedef conformance via nu's own subtype machinery.
pub fn conform(value: &Value, declared: &Type) -> LibQuestResult<()> {
    snafu::ensure_whatever!(
        value.is_subtype_of(declared),
        "value does not conform: declared {declared}, derived {}",
        value.get_type()
    );
    Ok(())
}

/// Read one whole-value .nuon file (a dataset: one table/record/value
/// per file).
pub fn load_value(path: &Path) -> LibQuestResult<Value> {
    from_nuon_text(&fs::read_to_string(path)?)
}

/// Write one whole-value .nuon file, newline-terminated.
pub fn save_value(path: &Path, value: &Value) -> LibQuestResult<()> {
    let text = to_nuon_text(value)?;
    ensure_parent(path)?;
    Ok(fs::write(path, text + "\n")?)
}

/// Read a NUON-LINES file (one record literal per line); blank lines
/// skip.
pub fn load_lines(path: &Path) -> LibQuestResult<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(from_nuon_text)
        .collect()
}

/// Write a NUON-LINES file whole, one asserted single-line value each.
pub fn save_lines(path: &Path, values: &[Value]) -> LibQuestResult<()> {
    let mut out = String::new();
    for value in values {
        out.push_str(&render_line(value)?);
        out.push('\n');
    }
    ensure_parent(path)?;
    Ok(fs::write(path, out)?)
}

/// Append one value to a NUON-LINES file as a single line.
pub fn append_line(path: &Path, value: &Value) -> LibQuestResult<()> {
    let line = render_line(value)?;
    ensure_parent(path)?;
    let mut file = fs::OpenOptions::new().create(true).append(true).open(path)?;
    Ok(io::Write::write_all(&mut file, format!("{line}\n").as_bytes())?)
}

/// One value as one line, guarded against a multi-line rendering.
fn render_line(value: &Value) -> LibQuestResult<String> {
    let text = to_nuon_text(value)?;
    snafu::ensure_whatever!(
        !text.contains('\n'),
        "nuon-lines rows must render single-line; got a multi-line rendering"
    );
    Ok(text)
}

fn ensure_parent(path: &Path) -> LibQuestResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}
