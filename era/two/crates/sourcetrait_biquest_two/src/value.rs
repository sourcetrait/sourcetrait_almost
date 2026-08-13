//! The lossless JSON and nu-value bridge, and the record helpers.
use crate::*;

pub(crate) fn span() -> harness::nu::Span {
    harness::nu::Span::unknown()
}

pub(crate) fn v_int(value: i64) -> harness::nu::Value {
    harness::nu::Value::int(value, span())
}

pub(crate) fn v_float(value: f64) -> harness::nu::Value {
    harness::nu::Value::float(value, span())
}

pub(crate) fn v_str(value: &str) -> harness::nu::Value {
    harness::nu::Value::string(value, span())
}

#[allow(dead_code)]
pub(crate) fn v_bool(value: bool) -> harness::nu::Value {
    harness::nu::Value::bool(value, span())
}

pub(crate) fn field<'a>(
    record: &'a harness::nu::Record,
    name: &str,
) -> BiquestResult<&'a harness::nu::Value> {
    match record.get(name) {
        Some(value) => Ok(value),
        None => snafu::whatever!("missing field {name}"),
    }
}

pub(crate) fn field_str(record: &harness::nu::Record, name: &str) -> BiquestResult<String> {
    match field(record, name)?.as_str() {
        Ok(text) => Ok(text.to_string()),
        Err(e) => snafu::whatever!("field {name}: {e}"),
    }
}

#[allow(dead_code)]
pub(crate) fn field_int(record: &harness::nu::Record, name: &str) -> BiquestResult<i64> {
    match field(record, name)?.as_int() {
        Ok(value) => Ok(value),
        Err(e) => snafu::whatever!("field {name}: {e}"),
    }
}

/// nu Value to JSON, the exact inverse of json_to_value.
#[allow(dead_code)]
pub(crate) fn value_to_json(value: &harness::nu::Value) -> BiquestResult<serde_json::Value> {
    Ok(match value {
        harness::nu::Value::Nothing { .. } => serde_json::Value::Null,
        harness::nu::Value::Bool { val, .. } => serde_json::Value::Bool(*val),
        harness::nu::Value::Int { val, .. } => serde_json::Value::Number((*val).into()),
        harness::nu::Value::Float { val, .. } => match serde_json::Number::from_f64(*val) {
            Some(number) => serde_json::Value::Number(number),
            None => snafu::whatever!("non-finite float cannot render to JSON"),
        },
        harness::nu::Value::String { val, .. } => serde_json::Value::String(val.clone()),
        harness::nu::Value::List { vals, .. } => {
            let mut items = Vec::with_capacity(vals.len());
            for item in vals {
                items.push(value_to_json(item)?);
            }
            serde_json::Value::Array(items)
        }
        harness::nu::Value::Record { val, .. } => {
            let mut map = serde_json::Map::new();
            for (key, item) in val.iter() {
                map.insert(key.clone(), value_to_json(item)?);
            }
            serde_json::Value::Object(map)
        }
        other => snafu::whatever!("unsupported value type for JSON: {}", other.get_type()),
    })
}

/// JSON to nu Value, lossless field for field.
#[allow(dead_code)]
pub(crate) fn json_to_value(json: &serde_json::Value) -> BiquestResult<harness::nu::Value> {
    Ok(match json {
        serde_json::Value::Null => harness::nu::Value::nothing(span()),
        serde_json::Value::Bool(flag) => v_bool(*flag),
        serde_json::Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                v_int(int)
            } else if number.is_u64() {
                snafu::whatever!("integer {number} exceeds i64; lossless conversion refused")
            } else if let Some(float) = number.as_f64() {
                v_float(float)
            } else {
                snafu::whatever!("unrepresentable JSON number {number}")
            }
        }
        serde_json::Value::String(text) => v_str(text),
        serde_json::Value::Array(items) => {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(json_to_value(item)?);
            }
            harness::nu::Value::list(values, span())
        }
        serde_json::Value::Object(map) => {
            let mut record = harness::nu::Record::new();
            for (key, value) in map {
                record.push(key.clone(), json_to_value(value)?);
            }
            harness::nu::Value::record(record, span())
        }
    })
}

pub(crate) fn epoch_seconds() -> i64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(elapsed) => elapsed.as_secs() as i64,
        Err(_) => 0,
    }
}
