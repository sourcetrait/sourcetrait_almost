//! The `shape` config convention: what a turn carries, in each direction.
use crate::*;

/// The config key a shape is declared under.
pub const SHAPE_KEY: &str = "shape";

/// The direction naming what the caller sends the model.
pub const REQUEST_KEY: &str = "request";

/// The direction naming what the model is asked to send back.
pub const RESPONSE_KEY: &str = "response";

/// One member of a direction's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeMember {
    /// A typed value, carrying its type and its NUON.
    Output,
    /// Unmarked prose, which is the checkpoint's own protocol.
    Text,
    /// A config record.
    Config,
}

/// The members in declaration order, which is also rendering order.
pub const MEMBERS: [ShapeMember; 3] =
    [ShapeMember::Output, ShapeMember::Text, ShapeMember::Config];

impl ShapeMember {
    /// The spelling a config record carries.
    pub fn spelling(&self) -> &'static str {
        match self {
            Self::Output => "output",
            Self::Text => "text",
            Self::Config => "config",
        }
    }

    /// Read one back from its spelling; anything else is refused.
    pub fn parse(spelling: &str) -> LibQuestResult<Self> {
        match MEMBERS.into_iter().find(|m| m.spelling() == spelling) {
            Some(member) => Ok(member),
            None => snafu::whatever!(
                "a shape member is one of output, text or config; got {spelling:?}"
            ),
        }
    }
}

/// What a turn carries in each direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub request: Vec<ShapeMember>,
    pub response: Vec<ShapeMember>,
}

impl Default for Shape {
    /// Text in, text out: the base model's own shape.
    fn default() -> Self {
        Self {
            request: vec![ShapeMember::Text],
            response: vec![ShapeMember::Text],
        }
    }
}

impl Shape {
    /// The shape a config record declares, per direction.
    ///
    /// Three granularities fall back independently, because a caller
    /// naming one direction has said nothing about the other.
    pub fn of(config: &nu::Value) -> LibQuestResult<Self> {
        let nu::Value::Record { val, .. } = config else {
            snafu::whatever!("a config is a record; got {}", config.get_type());
        };
        let Some(declared) = val.get(SHAPE_KEY) else {
            return Ok(Self::default());
        };
        let nu::Value::Record { val: declared, .. } = declared else {
            snafu::whatever!(
                "{SHAPE_KEY} is a record of {REQUEST_KEY} and {RESPONSE_KEY}; got {}",
                declared.get_type()
            );
        };
        Ok(Self {
            request: direction(declared, REQUEST_KEY)?,
            response: direction(declared, RESPONSE_KEY)?,
        })
    }

    /// Whether a direction names a member.
    pub fn responds_with(&self, member: ShapeMember) -> bool {
        self.response.contains(&member)
    }

    /// The value a shaped answer returns, or why it does not conform.
    ///
    /// A shortfall is an ENVELOPE rather than an error, because a model
    /// answering off the declared shape is the ordinary untrained case
    /// and the repair loop is what that vocabulary is for.
    pub fn value_of(&self, answer: &turn::Answer) -> Result<nu::Value, Envelope> {
        let span = nu::Span::unknown();
        let mut envelope = Envelope::default();
        let mut carried: Vec<(ShapeMember, nu::Value)> = Vec::new();

        for member in &self.response {
            match member {
                ShapeMember::Text => {
                    carried.push((*member, nu::Value::string(answer.rendered.clone(), span)));
                }
                ShapeMember::Output => match &answer.value {
                    Some(value) => carried.push((*member, value.clone())),
                    None => missing(&mut envelope, *member, "a typed value"),
                },
                ShapeMember::Config => match &answer.config {
                    Some(value) => carried.push((*member, value.clone())),
                    None => missing(&mut envelope, *member, "a config record"),
                },
            }
        }

        // The policing half. A config the caller never asked for is
        // DENIED rather than passed on: the shape is the contract, and
        // an undeclared block reaching a caller would make the contract
        // advisory.
        if answer.config.is_some() && !self.responds_with(ShapeMember::Config) {
            envelope.error(
                "shape::undeclared",
                Some(ShapeMember::Config.spelling()),
                "the answer carries a config the response shape does not declare",
            );
        }

        if !envelope.is_clean() {
            return Err(envelope);
        }
        match carried.len() {
            1 => Ok(carried.remove(0).1),
            _ => {
                let mut record = nu::Record::new();
                for (member, value) in carried {
                    record.push(member.spelling().to_string(), value);
                }
                Ok(nu::Value::record(record, span))
            }
        }
    }
}

/// One direction's members, or the trained default where it is absent.
fn direction(declared: &nu::Record, key: &str) -> LibQuestResult<Vec<ShapeMember>> {
    let Some(value) = declared.get(key) else {
        return Ok(vec![ShapeMember::Text]);
    };
    let nu::Value::List { vals, .. } = value else {
        snafu::whatever!("{key} is a list of shape members; got {}", value.get_type());
    };
    let mut members: Vec<ShapeMember> = Vec::with_capacity(vals.len());
    for item in vals {
        let nu::Value::String { val, .. } = item else {
            snafu::whatever!("a shape member is a string; got {}", item.get_type());
        };
        let member = ShapeMember::parse(val)?;
        if members.contains(&member) {
            snafu::whatever!("{key} names {} more than once", member.spelling());
        }
        members.push(member);
    }
    // An EMPTY list is a caller saying "carry nothing", which is not
    // what an absent key says. Substituting the default here would hide
    // the difference, so it is refused instead.
    snafu::ensure_whatever!(
        !members.is_empty(),
        "{key} names no shape members; omit it for the default"
    );
    Ok(members)
}

/// Record that the answer did not carry a member the response declared.
fn missing(envelope: &mut Envelope, member: ShapeMember, wanted: &str) {
    envelope.error(
        "shape::missing",
        Some(member.spelling()),
        &format!(
            "the response shape declares {}, so the answer owes {wanted}",
            member.spelling()
        ),
    );
}
