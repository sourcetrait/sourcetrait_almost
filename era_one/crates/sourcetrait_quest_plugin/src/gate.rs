use crate::*;

/// Which stage a check outcome came from (Pass = all stages green).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStage {
    Pass,
    TypedefParse,
    NuonParse,
    Conformance,
}

impl CheckStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            CheckStage::Pass => "pass",
            CheckStage::TypedefParse => "typedef_parse",
            CheckStage::NuonParse => "nuon_parse",
            CheckStage::Conformance => "conformance",
        }
    }
}

/// One conformance check's structured outcome - the probe battery's
/// row: the failing stage (or pass), the declared and derived type
/// renderings, and the parser error when a parse stage failed.
#[derive(Debug, Clone)]
pub struct CheckReport {
    pub stage: CheckStage,
    pub declared: String,
    pub derived: Option<String>,
    pub error: Option<String>,
}

/// Gate emitted NUON text against a declared typedef: parse the
/// typedef, parse the NUON, check deep conformance. Failures are data
/// (the probe's taxonomy), never errors.
pub fn check(typedef: &str, nuon_text: &str) -> CheckReport {
    let ty = match parse_typedef(typedef) {
        Ok(ty) => ty,
        Err(error) => {
            return CheckReport {
                stage: CheckStage::TypedefParse,
                declared: String::from(typedef),
                derived: None,
                error: Some(error.to_string()),
            };
        }
    };
    let declared = render_typedef(&ty);
    let value = match parse_nuon(nuon_text) {
        Ok(value) => value,
        Err(error) => {
            return CheckReport {
                stage: CheckStage::NuonParse,
                declared,
                derived: None,
                error: Some(error.to_string()),
            };
        }
    };
    let derived = render_typedef(&derive_type(&value));
    let stage = if conforms(&value, &ty) {
        CheckStage::Pass
    } else {
        CheckStage::Conformance
    };
    CheckReport {
        stage,
        declared,
        derived: Some(derived),
        error: None,
    }
}

/// quest_check's whole body: `quest_check <typedef> [nuon-file]`
/// (absent file = stdin), one NUON verdict record on stdout, exit 0 on
/// pass / 1 on any failing stage / 2 on usage-or-io faults.
pub fn check_main() -> i32 {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (typedef, nuon_text) = match arguments.as_slice() {
        [typedef] => {
            let mut buffer = String::new();
            if let Err(error) = io::stdin().read_to_string(&mut buffer) {
                eprintln!("quest_check: reading stdin failed: {error}");
                return 2;
            }
            (typedef.clone(), buffer)
        }
        [typedef, path] => match std::fs::read_to_string(path) {
            Ok(text) => (typedef.clone(), text),
            Err(error) => {
                eprintln!("quest_check: reading {path} failed: {error}");
                return 2;
            }
        },
        _ => {
            eprintln!("usage: quest_check <typedef> [nuon-file]  (absent file reads stdin)");
            return 2;
        }
    };
    let report = check(&typedef, &nuon_text);
    match render_nuon(&report_value(&report)) {
        Ok(text) => println!("{text}"),
        Err(error) => {
            eprintln!("quest_check: rendering the verdict failed: {error}");
            return 2;
        }
    }
    if report.stage == CheckStage::Pass { 0 } else { 1 }
}

/// The report as a nu record (the bin's NUON wire form).
fn report_value(report: &CheckReport) -> Value {
    let span = nu_protocol::Span::unknown();
    let mut record = nu_protocol::Record::new();
    record.push("stage", Value::string(report.stage.as_str(), span));
    record.push("declared", Value::string(report.declared.as_str(), span));
    record.push(
        "derived",
        match &report.derived {
            Some(derived) => Value::string(derived.as_str(), span),
            None => Value::nothing(span),
        },
    );
    record.push(
        "error",
        match &report.error {
            Some(error) => Value::string(error.as_str(), span),
            None => Value::nothing(span),
        },
    );
    Value::record(record, span)
}
