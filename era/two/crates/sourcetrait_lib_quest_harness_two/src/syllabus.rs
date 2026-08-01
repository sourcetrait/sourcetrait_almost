//! The syllabus tree: training assets discovered, validated and rendered.
use crate::*;

/// The stages a syllabus tree carries, in the order they are trained.
pub const STAGES: [&str; 3] = ["sft", "dpo", "rlvr"];

/// Where a method keeps its question templates.
const TEMPLATE_DIR: &str = "tmpl";

/// Where a method keeps its set files, one per template.
const SET_DIR: &str = "set";

/// Where a method keeps its answer key, one file per template.
const ACCEPT_DIR: &str = "accept";

const TEMPLATE_EXT: &str = "liquid";
const TYPEDEF_EXT: &str = "nutype";
const DATA_EXT: &str = "nuon";

/// The binding a slotted template reads its infill record through.
pub const FILL_BINDING: &str = "train";

/// Where one method sits: its stage, its syllabus, and its own name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MethodPath {
    pub stage: String,
    pub syllabus: Vec<String>,
    pub method: String,
}

impl MethodPath {
    /// The directory this method occupies under a root.
    pub fn dir(&self, root: &Path) -> PathBuf {
        let mut dir = root.join(&self.stage);
        for segment in &self.syllabus {
            dir = dir.join(segment);
        }
        dir.join(&self.method)
    }

    /// The syllabus as one slash-joined token; empty when there is none.
    pub fn syllabus_path(&self) -> String {
        self.syllabus.join("/")
    }
}

/// One accepted answer: the value, and the type it arrives as.
#[derive(Debug, Clone, PartialEq)]
pub struct Accepted {
    pub answer: nu::Value,
    /// A typedef string; an answer that IS a typedef is a string.
    pub typedef: String,
}

/// One set row joined to its answer key, the question rendered.
#[derive(Debug, Clone)]
pub struct RenderedCase {
    pub stage: String,
    pub syllabus: String,
    pub method: String,
    pub template: String,
    /// The case label joining the set row to its accepted answers.
    pub teach: String,
    /// The runtime call record's config half; Nothing on a bare call.
    pub config: nu::Value,
    /// The piped value; Nothing on an infill row.
    pub input: nu::Value,
    /// The question as asked: the template, infilled where the row says.
    pub prompt: String,
    pub accepted: Vec<Accepted>,
}

impl RenderedCase {
    /// The case as a nu record, for a generated set written as NUON.
    pub fn to_value(&self) -> nu::Value {
        let span = nu::Span::unknown();
        let accepted = self
            .accepted
            .iter()
            .map(|row| {
                nu::Value::record(
                    nu::record! {
                        "answer" => row.answer.clone(),
                        "typedef" => nu::Value::string(row.typedef.clone(), span),
                    },
                    span,
                )
            })
            .collect();
        nu::Value::record(
            nu::record! {
                "stage" => nu::Value::string(self.stage.clone(), span),
                "syllabus" => nu::Value::string(self.syllabus.clone(), span),
                "method" => nu::Value::string(self.method.clone(), span),
                "template" => nu::Value::string(self.template.clone(), span),
                "teach" => nu::Value::string(self.teach.clone(), span),
                "config" => self.config.clone(),
                "input" => self.input.clone(),
                "prompt" => nu::Value::string(self.prompt.clone(), span),
                "accepted" => nu::Value::list(accepted, span),
            },
            span,
        )
    }
}

/// One set row, before its answers are joined.
struct SetRow {
    teach: String,
    config: nu::Value,
    input: nu::Value,
}

/// Every method under a root, in stage order and then path order.
pub fn methods(root: &Path) -> HarnessQuestResult<Vec<MethodPath>> {
    let mut found = Vec::new();
    for stage in STAGES {
        let dir = root.join(stage);
        if !dir.is_dir() {
            continue;
        }
        let mut syllabus = Vec::new();
        walk(&dir, stage, &mut syllabus, &mut found)?;
    }
    Ok(found)
}

/// Render one method: each template's set rows against its answer key.
///
/// A template pairs with ONE set file and ONE accept file by its own
/// stem. Every fault is refused rather than skipped, because a skip
/// would let a drifted row vanish from a generated set in silence.
pub fn render(root: &Path, method: &MethodPath) -> HarnessQuestResult<Vec<RenderedCase>> {
    let dir = method.dir(root);
    let template_dir = dir.join(TEMPLATE_DIR);

    let templates = sorted_files(&template_dir, TEMPLATE_EXT)?;
    snafu::ensure_whatever!(
        !templates.is_empty(),
        "{} carries no {TEMPLATE_EXT} template",
        template_dir.display()
    );
    for typedef in sorted_files(&template_dir, TYPEDEF_EXT)? {
        snafu::ensure_whatever!(
            typedef.with_extension(TEMPLATE_EXT).is_file(),
            "{} contracts a template that is not there",
            typedef.display()
        );
    }

    let mut rendered = Vec::new();
    for template in &templates {
        let name = stem(template)?;
        let source = fs::read_to_string(template)?;
        let contract = contract_of(template)?;
        let rows = set_rows(&dir, &name)?;
        let mut answers = accept_rows(&dir, &name)?;

        for row in rows {
            let Some(accepted) = answers.remove(&row.teach) else {
                snafu::whatever!(
                    "set row `{}` of `{name}` has no accepted answers",
                    row.teach
                );
            };
            let prompt = prompt_of(&source, contract.as_ref(), &name, &row)?;
            rendered.push(RenderedCase {
                stage: method.stage.clone(),
                syllabus: method.syllabus_path(),
                method: method.method.clone(),
                template: name.clone(),
                teach: row.teach,
                config: row.config,
                input: row.input,
                prompt,
                accepted,
            });
        }

        if let Some(orphan) = answers.keys().next() {
            snafu::whatever!(
                "accepted answers for `{orphan}` of `{name}` join no set row"
            );
        }
    }
    Ok(rendered)
}

/// What one emission amounted to, and the REV it landed as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emitted {
    pub rev: usize,
    pub methods: usize,
    pub cases: usize,
}

/// Prove every method renders, and pin the run's provenance as a REV.
pub fn emit(root: &Path, railroad: &railroad::Railroad) -> HarnessQuestResult<Emitted> {
    let span = nu::Span::unknown();
    let found = methods(root)?;
    snafu::ensure_whatever!(
        !found.is_empty(),
        "{} carries no method, so there is nothing to generate",
        root.display()
    );
    let source = railroad::revision_of(root)?;

    let mut drawn = Vec::with_capacity(found.len());
    let mut total = 0usize;
    for method in &found {
        let rendered = render(root, method)?;
        total += rendered.len();
        drawn.push(nu::Value::record(
            nu::record! {
                "stage" => nu::Value::string(method.stage.clone(), span),
                "syllabus" => nu::Value::string(method.syllabus_path(), span),
                "method" => nu::Value::string(method.method.clone(), span),
                "cases" => nu::Value::int(rendered.len() as i64, span),
            },
            span,
        ));
    }

    nu::save_value(
        &railroad.dir().join(format!("provenance.{DATA_EXT}")),
        &nu::Value::record(
            nu::record! {
                "training_version" => nu::Value::string(consts::TRAINING_VERSION, span),
                "source_commit" => nu::Value::string(source.commit, span),
                "source_dirty" => nu::Value::bool(source.dirty, span),
                "drawn" => nu::Value::list(drawn.clone(), span),
            },
            span,
        ),
    )?;

    Ok(Emitted {
        rev: railroad.commit()?,
        methods: drawn.len(),
        cases: total,
    })
}

/// The slot contract standing beside a template, when one does.
///
/// Presence is the declaration: a template with a `.nutype` beside it
/// carries slots and its rows ride the infill channel; one without is
/// slotless and its rows pipe their data.
fn contract_of(template: &Path) -> HarnessQuestResult<Option<nu::Type>> {
    let path = template.with_extension(TYPEDEF_EXT);
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(nu::parse_typedef(fs::read_to_string(&path)?.trim())?))
}

/// The question one row is asked, decided by its data channel.
fn prompt_of(
    source: &str,
    contract: Option<&nu::Type>,
    template: &str,
    row: &SetRow,
) -> HarnessQuestResult<String> {
    let piped = !matches!(row.input, nu::Value::Nothing { .. });
    let liquid = liquid_of(&row.config, template, &row.teach)?;
    match (piped, liquid) {
        (true, Some(_)) => snafu::whatever!(
            "row `{}` of `{template}` carries both channels; a row rides at most one",
            row.teach
        ),
        // The no-input row: an ask whose answer lives outside the
        // conversation binds nothing, so neither channel is the honest
        // spelling - legal exactly where the template is slotless.
        (false, None) => {
            snafu::ensure_whatever!(
                contract.is_none(),
                "`{template}` carries a slot contract, so row `{}` must infill it",
                row.teach
            );
            Ok(source.trim_end_matches('\n').to_string())
        }
        (true, None) => {
            snafu::ensure_whatever!(
                contract.is_none(),
                "`{template}` carries a slot contract, so row `{}` cannot pipe its data",
                row.teach
            );
            Ok(source.trim_end_matches('\n').to_string())
        }
        (false, Some(fill)) => {
            let Some(declared) = contract else {
                snafu::whatever!(
                    "row `{}` of `{template}` infills a slotless template",
                    row.teach
                );
            };
            let Some(train) = fill.get(FILL_BINDING) else {
                snafu::whatever!(
                    "row `{}` of `{template}` binds no `{FILL_BINDING}` on its infill channel",
                    row.teach
                );
            };
            for (key, _) in fill.iter() {
                snafu::ensure_whatever!(
                    key == FILL_BINDING,
                    "row `{}` of `{template}` binds `{key}`; the infill channel binds \
                     `{FILL_BINDING}` alone",
                    row.teach
                );
            }
            if let Err(error) = nu::conform(train, declared) {
                snafu::whatever!(
                    "row `{}` of `{template}` does not fit the slot contract: {error}",
                    row.teach
                );
            }
            let bindings: Vec<(String, nu::Value)> = fill
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            Ok(template::render(source, &bindings)?
                .trim_end_matches('\n')
                .to_string())
        }
    }
}

/// The liquid record a row's config carries, when it carries one.
fn liquid_of<'a>(
    config: &'a nu::Value,
    template: &str,
    teach: &str,
) -> HarnessQuestResult<Option<&'a nu::Record>> {
    let record = match config {
        nu::Value::Nothing { .. } => return Ok(None),
        nu::Value::Record { val, .. } => val,
        other => snafu::whatever!(
            "row `{teach}` of `{template}` carries a config that is neither null nor \
             a record; got {}",
            other.get_type()
        ),
    };
    match record.get(think_harness::config::LIQUID_KEY) {
        None => Ok(None),
        Some(nu::Value::Record { val, .. }) => Ok(Some(&**val)),
        Some(other) => snafu::whatever!(
            "row `{teach}` of `{template}` carries a {} liquid value; the infill \
             channel is a record",
            other.get_type()
        ),
    }
}

/// One template's set rows, in file order.
fn set_rows(dir: &Path, template: &str) -> HarnessQuestResult<Vec<SetRow>> {
    let path = dir.join(SET_DIR).join(format!("{template}.{DATA_EXT}"));
    snafu::ensure_whatever!(
        path.is_file(),
        "`{template}` has no set file at {}",
        path.display()
    );
    let mut out: Vec<SetRow> = Vec::new();
    for row in table_rows(&nu::load_value(&path)?, &path)? {
        exact_keys(&row, &["teach", "data"], &path)?;
        let teach = string_field(&row, "teach", &path)?;
        snafu::ensure_whatever!(
            !out.iter().any(|known| known.teach == teach),
            "{} names `{teach}` more than once",
            path.display()
        );
        let data = match row.get("data") {
            Some(nu::Value::Record { val, .. }) => (**val).clone(),
            _ => snafu::whatever!(
                "row `{teach}` of {} carries no data record",
                path.display()
            ),
        };
        exact_keys(&data, &["config", "input"], &path)?;
        out.push(SetRow {
            teach,
            config: data.get("config").expect("checked above").clone(),
            input: data.get("input").expect("checked above").clone(),
        });
    }
    Ok(out)
}

/// One template's answer key, each answer conforming to its own typedef.
fn accept_rows(
    dir: &Path,
    template: &str,
) -> HarnessQuestResult<HashMap<String, Vec<Accepted>>> {
    let path = dir.join(ACCEPT_DIR).join(format!("{template}.{DATA_EXT}"));
    snafu::ensure_whatever!(
        path.is_file(),
        "`{template}` has no accept file at {}",
        path.display()
    );
    let mut out: HashMap<String, Vec<Accepted>> = HashMap::new();
    for row in table_rows(&nu::load_value(&path)?, &path)? {
        exact_keys(&row, &["teach", "accepted"], &path)?;
        let teach = string_field(&row, "teach", &path)?;
        snafu::ensure_whatever!(
            !out.contains_key(&teach),
            "{} names `{teach}` more than once",
            path.display()
        );
        let Some(accepted) = row.get("accepted") else {
            snafu::whatever!("row `{teach}` of {} carries no answers", path.display());
        };
        let mut answers = Vec::new();
        for answer in table_rows(accepted, &path)? {
            exact_keys(&answer, &["answer", "typedef"], &path)?;
            let typedef = string_field(&answer, "typedef", &path)?;
            let declared = nu::parse_typedef(&typedef)?;
            let value = answer.get("answer").expect("checked above").clone();
            if let Err(error) = nu::conform(&value, &declared) {
                snafu::whatever!(
                    "an accepted answer for `{teach}` of `{template}` misses its own \
                     typedef: {error}"
                );
            }
            answers.push(Accepted {
                answer: value,
                typedef,
            });
        }
        snafu::ensure_whatever!(
            !answers.is_empty(),
            "row `{teach}` of {} accepts nothing",
            path.display()
        );
        out.insert(teach, answers);
    }
    Ok(out)
}

/// A whole-value NUON table as its rows, each one a record.
fn table_rows(value: &nu::Value, path: &Path) -> HarnessQuestResult<Vec<nu::Record>> {
    let nu::Value::List { vals, .. } = value else {
        snafu::whatever!(
            "{} is a table of rows; got {}",
            path.display(),
            value.get_type()
        );
    };
    let mut rows = Vec::with_capacity(vals.len());
    for item in vals {
        match item {
            nu::Value::Record { val, .. } => rows.push((**val).clone()),
            other => snafu::whatever!(
                "{} carries a {} where a row belongs",
                path.display(),
                other.get_type()
            ),
        }
    }
    Ok(rows)
}

/// Refuse a row whose keys drift from the declared shape.
fn exact_keys(record: &nu::Record, keys: &[&str], path: &Path) -> HarnessQuestResult<()> {
    for key in keys {
        snafu::ensure_whatever!(
            record.get(key).is_some(),
            "{} carries a row with no `{key}`",
            path.display()
        );
    }
    for (key, _) in record.iter() {
        snafu::ensure_whatever!(
            keys.contains(&key.as_str()),
            "{} carries a row with a stray `{key}`",
            path.display()
        );
    }
    Ok(())
}

/// A row's string field.
fn string_field(record: &nu::Record, key: &str, path: &Path) -> HarnessQuestResult<String> {
    match record.get(key) {
        Some(nu::Value::String { val, .. }) => Ok(val.clone()),
        Some(other) => snafu::whatever!(
            "{} carries a {} `{key}`; a `{key}` is a string",
            path.display(),
            other.get_type()
        ),
        None => snafu::whatever!("{} carries a row with no `{key}`", path.display()),
    }
}

/// Descend a stage, collecting the directories that hold templates.
fn walk(
    dir: &Path,
    stage: &str,
    syllabus: &mut Vec<String>,
    found: &mut Vec<MethodPath>,
) -> HarnessQuestResult<()> {
    for child in sorted_dirs(dir)? {
        let name = dir_name(&child)?;
        if child.join(TEMPLATE_DIR).is_dir() {
            found.push(MethodPath {
                stage: stage.to_string(),
                syllabus: syllabus.clone(),
                method: name,
            });
            continue;
        }
        syllabus.push(name);
        walk(&child, stage, syllabus, found)?;
        syllabus.pop();
    }
    Ok(())
}

/// A directory's own subdirectories, sorted so a walk is reproducible.
fn sorted_dirs(dir: &Path) -> HarnessQuestResult<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            dirs.push(entry.path());
        }
    }
    dirs.sort();
    Ok(dirs)
}

/// A directory's files carrying one extension, sorted.
fn sorted_files(dir: &Path, extension: &str) -> HarnessQuestResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let carries = path.extension().and_then(|found| found.to_str()) == Some(extension);
        if entry.file_type()?.is_file() && carries {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn stem(path: &Path) -> HarnessQuestResult<String> {
    match path.file_stem().and_then(|stem| stem.to_str()) {
        Some(stem) => Ok(stem.to_string()),
        None => snafu::whatever!("{} carries no readable stem", path.display()),
    }
}

fn dir_name(path: &Path) -> HarnessQuestResult<String> {
    match path.file_name().and_then(|name| name.to_str()) {
        Some(name) => Ok(name.to_string()),
        None => snafu::whatever!("{} carries no readable name", path.display()),
    }
}
