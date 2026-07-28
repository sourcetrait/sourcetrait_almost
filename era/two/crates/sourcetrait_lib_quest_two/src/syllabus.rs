//! The syllabus tree: training assets discovered, filled and rendered.
use crate::*;

/// The stages a syllabus tree carries, in the order they are trained.
pub const STAGES: [&str; 3] = ["sft", "dpo", "rlvr"];

/// Where a method keeps its templates.
const TEMPLATE_DIR: &str = "tmpl";

/// Where a method keeps its case fills.
const CASE_DIR: &str = "set";

const TEMPLATE_EXT: &str = "liquid";
const TYPEDEF_EXT: &str = "nutype";
const CASE_EXT: &str = "nuon";

/// The channel every fill binds to, whatever the syllabus.
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

/// One case rendered: the text, and everything it came from.
#[derive(Debug, Clone)]
pub struct RenderedCase {
    pub stage: String,
    pub syllabus: String,
    pub method: String,
    pub template: String,
    pub case: String,
    pub fill: nu::Value,
    pub text: String,
}

impl RenderedCase {
    /// The case as a nu record, for a generated set written as NUON.
    pub fn to_value(&self) -> nu::Value {
        let span = nu::Span::unknown();
        nu::Value::record(
            nu::record! {
                "stage" => nu::Value::string(self.stage.clone(), span),
                "syllabus" => nu::Value::string(self.syllabus.clone(), span),
                "method" => nu::Value::string(self.method.clone(), span),
                "template" => nu::Value::string(self.template.clone(), span),
                "case" => nu::Value::string(self.case.clone(), span),
                "fill" => self.fill.clone(),
                "text" => nu::Value::string(self.text.clone(), span),
            },
            span,
        )
    }
}

/// Every method under a root, in stage order and then path order.
pub fn methods(root: &Path) -> LibQuestResult<Vec<MethodPath>> {
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

/// Render one method's cases, each against each of its templates.
pub fn render(root: &Path, method: &MethodPath) -> LibQuestResult<Vec<RenderedCase>> {
    let dir = method.dir(root);
    let template_dir = dir.join(TEMPLATE_DIR);
    let case_dir = dir.join(CASE_DIR);
    snafu::ensure_whatever!(
        case_dir.is_dir(),
        "{} carries no {CASE_DIR}, so the method has no cases to fill",
        dir.display()
    );

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

    let cases = sorted_files(&case_dir, CASE_EXT)?;
    snafu::ensure_whatever!(
        !cases.is_empty(),
        "{} carries no {CASE_EXT} case",
        case_dir.display()
    );

    let mut rendered = Vec::with_capacity(templates.len() * cases.len());
    for template in &templates {
        let name = stem(template)?;
        let typedef = template.with_extension(TYPEDEF_EXT);
        snafu::ensure_whatever!(
            typedef.is_file(),
            "{} has no {TYPEDEF_EXT} beside it, so its fills have no contract",
            template.display()
        );
        let declared = nu::parse_typedef(fs::read_to_string(&typedef)?.trim())?;
        let source = fs::read_to_string(template)?;

        for case in &cases {
            let fill = nu::from_nuon_text(&fs::read_to_string(case)?)?;
            if let Err(error) = nu::conform(&fill, &declared) {
                snafu::whatever!(
                    "case {} does not fit {}: {error}",
                    case.display(),
                    typedef.display()
                );
            }
            let bindings = [(String::from(FILL_BINDING), fill.clone())];
            rendered.push(RenderedCase {
                stage: method.stage.clone(),
                syllabus: method.syllabus_path(),
                method: method.method.clone(),
                template: name.clone(),
                case: stem(case)?,
                text: template::render(&source, &bindings)?,
                fill,
            });
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
pub fn emit(root: &Path, railroad: &railroad::Railroad) -> LibQuestResult<Emitted> {
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
        &railroad.dir().join(format!("provenance.{CASE_EXT}")),
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

/// Descend a stage, collecting the directories that hold templates.
fn walk(
    dir: &Path,
    stage: &str,
    syllabus: &mut Vec<String>,
    found: &mut Vec<MethodPath>,
) -> LibQuestResult<()> {
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
fn sorted_dirs(dir: &Path) -> LibQuestResult<Vec<PathBuf>> {
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
fn sorted_files(dir: &Path, extension: &str) -> LibQuestResult<Vec<PathBuf>> {
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

fn stem(path: &Path) -> LibQuestResult<String> {
    match path.file_stem().and_then(|stem| stem.to_str()) {
        Some(stem) => Ok(stem.to_string()),
        None => snafu::whatever!("{} carries no readable stem", path.display()),
    }
}

fn dir_name(path: &Path) -> LibQuestResult<String> {
    match path.file_name().and_then(|name| name.to_str()) {
        Some(name) => Ok(name.to_string()),
        None => snafu::whatever!("{} carries no readable name", path.display()),
    }
}
