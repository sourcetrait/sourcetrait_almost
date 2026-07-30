//! Syllabus locks: what a walk finds, and what a fill has to satisfy.
use crate::syllabus::{
    MethodPath,
    methods,
    render,
};

/// The real scaffold's contract shape, multi-line and sugar-typed.
const CONTRACT: &str = "record<\n  proc: record<\n    path: path\n  >\n>";

/// A scratch root that removes itself, so a lock leaves no residue.
struct Scratch {
    base: std::path::PathBuf,
    root: std::path::PathBuf,
    yard: std::path::PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let base = std::env::temp_dir().join(format!(
            "quest_syllabus_{name}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("mix");
        let yard = base.join("yard");
        std::fs::create_dir_all(&root).expect("scratch root");
        Self { base, root, yard }
    }

    /// Put the tree under version control, so a render can be pinned.
    fn version(&self) {
        run_git(&self.root, &["init", "-b", "main"]);
        run_git(&self.root, &["add", "--all"]);
        run_git(&self.root, &["commit", "-m", "fixture"]);
    }

    /// Lay one method's two directories and hand back where it sits.
    fn method(&self, stage: &str, syllabus: &[&str], method: &str) -> std::path::PathBuf {
        let mut dir = self.root.join(stage);
        for segment in syllabus {
            dir = dir.join(segment);
        }
        let dir = dir.join(method);
        std::fs::create_dir_all(dir.join("tmpl")).expect("tmpl");
        std::fs::create_dir_all(dir.join("set")).expect("set");
        dir
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

fn write(path: std::path::PathBuf, text: &str) {
    std::fs::write(path, text).expect("writes");
}

fn run_git(dir: &std::path::Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .expect("git runs");
    assert!(
        output.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A method carrying one template, its contract, and one case.
fn one_case_method(scratch: &Scratch) -> MethodPath {
    let dir = scratch.method("sft", &["modeling", "nu", "typedef"], "design_schema");
    write(
        dir.join("tmpl").join("linux_proc.liquid"),
        "Design a schema for {{ train.proc.path }}.",
    );
    write(dir.join("tmpl").join("linux_proc.nutype"), CONTRACT);
    write(
        dir.join("set").join("mounts.nuon"),
        "{\n  proc: {\n    path: /proc/mounts\n  }\n}",
    );
    MethodPath {
        stage: String::from("sft"),
        syllabus: vec![
            String::from("modeling"),
            String::from("nu"),
            String::from("typedef"),
        ],
        method: String::from("design_schema"),
    }
}

#[test]
fn a_walk_finds_methods_by_their_template_directory() {
    let scratch = Scratch::new("walk");
    one_case_method(&scratch);
    scratch.method("dpo", &["formatting"], "relax_one_constraint");
    scratch.method("sft", &[], "bare_method");
    // A corpus sits beside the stages in the real repository, so a walk
    // that read the root's directories would descend it.
    std::fs::create_dir_all(scratch.root.join("corpus").join("nushell").join("tmpl"))
        .expect("corpus");

    let found = methods(&scratch.root).expect("walks");
    assert_eq!(
        found.iter().map(|m| m.stage.as_str()).collect::<Vec<_>>(),
        ["sft", "sft", "dpo"],
        "stages come in training order, not alphabetical order"
    );
    assert_eq!(
        found.iter().map(|m| m.syllabus_path()).collect::<Vec<_>>(),
        ["", "modeling/nu/typedef", "formatting"],
        "a method may sit at any syllabus depth, including none"
    );
    assert_eq!(
        found.iter().map(|m| m.method.as_str()).collect::<Vec<_>>(),
        ["bare_method", "design_schema", "relax_one_constraint"]
    );
}

#[test]
fn a_case_renders_with_its_fill_bound_as_train() {
    let scratch = Scratch::new("render");
    let method = one_case_method(&scratch);

    let rendered = render(&scratch.root, &method).expect("renders");
    assert_eq!(rendered.len(), 1);
    let case = &rendered[0];
    assert_eq!(case.text, "Design a schema for /proc/mounts.");
    assert_eq!(case.stage, "sft");
    assert_eq!(case.syllabus, "modeling/nu/typedef");
    assert_eq!(case.method, "design_schema");
    assert_eq!(case.template, "linux_proc");
    assert_eq!(case.case, "mounts");

    let record = case.to_value();
    assert!(
        record.get_type().to_string().contains("fill"),
        "provenance carries the fill it was rendered from"
    );
}

#[test]
fn every_template_renders_every_case_in_a_stable_order() {
    let scratch = Scratch::new("cross");
    let method = one_case_method(&scratch);
    let dir = method.dir(&scratch.root);
    write(
        dir.join("tmpl").join("another.liquid"),
        "Model {{ train.proc.path }} instead.",
    );
    write(dir.join("tmpl").join("another.nutype"), CONTRACT);
    write(
        dir.join("set").join("meminfo.nuon"),
        "{\n  proc: {\n    path: /proc/meminfo\n  }\n}",
    );

    let rendered = render(&scratch.root, &method).expect("renders");
    assert_eq!(
        rendered
            .iter()
            .map(|case| format!("{}:{}", case.template, case.case))
            .collect::<Vec<_>>(),
        [
            "another:meminfo",
            "another:mounts",
            "linux_proc:meminfo",
            "linux_proc:mounts"
        ],
        "templates and cases are both walked sorted, so a set is reproducible"
    );
}

#[test]
fn a_fill_that_drifts_from_its_contract_is_refused() {
    let scratch = Scratch::new("drift");
    let method = one_case_method(&scratch);
    write(
        method.dir(&scratch.root).join("set").join("wrong.nuon"),
        "{\n  proc: {\n    depth: 3\n  }\n}",
    );
    assert!(
        render(&scratch.root, &method).is_err(),
        "a fill missing a declared field fails at load rather than \
         rendering something wrong"
    );
}

#[test]
fn a_template_without_its_contract_is_refused() {
    let scratch = Scratch::new("uncontracted");
    let method = one_case_method(&scratch);
    write(
        method.dir(&scratch.root).join("tmpl").join("loose.liquid"),
        "no contract stands beside this",
    );
    assert!(render(&scratch.root, &method).is_err());
}

#[test]
fn a_contract_whose_template_is_gone_is_refused() {
    let scratch = Scratch::new("orphaned");
    let method = one_case_method(&scratch);
    write(
        method.dir(&scratch.root).join("tmpl").join("renamed.nutype"),
        CONTRACT,
    );
    assert!(
        render(&scratch.root, &method).is_err(),
        "an orphaned contract is a rename that half landed"
    );
}

#[test]
fn an_emission_lands_in_a_railroad_as_its_first_rev() {
    let scratch = Scratch::new("emit");
    one_case_method(&scratch);
    let dpo = scratch.method("dpo", &["formatting"], "relax_one_constraint");
    write(dpo.join("tmpl").join("relax.liquid"), "Relax {{ train.proc.path }}.");
    write(dpo.join("tmpl").join("relax.nutype"), CONTRACT);
    write(
        dpo.join("set").join("mounts.nuon"),
        "{\n  proc: {\n    path: /proc/mounts\n  }\n}",
    );
    scratch.version();

    let road = crate::railroad::Railroad::lay_in(&scratch.yard).expect("lays");
    let emitted = crate::syllabus::emit(&scratch.root, &road).expect("emits");

    assert_eq!(emitted.rev, 1, "the provenance is the railroad's first change");
    assert_eq!(emitted.methods, 2);
    assert_eq!(emitted.cases, 2);

    let provenance = crate::nu::load_value(&road.dir().join("provenance.nuon"))
        .expect("provenance reads back");
    let version = provenance
        .get_data_by_key("training_version")
        .expect("carries the scheme version");
    assert_eq!(
        version.as_str().expect("a string"),
        crate::consts::TRAINING_VERSION,
        "a run names the scheme version its binary was built from"
    );
    let commit = provenance
        .get_data_by_key("source_commit")
        .expect("carries the source commit");
    assert_eq!(
        commit.as_str().expect("a string").len(),
        40,
        "the render is reproducible from the tree it was drawn at"
    );
    assert!(
        !provenance
            .get_data_by_key("source_dirty")
            .expect("carries the source state")
            .as_bool()
            .expect("a bool"),
        "a fixture committed whole is clean"
    );

    for stage in ["sft", "dpo", "rlvr"] {
        assert!(
            !road.dir().join(format!("{stage}.nuon")).exists(),
            "{stage} writes no aggregate; the method is the unit"
        );
    }
}

#[test]
fn an_unversioned_source_tree_is_refused() {
    let scratch = Scratch::new("unversioned");
    one_case_method(&scratch);

    let road = crate::railroad::Railroad::lay_in(&scratch.yard).expect("lays");
    assert!(
        crate::syllabus::emit(&scratch.root, &road).is_err(),
        "a run that cannot name the tree it drew from is not reproducible"
    );
}

#[test]
fn a_dirty_source_tree_is_recorded_as_dirty() {
    let scratch = Scratch::new("dirty");
    let method = one_case_method(&scratch);
    scratch.version();
    write(
        method.dir(&scratch.root).join("set").join("meminfo.nuon"),
        "{\n  proc: {\n    path: /proc/meminfo\n  }\n}",
    );

    let road = crate::railroad::Railroad::lay_in(&scratch.yard).expect("lays");
    crate::syllabus::emit(&scratch.root, &road).expect("emits");

    let provenance = crate::nu::load_value(&road.dir().join("provenance.nuon"))
        .expect("provenance reads back");
    assert!(
        provenance
            .get_data_by_key("source_dirty")
            .expect("carries the source state")
            .as_bool()
            .expect("a bool"),
        "an uncommitted case means the commit does not describe the render"
    );
}

#[test]
fn a_method_with_no_cases_is_refused() {
    let scratch = Scratch::new("caseless");
    let dir = scratch.method("rlvr", &["verifying"], "value_equals");
    write(dir.join("tmpl").join("only.liquid"), "{{ train.proc.path }}");
    write(dir.join("tmpl").join("only.nutype"), CONTRACT);

    let method = MethodPath {
        stage: String::from("rlvr"),
        syllabus: vec![String::from("verifying")],
        method: String::from("value_equals"),
    };
    assert!(render(&scratch.root, &method).is_err());
}
