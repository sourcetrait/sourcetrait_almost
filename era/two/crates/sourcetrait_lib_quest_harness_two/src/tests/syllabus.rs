//! Syllabus locks: what a walk finds, and what a row has to satisfy.
use crate::syllabus::{
    MethodPath,
    methods,
    render,
};

/// The slot contract the infill fixture's rows must fit.
const CONTRACT: &str = "record<\n  path: path\n>";

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

    /// Lay one method's three directories and hand back where it sits.
    fn method(&self, stage: &str, syllabus: &[&str], method: &str) -> std::path::PathBuf {
        let mut dir = self.root.join(stage);
        for segment in syllabus {
            dir = dir.join(segment);
        }
        let dir = dir.join(method);
        std::fs::create_dir_all(dir.join("tmpl")).expect("tmpl");
        std::fs::create_dir_all(dir.join("set")).expect("set");
        std::fs::create_dir_all(dir.join("accept")).expect("accept");
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

/// A slotted template with one infill row and its accepted answer.
fn infill_method(scratch: &Scratch) -> MethodPath {
    let dir = scratch.method("sft", &["modeling", "nu", "typedef"], "design_schema");
    write(
        dir.join("tmpl").join("linux_proc.liquid"),
        "Design a schema for {{ train.path }}.\n",
    );
    write(dir.join("tmpl").join("linux_proc.nutype"), CONTRACT);
    write(
        dir.join("set").join("linux_proc.nuon"),
        "[\n  [teach, data];\n  [mounts, { config: { liquid: { train: { path: \
         \"/proc/mounts\" } } }, input: null }]\n]\n",
    );
    write(
        dir.join("accept").join("linux_proc.nuon"),
        "[\n  [teach, accepted];\n  [mounts, [[answer, typedef]; \
         [\"table<device: string>\", \"string\"]]]\n]\n",
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

/// A slotless template whose rows pipe their data.
fn piped_method(scratch: &Scratch) -> MethodPath {
    let dir = scratch.method("sft", &["offloading"], "data_transform");
    write(dir.join("tmpl").join("join_together.liquid"), "Join this together\n");
    write(
        dir.join("set").join("join_together.nuon"),
        "[\n  [teach, data];\n  [strings_two, { config: null, input: [to, morrow] }]\n  \
         [strings_declared, { config: { shape: { response: [output] } }, \
         input: [data, base] }]\n]\n",
    );
    write(
        dir.join("accept").join("join_together.nuon"),
        "[\n  [teach, accepted];\n  [strings_two, [[answer, typedef]; [\"tomorrow\", \
         \"string\"]]]\n  [strings_declared, [[answer, typedef]; [\"database\", \
         \"string\"]]]\n]\n",
    );
    MethodPath {
        stage: String::from("sft"),
        syllabus: vec![String::from("offloading")],
        method: String::from("data_transform"),
    }
}

#[test]
fn a_walk_finds_methods_by_their_template_directory() {
    let scratch = Scratch::new("walk");
    infill_method(&scratch);
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
fn an_infill_row_renders_with_its_train_record_bound() {
    let scratch = Scratch::new("infill");
    let method = infill_method(&scratch);

    let rendered = render(&scratch.root, &method).expect("renders");
    assert_eq!(rendered.len(), 1);
    let case = &rendered[0];
    assert_eq!(case.prompt, "Design a schema for /proc/mounts.");
    assert_eq!(case.stage, "sft");
    assert_eq!(case.syllabus, "modeling/nu/typedef");
    assert_eq!(case.method, "design_schema");
    assert_eq!(case.template, "linux_proc");
    assert_eq!(case.teach, "mounts");
    assert!(
        matches!(case.input, crate::nu::Value::Nothing { .. }),
        "an infill row pipes nothing"
    );
    assert_eq!(case.accepted.len(), 1);
    assert_eq!(case.accepted[0].typedef, "string");

    let record = case.to_value();
    let shape = record.get_type().to_string();
    assert!(shape.contains("teach"), "provenance carries the case label: {shape}");
    assert!(shape.contains("accepted"), "the answer key travels with the case: {shape}");
}

#[test]
fn a_piped_row_carries_the_template_as_its_question() {
    let scratch = Scratch::new("piped");
    let method = piped_method(&scratch);

    let rendered = render(&scratch.root, &method).expect("renders");
    assert_eq!(rendered.len(), 2, "one case per set row, in file order");
    assert_eq!(rendered[0].prompt, "Join this together");
    assert_eq!(rendered[0].teach, "strings_two");
    assert!(
        matches!(rendered[0].config, crate::nu::Value::Nothing { .. }),
        "a bare call's config is null"
    );
    assert_eq!(
        crate::nu::to_nuon_text(&rendered[0].input).expect("renders"),
        "[to, morrow]",
        "the piped value rides the case"
    );
    // Declarations compose with either channel: the second row is the
    // same question under a declared response shape.
    assert_eq!(rendered[1].teach, "strings_declared");
    let crate::nu::Value::Record { val, .. } = &rendered[1].config else {
        panic!("a declared row's config is a record");
    };
    assert!(val.get("shape").is_some());
}

#[test]
fn templates_walk_sorted_and_rows_keep_file_order() {
    let scratch = Scratch::new("order");
    let method = piped_method(&scratch);
    let dir = method.dir(&scratch.root);
    write(dir.join("tmpl").join("add_up.liquid"), "What do these add up to?\n");
    write(
        dir.join("set").join("add_up.nuon"),
        "[\n  [teach, data];\n  [ints, { config: null, input: [12, 45, 7] }]\n]\n",
    );
    write(
        dir.join("accept").join("add_up.nuon"),
        "[\n  [teach, accepted];\n  [ints, [[answer, typedef]; [64, \"int\"]]]\n]\n",
    );

    let rendered = render(&scratch.root, &method).expect("renders");
    assert_eq!(
        rendered
            .iter()
            .map(|case| format!("{}:{}", case.template, case.teach))
            .collect::<Vec<_>>(),
        [
            "add_up:ints",
            "join_together:strings_two",
            "join_together:strings_declared"
        ],
        "templates sort; a set file's rows keep their authored order"
    );
}

#[test]
fn a_row_carrying_both_channels_is_refused() {
    let scratch = Scratch::new("both");
    let method = infill_method(&scratch);
    write(
        method.dir(&scratch.root).join("set").join("linux_proc.nuon"),
        "[\n  [teach, data];\n  [mounts, { config: { liquid: { train: { path: \
         \"/proc/mounts\" } } }, input: [1, 2] }]\n]\n",
    );
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("both channels"), "got {error}");
}

#[test]
fn a_no_input_row_renders_on_a_slotless_template() {
    let scratch = Scratch::new("neither");
    let method = piped_method(&scratch);
    write(
        method.dir(&scratch.root).join("set").join("join_together.nuon"),
        "[\n  [teach, data];\n  [strings_two, { config: null, input: null }]\n  \
         [strings_declared, { config: null, input: [data, base] }]\n]\n",
    );
    let rendered = render(&scratch.root, &method).expect("renders");
    let case = rendered
        .iter()
        .find(|case| case.teach == "strings_two")
        .expect("the no-input row renders");
    assert_eq!(
        case.prompt, "Join this together",
        "a no-input row's prompt is the bare template body"
    );
    assert!(
        matches!(case.input, crate::nu::Value::Nothing { .. }),
        "a no-input row carries no piped value"
    );
}

#[test]
fn a_no_input_row_against_a_slotted_template_is_refused() {
    let scratch = Scratch::new("neither_slotted");
    let method = infill_method(&scratch);
    write(
        method.dir(&scratch.root).join("set").join("linux_proc.nuon"),
        "[\n  [teach, data];\n  [mounts, { config: null, input: null }]\n]\n",
    );
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("must infill it"), "got {error}");
}

#[test]
fn an_infill_row_against_a_slotless_template_is_refused() {
    let scratch = Scratch::new("slotless");
    let method = piped_method(&scratch);
    write(
        method.dir(&scratch.root).join("set").join("join_together.nuon"),
        "[\n  [teach, data];\n  [strings_two, { config: { liquid: { train: { data: \
         \"my dog blue\" } } }, input: null }]\n  [strings_declared, { config: null, \
         input: [data, base] }]\n]\n",
    );
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("slotless"), "got {error}");
}

#[test]
fn a_piped_row_against_a_slotted_template_is_refused() {
    let scratch = Scratch::new("slotted");
    let method = infill_method(&scratch);
    write(
        method.dir(&scratch.root).join("set").join("linux_proc.nuon"),
        "[\n  [teach, data];\n  [mounts, { config: null, input: \"/proc/mounts\" }]\n]\n",
    );
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("slot contract"), "got {error}");
}

#[test]
fn a_train_record_off_its_contract_is_refused() {
    let scratch = Scratch::new("drift");
    let method = infill_method(&scratch);
    write(
        method.dir(&scratch.root).join("set").join("linux_proc.nuon"),
        "[\n  [teach, data];\n  [mounts, { config: { liquid: { train: { depth: 3 } } }, \
         input: null }]\n]\n",
    );
    assert!(
        render(&scratch.root, &method).is_err(),
        "a train record missing a declared field fails at load rather than \
         rendering something wrong"
    );
}

#[test]
fn a_set_row_with_no_accepted_answers_is_refused() {
    let scratch = Scratch::new("unanswered");
    let method = infill_method(&scratch);
    write(
        method.dir(&scratch.root).join("accept").join("linux_proc.nuon"),
        "[\n  [teach, accepted];\n  [meminfo, [[answer, typedef]; [\"record<a: int>\", \
         \"string\"]]]\n]\n",
    );
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("no accepted answers"), "got {error}");
}

#[test]
fn an_accepted_answer_joining_no_set_row_is_refused() {
    let scratch = Scratch::new("orphan_answer");
    let method = infill_method(&scratch);
    write(
        method.dir(&scratch.root).join("accept").join("linux_proc.nuon"),
        "[\n  [teach, accepted];\n  [mounts, [[answer, typedef]; [\"table<device: \
         string>\", \"string\"]]]\n  [meminfo, [[answer, typedef]; [\"record<a: int>\", \
         \"string\"]]]\n]\n",
    );
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("join"), "got {error}");
}

#[test]
fn an_accepted_answer_missing_its_own_typedef_is_refused() {
    let scratch = Scratch::new("mistyped");
    let method = piped_method(&scratch);
    write(
        method.dir(&scratch.root).join("accept").join("join_together.nuon"),
        "[\n  [teach, accepted];\n  [strings_two, [[answer, typedef]; [\"tomorrow\", \
         \"int\"]]]\n  [strings_declared, [[answer, typedef]; [\"database\", \
         \"string\"]]]\n]\n",
    );
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("misses its own typedef"), "got {error}");
}

#[test]
fn a_template_without_a_set_file_is_refused() {
    let scratch = Scratch::new("setless");
    let method = infill_method(&scratch);
    std::fs::remove_file(
        method.dir(&scratch.root).join("set").join("linux_proc.nuon"),
    )
    .expect("removes");
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("no set file"), "got {error}");
}

#[test]
fn a_template_without_an_accept_file_is_refused() {
    let scratch = Scratch::new("acceptless");
    let method = infill_method(&scratch);
    std::fs::remove_file(
        method.dir(&scratch.root).join("accept").join("linux_proc.nuon"),
    )
    .expect("removes");
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("no accept file"), "got {error}");
}

#[test]
fn a_duplicate_teach_is_refused() {
    let scratch = Scratch::new("twice");
    let method = piped_method(&scratch);
    write(
        method.dir(&scratch.root).join("set").join("join_together.nuon"),
        "[\n  [teach, data];\n  [strings_two, { config: null, input: [to, morrow] }]\n  \
         [strings_two, { config: null, input: [note, book] }]\n]\n",
    );
    let error = render(&scratch.root, &method).expect_err("refused").to_string();
    assert!(error.contains("more than once"), "got {error}");
}

#[test]
fn a_contract_whose_template_is_gone_is_refused() {
    let scratch = Scratch::new("orphaned");
    let method = infill_method(&scratch);
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
    infill_method(&scratch);
    piped_method(&scratch);
    scratch.version();

    let road = crate::railroad::Railroad::lay_in(&scratch.yard).expect("lays");
    let emitted = crate::syllabus::emit(&scratch.root, &road).expect("emits");

    assert_eq!(emitted.rev, 1, "the provenance is the railroad's first change");
    assert_eq!(emitted.methods, 2);
    assert_eq!(emitted.cases, 3, "one case per set row across both methods");

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
    infill_method(&scratch);

    let road = crate::railroad::Railroad::lay_in(&scratch.yard).expect("lays");
    assert!(
        crate::syllabus::emit(&scratch.root, &road).is_err(),
        "a run that cannot name the tree it drew from is not reproducible"
    );
}

#[test]
fn a_dirty_source_tree_is_recorded_as_dirty() {
    let scratch = Scratch::new("dirty");
    let method = infill_method(&scratch);
    scratch.version();
    write(
        method.dir(&scratch.root).join("set").join("linux_proc.nuon"),
        "[\n  [teach, data];\n  [mounts, { config: { shape: { response: [output] }, \
         liquid: { train: { path: \"/proc/mounts\" } } }, input: null }]\n]\n",
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
        "an uncommitted row means the commit does not describe the render"
    );
}
