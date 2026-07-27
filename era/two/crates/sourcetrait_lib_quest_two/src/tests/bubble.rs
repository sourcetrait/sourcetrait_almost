//! Bubble locks: real nu under real confinement, so what these exercise
//! is what ships rather than a double standing in for it.
use crate::bubble::{
    BubbleHarness,
    BubbleWorld,
};
use crate::harness::{
    ClientHarness,
    HarnessRequest,
    HarnessResponse,
    QuestNuValue,
    RequestBinding,
};

/// A scratch directory that removes itself.
struct Scratch {
    root: std::path::PathBuf,
}

impl Scratch {
    fn make(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("core_bubble_{name}"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch root");
        Self { root }
    }

    fn write(&self, name: &str, body: &str) {
        std::fs::write(self.root.join(name), body).expect("writes");
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn request(mode: &str, source: &str, output: &str, bindings: &[(&str, &str)]) -> HarnessRequest {
    HarnessRequest {
        mode: mode.to_string(),
        source: source.to_string(),
        output: output.to_string(),
        bindings: bindings
            .iter()
            .map(|(pass, nuon)| RequestBinding {
                pass: (*pass).to_string(),
                value: QuestNuValue::from_nuon(nuon).expect("the fixture parses"),
            })
            .collect(),
    }
}

fn refusal(response: &HarnessResponse) -> (&str, &str) {
    match response {
        HarnessResponse::Failed { kind, message } => (kind, message),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// The whole round trip: a def, a channel bound to it, and a value back.
#[test]
fn a_mode_runs_and_answers_with_its_value() {
    let mut harness = BubbleHarness::default();
    let response = harness
        .serve(&request(
            "shout",
            "def shout []: list<string> -> list<string> { $in | each {|w| $w | str upcase } }",
            "list<string>",
            &[("$in", "[foo, bar]")],
        ))
        .expect("the sandbox ran");

    let HarnessResponse::Value { value, declared } = &response else {
        panic!("{response:?}");
    };
    assert_eq!(value.to_nuon().expect("renders"), "[FOO, BAR]");
    assert_eq!(declared, "list<string>");
}

/// The args positional arrives as a NUON literal at the call site, which
/// works because NUON is valid nu literal syntax.
#[test]
fn an_args_positional_is_bound_at_the_call_site() {
    let mut harness = BubbleHarness::default();
    let response = harness
        .serve(&request(
            "add",
            "def add [args: record<a: int, b: int>]: nothing -> int { $args.a + $args.b }",
            "int",
            &[("$args", "{a: 40, b: 2}")],
        ))
        .expect("the sandbox ran");

    let HarnessResponse::Value { value, .. } = &response else {
        panic!("{response:?}");
    };
    assert_eq!(value.to_nuon().expect("renders"), "42");
}

/// THE WORLD IS A PARAMETER, and this is the lock that says so: the same
/// harness answers differently because the world it was pointed at
/// changed, with no test mode and no path argument in the def.
#[test]
fn the_world_decides_what_a_mode_reads() {
    let first = Scratch::make("proc_first");
    first.write("meminfo", "MemTotal:       1024 kB\n");
    let second = Scratch::make("proc_second");
    second.write("meminfo", "MemTotal:       9999 kB\n");

    let source = "def total []: nothing -> string { open --raw /proc/meminfo | lines | first }";
    let ask = || request("total", source, "string", &[]);

    let mut harness = BubbleHarness::new(BubbleWorld {
        proc: Some(first.root.clone()),
        ..BubbleWorld::default()
    });

    let HarnessResponse::Value { value, .. } = harness.serve(&ask()).expect("ran") else {
        panic!("the first world answered with a refusal");
    };
    assert!(
        value.to_nuon().expect("renders").contains("1024"),
        "the mode read the world it was given"
    );

    let mut next = harness.world().clone();
    next.proc = Some(second.root.clone());
    harness.set_world(next);

    let HarnessResponse::Value { value, .. } = harness.serve(&ask()).expect("ran") else {
        panic!("the second world answered with a refusal");
    };
    assert!(
        value.to_nuon().expect("renders").contains("9999"),
        "changing the world between turns changed the answer"
    );
}

/// A path the world does not mock reads the SANDBOX's own procfs rather
/// than failing, which is the trade the fresh-procfs profile makes.
///
/// The verifier profile beside this one replaces `/proc` wholesale so an
/// unmocked read fails loudly, and that is the better property - but it
/// cannot be had here, because nushell resolves its own executable
/// through `/proc/self/exe` and will not start without a real procfs.
#[test]
fn an_unmocked_path_reads_the_sandbox_procfs() {
    let scratch = Scratch::make("proc_sparse");
    scratch.write("meminfo", "MemTotal:       1 kB\n");

    let mut harness = BubbleHarness::new(BubbleWorld {
        proc: Some(scratch.root.clone()),
        ..BubbleWorld::default()
    });

    let response = harness
        .serve(&request(
            "cpus",
            "def cpus []: nothing -> int { open --raw /proc/cpuinfo | lines | length }",
            "int",
            &[],
        ))
        .expect("the sandbox ran");

    let HarnessResponse::Value { value, .. } = &response else {
        panic!("{response:?}");
    };
    let lines: i64 = value.to_nuon().expect("renders").parse().expect("an int");
    assert!(lines > 0, "the fresh procfs answered rather than the mock");
}

/// A mode that never returns is STOPPED rather than waited on, because a
/// generated pipeline can loop forever.
#[test]
fn a_mode_that_never_returns_is_stopped_at_the_deadline() {
    let mut harness = BubbleHarness::new(BubbleWorld {
        timeout: std::time::Duration::from_millis(400),
        ..BubbleWorld::default()
    });

    // The body has to TYPECHECK as an int even though it never returns
    // one: nushell parse-checks a statically-typed result, so a bare
    // `loop { }` against `-> int` is rejected before it can spin.
    let started = std::time::Instant::now();
    let response = harness
        .serve(&request(
            "spin",
            "def spin []: nothing -> int { mut n = 0; loop { $n += 1 }; $n }",
            "int",
            &[],
        ))
        .expect("the sandbox ran");
    let waited = started.elapsed();

    let (kind, _) = refusal(&response);
    assert_eq!(kind, "harness::timeout");
    assert!(
        waited < std::time::Duration::from_secs(5),
        "the deadline bounded the wait rather than the mode ending on its own"
    );
}

/// Nushell parse-checks a STATIC result against a def's output type but
/// does not enforce a DYNAMIC one, so a body that builds its answer at
/// runtime returns off its own signature happily. This is where that is
/// caught, which is the whole reason the request carries the type.
///
/// Built through `from nuon` deliberately: a literal `[1 2 3]` here is
/// rejected by the parser instead, which would prove nothing about our
/// own check.
#[test]
fn an_answer_off_its_own_signature_is_refused() {
    let mut harness = BubbleHarness::default();
    let response = harness
        .serve(&request(
            "lie",
            "def lie []: nothing -> list<string> { '[1 2 3]' | from nuon }",
            "list<string>",
            &[],
        ))
        .expect("the sandbox ran");

    let (kind, message) = refusal(&response);
    assert_eq!(kind, "harness::conformance");
    assert!(message.contains("lie"), "{message}");
}

/// A mode that fails is a RESPONSE rather than an error, so the only Err
/// this can produce is the sandbox failing to start at all.
#[test]
fn a_failing_mode_answers_rather_than_erroring() {
    let mut harness = BubbleHarness::default();
    let response = harness
        .serve(&request(
            "boom",
            "def boom []: nothing -> int { error make { msg: \"no\" } }",
            "int",
            &[],
        ))
        .expect("the sandbox ran rather than erroring");

    let (kind, _) = refusal(&response);
    assert_eq!(kind, "harness::bubble");
}

/// The confinement profile itself, since it is the part that is
/// security-relevant and the part a later edit could quietly widen.
#[test]
fn the_default_world_is_the_confined_one() {
    let argv = BubbleWorld::default().argv();
    assert!(argv.windows(3).any(|w| w == ["--ro-bind", "/", "/"]));
    assert!(argv.iter().any(|a| a == "--die-with-parent"));
    assert!(
        argv.windows(2).any(|w| w == ["--proc", "/proc"]),
        "a fresh procfs is mounted, without which nushell cannot start"
    );
    assert!(
        argv.iter().any(|a| a == "--unshare-net"),
        "networking is off unless a world asks for it"
    );
    assert!(
        !argv.iter().any(|a| a == "--bind"),
        "nothing is writable by default"
    );

    let opened = BubbleWorld {
        network: true,
        writable: vec![std::path::PathBuf::from("/scratch")],
        ..BubbleWorld::default()
    }
    .argv();
    assert!(!opened.iter().any(|a| a == "--unshare-net"));
    assert!(
        opened.windows(3).any(|w| w == ["--bind", "/scratch", "/scratch"]),
        "a writable path is bound over itself"
    );
}
