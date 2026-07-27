//! The locked-down nu engine the evaluate sub-turn round-trips on.
use crate::*;

/// Stack size for any thread that parses; the parser is stack-hungry.
pub const PARSE_STACK_BYTES: usize = 16 * 1024 * 1024;

/// Names that would reach the operating system or end the host.
pub const DENIED: [&str; 10] = [
    "exit",
    "exec",
    "panic",
    "run-external",
    "cd",
    "open",
    "save",
    "rm",
    "http",
    "ls",
];

/// Parse-time loaders registration cannot close; the sandbox does.
pub const PARSE_TIME_LOADERS: [&str; 2] = ["use", "overlay use"];

/// The engine the inside path runs on: no filesystem, no externals.
pub struct QuestnessEvaluator {
    base: r::nu::EngineState,
}

impl QuestnessEvaluator {
    /// Build the base engine once; every evaluation clones it.
    pub fn new() -> QuestCoreResult<Self> {
        let mut base = nu_cmd_lang::create_default_context();
        base = register_filters(base)?;

        let mut config = base.get_config().as_ref().clone();
        config.use_ansi_coloring = nu_protocol::UseAnsiColoring::False;
        base.set_config(config);

        base.is_interactive = false;
        base.is_login = false;
        base.is_lsp = false;
        // Our stdout is a protocol channel: an external gets a null
        // stdin rather than ours, and `print` routes to stderr.
        base.is_mcp = true;
        base.add_env_var(String::from("PWD"), nu::Value::string("", nu::Span::unknown()));

        Ok(Self { base })
    }

    /// The command names this engine resolves, sorted.
    pub fn decl_names(&self) -> Vec<String> {
        self.base
            .get_decls_sorted(false)
            .into_iter()
            .map(|(name, _)| String::from_utf8_lossy(&name).to_string())
            .collect()
    }

    /// Whether a name resolves to a command on this engine.
    pub fn resolves(&self, name: &str) -> bool {
        self.decl_names().iter().any(|known| known == name)
    }

    /// Parse a body and return the one declaration it introduces.
    pub(crate) fn parse_fresh_decl(
        &self,
        source: &str,
        before: &[String],
    ) -> QuestCoreResult<(String, nu_protocol::Signature)> {
        let mut engine = self.base.clone();
        let mut working_set = r::nu::StateWorkingSet::new(&engine);
        let _block = r::nu::parse(&mut working_set, None, source.as_bytes(), false);
        if let Some(error) = working_set.parse_errors.first() {
            let rendered = nu_protocol::format_cli_error(None, &working_set, error, None);
            snafu::whatever!("{rendered}");
        }
        let delta = working_set.render();
        if let Err(error) = engine.merge_delta(delta) {
            snafu::whatever!("merging the parsed definition failed: {error}");
        }
        let mut fresh: Vec<(String, nu_protocol::Signature)> = Vec::new();
        for (name, id) in engine.get_decls_sorted(false) {
            let name = String::from_utf8_lossy(&name).to_string();
            if before.contains(&name) {
                continue;
            }
            fresh.push((name, engine.get_decl(id).signature()));
        }
        match fresh.len() {
            1 => Ok(fresh.remove(0)),
            0 => snafu::whatever!("the body declares nothing; a nu block carries a definition"),
            count => snafu::whatever!("the body declares {count} definitions; expected one"),
        }
    }

    /// Evaluate one source body, optionally piping a value into it.
    ///
    /// Runs on its own sized thread: a stack overflow in the parser is
    /// a fatal abort rather than a catchable error, so an async
    /// runtime worker's stack is not safe to parse on.
    pub fn evaluate(&self, source: &str, input: Option<nu::Value>) -> QuestCoreResult<nu::Value> {
        let engine = self.base.clone();
        let source = source.to_string();
        let spawned = std::thread::Builder::new()
            .name(String::from("questness-eval"))
            .stack_size(PARSE_STACK_BYTES)
            .spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    eval_once(engine, &source, input)
                }))
            });
        let spawned = match spawned {
            Ok(spawned) => spawned,
            Err(error) => snafu::whatever!("could not spawn the evaluator thread: {error}"),
        };
        match spawned.join() {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => snafu::whatever!("the evaluator panicked"),
            Err(_) => snafu::whatever!("the evaluator thread was lost"),
        }
    }
}

/// Parse-check, merge, evaluate, and take the one value back out.
fn eval_once(
    mut engine: r::nu::EngineState,
    source: &str,
    input: Option<nu::Value>,
) -> QuestCoreResult<nu::Value> {
    engine.set_signals(nu_protocol::Signals::new(std::sync::Arc::new(
        std::sync::atomic::AtomicBool::new(false),
    )));

    let mut working_set = r::nu::StateWorkingSet::new(&engine);
    let block = r::nu::parse(&mut working_set, None, source.as_bytes(), false);
    if let Some(error) = working_set.parse_errors.first() {
        let rendered = nu_protocol::format_cli_error(None, &working_set, error, None);
        snafu::whatever!("{rendered}");
    }
    if let Some(error) = working_set.compile_errors.first() {
        let rendered = nu_protocol::format_cli_error(None, &working_set, error, None);
        snafu::whatever!("{rendered}");
    }
    let delta = working_set.render();
    if let Err(error) = engine.merge_delta(delta) {
        snafu::whatever!("merging the parsed source failed: {error}");
    }

    let mut stack = r::nu::Stack::new().capture_all();
    let pipeline = match input {
        Some(value) => nu_protocol::PipelineData::value(value, None),
        None => nu_protocol::PipelineData::empty(),
    };
    let executed = match nu_engine::eval_block_with_early_return::<r::nu::WithoutDebug>(
        &engine,
        &mut stack,
        &block,
        pipeline,
    ) {
        Ok(executed) => executed,
        Err(error) => snafu::whatever!("{error}"),
    };
    match executed.body.into_value(nu::Span::unknown()) {
        Ok(value) => Ok(value),
        Err(error) => snafu::whatever!("{error}"),
    }
}

/// Add the filter set on top of the language core, then merge.
fn register_filters(mut engine: r::nu::EngineState) -> QuestCoreResult<r::nu::EngineState> {
    use nu_command::{
        Append, DropColumn, Each, Enumerate, Filter, Find, First, Flatten, Get, Last, Length,
        Prepend, Reject, Reverse, Select, Skip, Sort, Take, Uniq, Where, Wrap,
    };
    let delta = {
        let mut ws = r::nu::StateWorkingSet::new(&engine);
        ws.add_decl(Box::new(Append));
        ws.add_decl(Box::new(DropColumn));
        ws.add_decl(Box::new(Each));
        ws.add_decl(Box::new(Enumerate));
        ws.add_decl(Box::new(Filter));
        ws.add_decl(Box::new(Find));
        ws.add_decl(Box::new(First));
        ws.add_decl(Box::new(Flatten));
        ws.add_decl(Box::new(Get));
        ws.add_decl(Box::new(Last));
        ws.add_decl(Box::new(Length));
        ws.add_decl(Box::new(Prepend));
        ws.add_decl(Box::new(Reject));
        ws.add_decl(Box::new(Reverse));
        ws.add_decl(Box::new(Select));
        ws.add_decl(Box::new(Skip));
        ws.add_decl(Box::new(Sort));
        ws.add_decl(Box::new(Take));
        ws.add_decl(Box::new(Uniq));
        ws.add_decl(Box::new(Where));
        ws.add_decl(Box::new(Wrap));
        ws.render()
    };
    if let Err(error) = engine.merge_delta(delta) {
        snafu::whatever!("registering the filter set failed: {error}");
    }
    Ok(engine)
}
