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
pub struct ThinkHarnessEvaluator {
    base: r::nu::EngineState,
}

impl ThinkHarnessEvaluator {
    /// Build the base engine once; every evaluation clones it.
    pub fn new() -> HarnessQuestResult<Self> {
        let mut base = nu_cmd_lang::create_default_context();
        base = register_commands(base)?;

        let mut config = base.get_config().as_ref().clone();
        config.use_ansi_coloring = nu_protocol::UseAnsiColoring::False;
        base.set_config(config);

        base.is_interactive = false;
        base.is_login = false;
        base.is_lsp = false;
        base.is_mcp = true;
        base.add_env_var(String::from("PWD"), nu::Value::string("", nu::Span::unknown()));

        let engine = Self { base };
        engine.lock_boundary()?;
        Ok(engine)
    }

    /// Assert the boundary this engine claims, at construction.
    ///
    /// Non-registration is what forbids the denied set, so the day
    /// someone adds the shell context this fails loudly rather than
    /// quietly widening what a sub-turn can reach.
    fn lock_boundary(&self) -> HarnessQuestResult<()> {
        for name in DENIED {
            if self.resolves(name) {
                snafu::whatever!(
                    "`{name}` resolves on the evaluator: the lockdown has been widened"
                );
            }
        }
        for name in PARSE_TIME_LOADERS {
            let head = name.split(' ').next().unwrap_or(name);
            if !(self.resolves(head) || self.resolves(name)) {
                snafu::whatever!(
                    "`{name}` no longer resolves: the parse-time reach this records is gone, \
                     and the sandbox note it justifies wants revisiting"
                );
            }
        }
        Ok(())
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
    ) -> HarnessQuestResult<(String, nu_protocol::Signature, bool)> {
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
        let mut fresh: Vec<(String, nu_protocol::Signature, bool)> = Vec::new();
        for (name, id) in engine.get_decls_sorted(false) {
            let name = String::from_utf8_lossy(&name).to_string();
            if before.contains(&name) {
                continue;
            }
            // `def --env` is a property of the BLOCK rather than a named
            // flag on the signature, so it is unreachable from `named`.
            let decl = engine.get_decl(id);
            let redirects_env = decl
                .block_id()
                .map(|block_id| engine.get_block(block_id).redirect_env)
                .unwrap_or(false);
            fresh.push((name, decl.signature(), redirects_env));
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
    pub fn evaluate(&self, source: &str, input: Option<nu::Value>) -> HarnessQuestResult<nu::Value> {
        let engine = self.base.clone();
        let source = source.to_string();
        let spawned = std::thread::Builder::new()
            .name(String::from("think_harness-eval"))
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
) -> HarnessQuestResult<nu::Value> {
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

/// Add the command families on top of the language core, then merge.
///
/// Registration is an ALLOWLIST, so what is absent is what the sandbox
/// note in the module header is about: nothing here reaches the
/// filesystem, the network or the host, because nothing that could was
/// added. Every command in the families below is a pure value-to-value
/// transform.
fn register_commands(mut engine: r::nu::EngineState) -> HarnessQuestResult<r::nu::EngineState> {
    let delta = {
        let mut ws = r::nu::StateWorkingSet::new(&engine);
        add_filters(&mut ws);
        add_math(&mut ws);
        add_strings(&mut ws);
        add_conversions(&mut ws);
        ws.render()
    };
    if let Err(error) = engine.merge_delta(delta) {
        snafu::whatever!("registering the command set failed: {error}");
    }
    Ok(engine)
}

/// Reshaping a value: the family a mode reaches for first.
///
/// `columns` and `sort-by` were measured in by the data-transform
/// training set - a record's field names and a table's largest row are
/// asked of `evaluate` - the same way `split row` measured the string
/// family in. A trained transform whose head is unregistered dies as an
/// implicit external at serve, so the set follows the syllabus.
fn add_filters(ws: &mut r::nu::StateWorkingSet) {
    use nu_command::{
        Append, Columns, DropColumn, Each, Enumerate, Filter, Find, First, Flatten, Get, Last,
        Length, Prepend, Reject, Reverse, Select, Skip, Sort, SortBy, Take, Uniq, Where, Wrap,
    };
    ws.add_decl(Box::new(Append));
    ws.add_decl(Box::new(Columns));
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
    ws.add_decl(Box::new(SortBy));
    ws.add_decl(Box::new(Take));
    ws.add_decl(Box::new(Uniq));
    ws.add_decl(Box::new(Where));
    ws.add_decl(Box::new(Wrap));
}

/// Arithmetic over a collection, which a checking mode needs to say
/// anything quantitative about the value it was handed.
///
/// The bare `math` head rides along so an emission that reaches for it
/// without a subcommand gets nushell's own guidance rather than a
/// command-not-found the model cannot act on.
fn add_math(ws: &mut r::nu::StateWorkingSet) {
    use nu_command::{
        Math, MathAbs, MathAvg, MathCbrt, MathCeil, MathFloor, MathLog, MathMax, MathMedian,
        MathMin, MathMode, MathProduct, MathRound, MathSqrt, MathStddev, MathSum, MathVariance,
    };
    ws.add_decl(Box::new(Math));
    ws.add_decl(Box::new(MathAbs));
    ws.add_decl(Box::new(MathAvg));
    ws.add_decl(Box::new(MathCbrt));
    ws.add_decl(Box::new(MathCeil));
    ws.add_decl(Box::new(MathFloor));
    ws.add_decl(Box::new(MathLog));
    ws.add_decl(Box::new(MathMax));
    ws.add_decl(Box::new(MathMedian));
    ws.add_decl(Box::new(MathMin));
    ws.add_decl(Box::new(MathMode));
    ws.add_decl(Box::new(MathProduct));
    ws.add_decl(Box::new(MathRound));
    ws.add_decl(Box::new(MathSqrt));
    ws.add_decl(Box::new(MathStddev));
    ws.add_decl(Box::new(MathSum));
    ws.add_decl(Box::new(MathVariance));
}

/// String transforms: the everyday split and str set, all pure.
///
/// The data-transform class routes through `evaluate` by design, and
/// splitting text is its flagship case, so this family is load-bearing
/// rather than a convenience. The niche members beside these - expand,
/// distance, stats, the regex escape - are declined until something
/// asks for them.
fn add_strings(ws: &mut r::nu::StateWorkingSet) {
    use nu_command::{
        Split, SplitChars, SplitColumn, SplitList, SplitRow, SplitWords, Str, StrCapitalize,
        StrContains, StrDowncase, StrEndswith, StrIndexOf, StrJoin, StrLength, StrReplace,
        StrReverse, StrStartsWith, StrSubstring, StrTrim, StrUpcase,
    };
    ws.add_decl(Box::new(Split));
    ws.add_decl(Box::new(SplitChars));
    ws.add_decl(Box::new(SplitColumn));
    ws.add_decl(Box::new(SplitList));
    ws.add_decl(Box::new(SplitRow));
    ws.add_decl(Box::new(SplitWords));
    ws.add_decl(Box::new(Str));
    ws.add_decl(Box::new(StrCapitalize));
    ws.add_decl(Box::new(StrContains));
    ws.add_decl(Box::new(StrDowncase));
    ws.add_decl(Box::new(StrEndswith));
    ws.add_decl(Box::new(StrIndexOf));
    ws.add_decl(Box::new(StrJoin));
    ws.add_decl(Box::new(StrLength));
    ws.add_decl(Box::new(StrReplace));
    ws.add_decl(Box::new(StrReverse));
    ws.add_decl(Box::new(StrStartsWith));
    ws.add_decl(Box::new(StrSubstring));
    ws.add_decl(Box::new(StrTrim));
    ws.add_decl(Box::new(StrUpcase));
}

/// Text interchange in both directions, which is a named capability
/// rather than a convenience: moving a foreign format into NUON and back
/// is one of the things a mode exists to do.
///
/// TEXT FORMATS ONLY. The binary and spreadsheet readers beside these in
/// nushell - msgpack, ods, xlsx - are equally pure and are left out
/// because nothing asks for them, and an allowlist earns its keep by
/// what it declines.
fn add_conversions(ws: &mut r::nu::StateWorkingSet) {
    use nu_command::{
        FROM_YAML, FROM_YML, From, FromCsv, FromJson, FromKdl, FromMd, FromNuon, FromSsv,
        FromToml, FromTsv, FromXml, TO_YAML, TO_YML, To, ToCsv, ToJson, ToKdl, ToMd, ToNuon,
        ToText, ToToml, ToTsv, ToXml,
    };
    ws.add_decl(Box::new(From));
    ws.add_decl(Box::new(FromCsv));
    ws.add_decl(Box::new(FromJson));
    ws.add_decl(Box::new(FromKdl));
    ws.add_decl(Box::new(FromMd));
    ws.add_decl(Box::new(FromNuon));
    ws.add_decl(Box::new(FromSsv));
    ws.add_decl(Box::new(FromToml));
    ws.add_decl(Box::new(FromTsv));
    ws.add_decl(Box::new(FromXml));
    ws.add_decl(Box::new(FROM_YAML));
    ws.add_decl(Box::new(FROM_YML));
    ws.add_decl(Box::new(To));
    ws.add_decl(Box::new(ToCsv));
    ws.add_decl(Box::new(ToJson));
    ws.add_decl(Box::new(ToKdl));
    ws.add_decl(Box::new(ToMd));
    ws.add_decl(Box::new(ToNuon));
    ws.add_decl(Box::new(ToText));
    ws.add_decl(Box::new(ToToml));
    ws.add_decl(Box::new(ToTsv));
    ws.add_decl(Box::new(ToXml));
    ws.add_decl(Box::new(TO_YAML));
    ws.add_decl(Box::new(TO_YML));
}
