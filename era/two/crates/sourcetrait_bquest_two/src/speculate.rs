//! The Speculation:DepthProbe instrument: record plain greedy
//! transcripts (the token streams speculation would ride), then
//! replay them offline through the real lookup index under candidate
//! policies and pass-cost models.
//!
//! Everything data-shaped is whole-value .nuon: the fixture plan in,
//! the per-transcript artifacts out (machine-written, never
//! hand-edited), the report out. Phase annotations ride SIBLING
//! hand-authored .nuon files (spec_phases_<name>.nuon) joined at
//! simulate time.
//!
//! Replay fidelity: the round loop mirrors generate.rs
//! step()/stage_or_speculate() - the emitted token joins the index
//! BEFORE drafting, queue-popped tokens owe no round, the bonus rides
//! as pending, a sample_len end skips the final round while a
//! stop-token end pays it, and accepted drafts join the index at
//! round close. One conservative bias: a draft that would accept the
//! STOP token itself (echo-shaped turn tails) scores as a rejection
//! here because stops are never emitted into the recorded stream - at
//! most one round per stop-ended turn reads one pass heavy.
use crate::*;

/// The fixture plan shape (`--fixtures`).
const FIXTURE_TYPEDEF: &str =
    "table<name: string, kind: string, turns: list<string>, budget: int>";

/// The recorded-transcript artifact shape (spec_transcript_<name>.nuon).
const TRANSCRIPT_TYPEDEF: &str = "record<name: string, kind: string, \
     meta: record<model: string, settings: string, device: string, \
     bquest_version: string, recorded_at: int>, \
     turns: table<index: int, prompt: string, text: string, \
     suffix_ids: list<int>, emitted_ids: list<int>, consumed: int, \
     finish: string, prefill_seconds: float, decode_seconds: float>>";

/// The hand-authored phase-annotation shape (spec_phases_<name>.nuon);
/// `end` is exclusive, -1 = to the turn's end; token indices index
/// emitted_ids.
const PHASES_TYPEDEF: &str = "table<turn: int, start: int, end: int, phase: string>";

/// Gated-off rounds re-probe every this-many suppressions so a cold
/// gate can observe recovery (the EMA only updates on drafted rounds).
const RECOVERY_PROBE: usize = 32;

/// EMA weight for the gated policies' accept-depth tracking.
const EMA_ALPHA: f64 = 0.25;

/// One fixture-plan row.
struct TranscriptPlan {
    name: String,
    kind: String,
    turns: Vec<String>,
    budget: usize,
}

fn read_plans(value: &lib::nu::Value) -> BquestResult<Vec<TranscriptPlan>> {
    let rows = match value.as_list() {
        Ok(list) => list,
        Err(e) => snafu::whatever!("the fixture plan must be a table: {e}"),
    };
    let mut plans = Vec::with_capacity(rows.len());
    for row in rows {
        let record = match row.as_record() {
            Ok(record) => record,
            Err(e) => snafu::whatever!("fixture row: {e}"),
        };
        plans.push(TranscriptPlan {
            name: field_str(record, "name")?,
            kind: field_str(record, "kind")?,
            turns: field_strings(record, "turns")?,
            budget: field_int(record, "budget")? as usize,
        });
    }
    Ok(plans)
}

pub(crate) fn speculate_record(cli: &Cli, args: &SpeculateRecordArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let config = lib::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let settings = lib::LibSettings::load_from_dir(cli.dir.as_ref(), cli.settings.as_deref())?;
    let settings_token = cli.settings.clone().unwrap_or_else(|| String::from("default"));

    let plan_value = lib::nu::load_value(&args.fixtures)?;
    lib::nu::conform(&plan_value, &lib::nu::parse_typedef(FIXTURE_TYPEDEF)?)?;
    let plans = read_plans(&plan_value)?;
    let only: Option<Vec<String>> = args
        .only
        .as_ref()
        .map(|csv| csv.split(',').map(|t| t.trim().to_string()).collect());

    #[cfg(feature = "cuda")]
    let (device, dtype, device_label) =
        (candle_core::Device::new_cuda(0)?, candle_core::DType::BF16, "cuda_bf16");
    #[cfg(not(feature = "cuda"))]
    let (device, dtype, device_label) =
        (candle_core::Device::Cpu, candle_core::DType::F32, "cpu_f32");

    let model_dir = config.model_dir();
    let checkpoint = lib::load_config(&model_dir)?;
    let tokenizer = lib::load_tokenizer(&model_dir)?;
    lib::verify_token_map(&tokenizer)?;
    let weights = lib::load_weights(&config, dtype, &device)?;
    let mut model = lib::OlmoHybrid::new(&checkpoint, settings, weights)?;
    eprintln!(
        "speculate record: model {} loaded ({:.1}s, {device_label})",
        config.model,
        started.elapsed().as_secs_f64()
    );

    let mut recorded = 0usize;
    for plan in &plans {
        if let Some(filter) = &only
            && !filter.contains(&plan.name)
        {
            continue;
        }
        let transcript = record_transcript(
            &mut model,
            &tokenizer,
            plan,
            &config.model,
            &settings_token,
            device_label,
        )?;
        let path = args.out.join(format!("spec_transcript_{}.nuon", plan.name));
        lib::nu::save_value(&path, &transcript)?;
        recorded += 1;
        eprintln!("speculate record: {} written", path.display());
    }
    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "recorded" => v_int(recorded as i64),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
            "out" => v_str(&args.out.display().to_string()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// Record one plan's turns: plain greedy (the depth statistics are
/// greedy-deterministic), chat-rendered, natural stops honored under
/// the per-turn budget; later turns chain live through generate_from
/// (the trail IS the restored-context contract).
fn record_transcript(
    model: &mut lib::OlmoHybrid,
    tokenizer: &tokenizers::Tokenizer,
    plan: &TranscriptPlan,
    model_id: &str,
    settings_token: &str,
    device_label: &str,
) -> BquestResult<lib::nu::Value> {
    let mut turn_rows: Vec<lib::nu::Value> = Vec::new();
    let mut trail: Vec<u32> = Vec::new();
    for (index, prompt) in plan.turns.iter().enumerate() {
        let options = lib::GenerateOptions {
            temperature: None,
            top_p: None,
            sample_len: plan.budget,
            chat: true,
            ignore_stops: false,
            speculate: false,
            ..lib::GenerateOptions::default()
        };
        let context_before = trail.len();
        let turn_started = std::time::Instant::now();
        let mut generation = if index == 0 {
            model.generate(tokenizer, prompt, &options)?
        } else {
            let restored = lib::RestoredContext {
                context_len: model.context_len(),
                context_ids: trail.clone(),
            };
            model.generate_from(tokenizer, &restored, prompt, &options)?
        };
        let mut emitted: Vec<u32> = Vec::new();
        let mut text = String::new();
        for step in generation.by_ref() {
            let step = step?;
            emitted.push(step.token_id);
            text.push_str(&step.chunk);
        }
        let report = generation.finish();
        text.push_str(&report.rest);
        let finish = match report.finish_reason {
            Some(lib::FinishReason::StopToken) => "stop_token",
            Some(lib::FinishReason::SampleLen) => "sample_len",
            None => "early_stop",
        };
        let suffix_ids =
            report.context_ids[context_before..context_before + report.prompt_token_count].to_vec();
        let consumed = report.context_ids.len() - context_before - report.prompt_token_count;
        eprintln!(
            "speculate record: {} turn {index}: {} suffix + {} emitted ({finish}, {:.1}s)",
            plan.name,
            suffix_ids.len(),
            emitted.len(),
            turn_started.elapsed().as_secs_f64()
        );
        turn_rows.push(lib::nu::Value::record(
            lib::nu::record! {
                "index" => v_int(index as i64),
                "prompt" => v_str(prompt),
                "text" => v_str(&text),
                "suffix_ids" => v_int_list(&suffix_ids),
                "emitted_ids" => v_int_list(&emitted),
                "consumed" => v_int(consumed as i64),
                "finish" => v_str(finish),
                "prefill_seconds" => v_float(report.prefill_seconds),
                "decode_seconds" => v_float(report.decode_seconds),
            },
            span(),
        ));
        trail = report.context_ids;
    }
    let recorded_at = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(elapsed) => elapsed.as_secs() as i64,
        Err(_) => 0,
    };
    Ok(lib::nu::Value::record(
        lib::nu::record! {
            "name" => v_str(&plan.name),
            "kind" => v_str(&plan.kind),
            "meta" => lib::nu::Value::record(
                lib::nu::record! {
                    "model" => v_str(model_id),
                    "settings" => v_str(settings_token),
                    "device" => v_str(device_label),
                    "bquest_version" => v_str(env!("CARGO_PKG_VERSION")),
                    "recorded_at" => v_int(recorded_at),
                },
                span(),
            ),
            "turns" => lib::nu::Value::list(turn_rows, span()),
        },
        span(),
    ))
}

/// One recorded turn, replay-ready.
pub(crate) struct RecordedTurn {
    pub(crate) suffix_ids: Vec<u32>,
    pub(crate) emitted_ids: Vec<u32>,
    pub(crate) consumed: usize,
    pub(crate) finish: String,
}

pub(crate) struct RecordedTranscript {
    pub(crate) name: String,
    pub(crate) kind: String,
    pub(crate) turns: Vec<RecordedTurn>,
}

fn read_transcript(path: &Path) -> BquestResult<RecordedTranscript> {
    let value = lib::nu::load_value(path)?;
    lib::nu::conform(&value, &lib::nu::parse_typedef(TRANSCRIPT_TYPEDEF)?)?;
    let record = match value.as_record() {
        Ok(record) => record,
        Err(e) => snafu::whatever!("transcript {}: {e}", path.display()),
    };
    let mut turns = Vec::new();
    for row in field_rows(record, "turns")? {
        turns.push(RecordedTurn {
            suffix_ids: field_ids(row, "suffix_ids")?,
            emitted_ids: field_ids(row, "emitted_ids")?,
            consumed: field_int(row, "consumed")? as usize,
            finish: field_str(row, "finish")?,
        });
    }
    Ok(RecordedTranscript {
        name: field_str(record, "name")?,
        kind: field_str(record, "kind")?,
        turns,
    })
}

/// One hand-authored phase-annotation row.
pub(crate) struct PhaseRow {
    pub(crate) turn: usize,
    pub(crate) start: i64,
    pub(crate) end: i64,
    pub(crate) phase: String,
}

fn read_phases(path: &Path) -> BquestResult<Vec<PhaseRow>> {
    let value = lib::nu::load_value(path)?;
    lib::nu::conform(&value, &lib::nu::parse_typedef(PHASES_TYPEDEF)?)?;
    let rows = match value.as_list() {
        Ok(list) => list,
        Err(e) => snafu::whatever!("phases {}: {e}", path.display()),
    };
    let mut phases = Vec::with_capacity(rows.len());
    for row in rows {
        let record = match row.as_record() {
            Ok(record) => record,
            Err(e) => snafu::whatever!("phases row: {e}"),
        };
        phases.push(PhaseRow {
            turn: field_int(record, "turn")? as usize,
            start: field_int(record, "start")?,
            end: field_int(record, "end")?,
            phase: field_str(record, "phase")?,
        });
    }
    Ok(phases)
}

/// A prose-family kind defaults every token to the prose phase;
/// anything else is unlabeled until its phases file lands.
fn default_phase(kind: &str) -> &'static str {
    if kind.starts_with("prose") {
        "prose"
    } else {
        "unlabeled"
    }
}

fn phase_label(
    rows: &[PhaseRow],
    default_label: &str,
    turn: usize,
    token_index: usize,
) -> String {
    for row in rows {
        if row.turn == turn
            && (token_index as i64) >= row.start
            && (row.end < 0 || (token_index as i64) < row.end)
        {
            return row.phase.clone();
        }
    }
    String::from(default_label)
}

/// The pass cost of a PARTIAL/REJECTED verification round: 2 in v1 as
/// built (verify + GDN shadow replay); 1 under a kernel-managed
/// accept (the fla/vllm mechanism). Full accepts and plain steps are
/// 1 pass in both.
#[derive(Clone, Copy)]
pub(crate) struct CostModel {
    pub(crate) name: &'static str,
    pub(crate) partial_passes: u64,
}

pub(crate) const COST_MODELS: [CostModel; 2] = [
    CostModel { name: "two_pass_v1", partial_passes: 2 },
    CostModel { name: "one_pass_kernel_accept", partial_passes: 1 },
];

/// The candidate policy shapes over the real v1 DraftPolicy.
#[derive(Clone, Copy)]
pub(crate) enum PolicyKind {
    /// The shipped policy verbatim.
    V1,
    /// The 2-gram floor structurally excluded (its hits never draft).
    NoTwoGram,
    /// Draft only while the accept-depth EMA clears the threshold.
    EmaGated { threshold: f64 },
    /// Per-matched-level accept-depth EMAs, each gated independently.
    LevelGated { threshold: f64 },
}

pub(crate) struct PolicyState {
    kind: PolicyKind,
    inner: lib::DraftPolicy,
    ema: f64,
    level_ema: [f64; 5],
    gate_skips: usize,
    level_skips: [usize; 5],
}

impl PolicyState {
    pub(crate) fn new(kind: PolicyKind) -> Self {
        Self {
            kind,
            inner: lib::DraftPolicy::new(),
            // Hot starts: gates open until observed depth argues
            // otherwise.
            ema: 2.0,
            level_ema: [2.0; 5],
            gate_skips: 0,
            level_skips: [0; 5],
        }
    }

    pub(crate) fn name(kind: PolicyKind) -> String {
        match kind {
            PolicyKind::V1 => String::from("v1"),
            PolicyKind::NoTwoGram => String::from("no_2gram"),
            PolicyKind::EmaGated { threshold } => format!("ema_gated_{threshold}"),
            PolicyKind::LevelGated { threshold } => format!("level_gated_{threshold}"),
        }
    }

    /// The round's draft, mirroring stage_or_speculate's derivation
    /// (probe_limit -> index.draft -> draft_limit truncation), with
    /// the candidate gate applied after the match. Gated-off rounds
    /// re-probe every RECOVERY_PROBE suppressions.
    fn round_draft(
        &mut self,
        index: &lib::LookupIndex,
        budget: usize,
    ) -> Option<(usize, Vec<u32>)> {
        let probe = self.inner.probe_limit();
        let (matched_n, mut tokens) = index.draft(probe.min(budget))?;
        tokens.truncate(self.inner.draft_limit(matched_n).min(budget));
        if tokens.is_empty() {
            return None;
        }
        let allowed = match self.kind {
            PolicyKind::V1 => true,
            PolicyKind::NoTwoGram => matched_n >= 3,
            PolicyKind::EmaGated { threshold } => {
                if self.ema >= threshold {
                    true
                } else {
                    self.gate_skips += 1;
                    if self.gate_skips >= RECOVERY_PROBE {
                        self.gate_skips = 0;
                        true
                    } else {
                        false
                    }
                }
            }
            PolicyKind::LevelGated { threshold } => {
                if self.level_ema[matched_n] >= threshold {
                    true
                } else {
                    self.level_skips[matched_n] += 1;
                    if self.level_skips[matched_n] >= RECOVERY_PROBE {
                        self.level_skips[matched_n] = 0;
                        true
                    } else {
                        false
                    }
                }
            }
        };
        if allowed { Some((matched_n, tokens)) } else { None }
    }

    fn record(&mut self, matched_n: usize, accepted: usize) {
        self.inner.record(accepted);
        self.ema = (1.0 - EMA_ALPHA) * self.ema + EMA_ALPHA * accepted as f64;
        self.level_ema[matched_n] =
            (1.0 - EMA_ALPHA) * self.level_ema[matched_n] + EMA_ALPHA * accepted as f64;
    }
}

pub(crate) fn candidate_policies() -> Vec<PolicyKind> {
    vec![
        PolicyKind::V1,
        PolicyKind::NoTwoGram,
        PolicyKind::EmaGated { threshold: 0.5 },
        PolicyKind::EmaGated { threshold: 1.0 },
        PolicyKind::EmaGated { threshold: 1.5 },
        PolicyKind::LevelGated { threshold: 0.5 },
        PolicyKind::LevelGated { threshold: 1.0 },
        PolicyKind::LevelGated { threshold: 1.5 },
    ]
}

#[derive(Default, Clone, Copy)]
pub(crate) struct PhaseTally {
    pub(crate) tokens: u64,
    pub(crate) rounds: u64,
    pub(crate) drafted: u64,
    pub(crate) accepted: u64,
    pub(crate) passes: u64,
}

/// Replay one transcript under one policy and one cost model,
/// tallying per phase label. A plain greedy decode reads exactly one
/// pass per token, so tokens_per_pass is the against-plain ratio
/// directly.
pub(crate) fn replay_transcript(
    transcript: &RecordedTranscript,
    phases: &[PhaseRow],
    policy_kind: PolicyKind,
    cost: &CostModel,
) -> HashMap<String, PhaseTally> {
    let default_label = default_phase(&transcript.kind);
    let mut tallies: HashMap<String, PhaseTally> = HashMap::new();
    let mut policy = PolicyState::new(policy_kind);
    let mut trail: Vec<u32> = Vec::new();
    for (turn_index, turn) in transcript.turns.iter().enumerate() {
        // Per-generation index, seeded like start_generation: the
        // committed trail plus this turn's rendered suffix.
        let mut index = lib::LookupIndex::new();
        index.extend(&trail);
        index.extend(&turn.suffix_ids);
        let emitted = &turn.emitted_ids;
        let total = emitted.len();
        let sample_len_end = turn.finish == "sample_len";
        let mut queue: VecDeque<u32> = VecDeque::new();
        let mut pending = emitted.first().copied();
        let mut position = 0usize;
        loop {
            let (token, from_queue) = match queue.pop_front() {
                Some(token) => (token, true),
                None => match pending.take() {
                    Some(token) => (token, false),
                    None => break,
                },
            };
            debug_assert_eq!(token, emitted[position]);
            let label = phase_label(phases, default_label, turn_index, position);
            position += 1;
            tallies.entry(label.clone()).or_default().tokens += 1;
            // The budget check precedes the round (a sample_len end
            // never pays a final round; a stop-token end does).
            if position >= total && sample_len_end {
                break;
            }
            if from_queue {
                continue;
            }
            let tally = tallies.entry(label).or_default();
            tally.rounds += 1;
            index.extend(&[token]);
            let budget = total - position;
            let draft = if budget > 0 {
                policy.round_draft(&index, budget)
            } else {
                None
            };
            match draft {
                None => {
                    tally.passes += 1;
                    pending = if position < total {
                        Some(emitted[position])
                    } else {
                        None
                    };
                }
                Some((matched_n, tokens)) => {
                    tally.drafted += tokens.len() as u64;
                    let mut accepted = 0usize;
                    while accepted < tokens.len()
                        && position + accepted < total
                        && tokens[accepted] == emitted[position + accepted]
                    {
                        accepted += 1;
                    }
                    let full = accepted == tokens.len();
                    tally.passes += if full { 1 } else { cost.partial_passes };
                    tally.accepted += accepted as u64;
                    policy.record(matched_n, accepted);
                    queue.extend(tokens[..accepted].iter().copied());
                    index.extend(&tokens[..accepted]);
                    pending = if position + accepted < total {
                        Some(emitted[position + accepted])
                    } else {
                        None
                    };
                }
            }
        }
        trail.extend_from_slice(&turn.suffix_ids);
        trail.extend_from_slice(&turn.emitted_ids[..turn.consumed]);
    }
    tallies
}

pub(crate) fn speculate_simulate(args: &SpeculateSimulateArgs) -> BquestResult<()> {
    let mut transcript_files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&args.transcripts)? {
        let path = entry?.path();
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("spec_transcript_") && n.ends_with(".nuon"))
        {
            transcript_files.push(path);
        }
    }
    transcript_files.sort();
    snafu::ensure_whatever!(
        !transcript_files.is_empty(),
        "no spec_transcript_*.nuon under {}",
        args.transcripts.display()
    );

    let mut result_rows: Vec<lib::nu::Value> = Vec::new();
    // (policy, cost) -> (prose tokens/passes, all tokens/passes).
    let mut aggregates: HashMap<String, (u64, u64, u64, u64)> = HashMap::new();
    let mut summary_keys: Vec<(String, String)> = Vec::new();
    for file in &transcript_files {
        let transcript = read_transcript(file)?;
        let phases_path = args
            .transcripts
            .join(format!("spec_phases_{}.nuon", transcript.name));
        let phases = if phases_path.is_file() {
            read_phases(&phases_path)?
        } else {
            Vec::new()
        };
        let annotated = if phases.is_empty() { "default" } else { "annotated" };
        eprintln!(
            "speculate simulate: {} ({} turns, {annotated} phases)",
            transcript.name,
            transcript.turns.len()
        );
        for policy_kind in candidate_policies() {
            let policy_name = PolicyState::name(policy_kind);
            for cost in &COST_MODELS {
                let tallies = replay_transcript(&transcript, &phases, policy_kind, cost);
                let key = format!("{policy_name}\u{1f}{}", cost.name);
                if !aggregates.contains_key(&key) {
                    summary_keys.push((policy_name.clone(), cost.name.to_string()));
                }
                let aggregate = aggregates.entry(key).or_default();
                let mut labels: Vec<&String> = tallies.keys().collect();
                labels.sort();
                for label in labels {
                    let tally = tallies[label];
                    let tokens_per_pass = if tally.passes == 0 {
                        1.0
                    } else {
                        tally.tokens as f64 / tally.passes as f64
                    };
                    if label == "prose" {
                        aggregate.0 += tally.tokens;
                        aggregate.1 += tally.passes;
                    }
                    aggregate.2 += tally.tokens;
                    aggregate.3 += tally.passes;
                    result_rows.push(lib::nu::Value::record(
                        lib::nu::record! {
                            "transcript" => v_str(&transcript.name),
                            "kind" => v_str(&transcript.kind),
                            "policy" => v_str(&policy_name),
                            "cost_model" => v_str(cost.name),
                            "phase" => v_str(label),
                            "tokens" => v_int(tally.tokens as i64),
                            "rounds" => v_int(tally.rounds as i64),
                            "drafted" => v_int(tally.drafted as i64),
                            "accepted" => v_int(tally.accepted as i64),
                            "passes" => v_int(tally.passes as i64),
                            "tokens_per_pass" => v_float(tokens_per_pass),
                        },
                        span(),
                    ));
                }
            }
        }
    }

    let mut summary_rows: Vec<lib::nu::Value> = Vec::new();
    for (policy_name, cost_name) in &summary_keys {
        let key = format!("{policy_name}\u{1f}{cost_name}");
        let (prose_tokens, prose_passes, all_tokens, all_passes) = aggregates[&key];
        let prose_tpp = if prose_passes == 0 {
            1.0
        } else {
            prose_tokens as f64 / prose_passes as f64
        };
        let overall_tpp = if all_passes == 0 {
            1.0
        } else {
            all_tokens as f64 / all_passes as f64
        };
        summary_rows.push(lib::nu::Value::record(
            lib::nu::record! {
                "policy" => v_str(policy_name),
                "cost_model" => v_str(cost_name),
                "prose_tokens" => v_int(prose_tokens as i64),
                "prose_tokens_per_pass" => v_float(prose_tpp),
                "overall_tokens_per_pass" => v_float(overall_tpp),
                "prose_floor_holds" => v_bool(prose_tpp >= 1.0 - 1e-9),
            },
            span(),
        ));
    }

    let report = lib::nu::Value::record(
        lib::nu::record! {
            "transcripts_dir" => v_str(&args.transcripts.display().to_string()),
            "results" => lib::nu::Value::list(result_rows, span()),
            "summary" => lib::nu::Value::list(summary_rows.clone(), span()),
        },
        span(),
    );
    lib::nu::save_value(&args.out, &report)?;
    eprintln!("speculate simulate: report -> {}", args.out.display());
    let stdout_summary = lib::nu::Value::list(summary_rows, span());
    println!("{}", lib::nu::to_nuon_text(&stdout_summary)?);
    Ok(())
}

/// The phase-annotation aid: emitted tokens as indexed decoded
/// pieces, one line each (turn:index, id, piece).
pub(crate) fn speculate_tokens(cli: &Cli, args: &SpeculateTokensArgs) -> BquestResult<()> {
    let config = lib::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let tokenizer = lib::load_tokenizer(&config.model_dir())?;
    let path = args
        .transcripts
        .join(format!("spec_transcript_{}.nuon", args.name));
    let transcript = read_transcript(&path)?;
    for (turn_index, turn) in transcript.turns.iter().enumerate() {
        if let Some(only) = args.turn
            && only != turn_index
        {
            continue;
        }
        for (position, id) in turn.emitted_ids.iter().enumerate() {
            let piece = match tokenizer.decode(&[*id], false) {
                Ok(piece) => piece,
                Err(e) => snafu::whatever!("decode failed at {turn_index}:{position}: {e}"),
            };
            println!("{turn_index}:{position}\t{id}\t{piece:?}");
        }
    }
    Ok(())
}
