//! Sampled generation graded by a mechanical verifier - the data the
//! reinforcement stage learns from.
//!
//! Rollouts ride the CANDLE engine rather than the burn oracle: the
//! oracle is a stateless full-sequence forward with no cache, so
//! generating a reply of any length through it would replay the whole
//! prefix per token. The engine has the decode path already.
//!
//! ## DEV
//! Rollout and training are separate VERBS, deliberately, rather than
//! two phases inside one loop. Both stacks hold a full copy of the
//! weights, and two resident copies do not fit the card - so the loop
//! is driven from outside, one process at a time: generate, score,
//! step, repeat. That also makes the scored artifact a real
//! checkpoint rather than a value in flight.
//!
//! The policy being sampled is whatever adapter the config names, so
//! an on-policy loop points the config at the adapter the previous
//! step wrote.
//! ##
use crate::*;

/// One graded reply.
const ROLLOUT_TYPEDEF: &str = "table<prompt_index: int, prompt_ids: list<int>, \
     response_ids: list<int>, response: string, reward: float, verifier: string>";

/// One rollout ready for training: the padded window, its response
/// mask, and the verifier's reward.
pub(crate) type PackedRollout = (Vec<u32>, Vec<u8>, f32);
/// Rollouts bucketed by the prompt they answer - the unit a
/// group-relative advantage is computed over.
pub(crate) type RolloutGroups = Vec<Vec<PackedRollout>>;

/// `bquest rollout run`: sample `group` replies per verifiable
/// prompt, grade each, and write the scored table.
pub(crate) fn rollout_run(cli: &Cli, args: &RolloutRunArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    snafu::ensure_whatever!(
        args.group >= 2,
        "a group of {} carries no relative signal - advantages are computed \
         within a group, so a single sample is always exactly average",
        args.group
    );
    let config = lib::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let settings = lib::LibSettings::load_from_dir(cli.dir.as_ref(), cli.settings.as_deref())?;
    let prompts = example::load_rlvr(&args.prompts)?;

    #[cfg(feature = "cuda")]
    let (device, dtype) = (candle_core::Device::new_cuda(0)?, candle_core::DType::BF16);
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let model_dir = config.model_dir();
    let checkpoint = lib::load_config(&model_dir)?;
    let tokenizer = lib::load_tokenizer(&model_dir)?;
    lib::verify_token_map(&tokenizer)?;
    let weights = lib::load_weights(&config, dtype, &device)?;
    let mut model = lib::OlmoHybrid::new(&checkpoint, settings, weights)?;
    let mut rows: Vec<lib::nu::Value> = Vec::new();
    let mut passed = 0usize;
    let mut attempted = 0usize;

    for (prompt_index, prompt) in prompts.iter().enumerate() {
        let prompt_ids = example::encode_prompt(&tokenizer, &prompt.prompt)?;
        let rendered = lib::chat_render(&prompt.prompt, true).text;
        for member in 0..args.group {
            let options = lib::GenerateOptions {
                temperature: Some(args.temperature),
                top_p: None,
                // Each member draws its own stream, or the group is
                // one reply repeated and every advantage is zero.
                seed: args.seed.wrapping_add((prompt_index * args.group + member) as u64),
                sample_len: args.max_tokens,
                chat: false,
                ignore_stops: false,
                speculate: false,
            };
            let mut generation = model.generate(&tokenizer, &rendered, &options)?;
            let mut response = String::new();
            for step in generation.by_ref() {
                response.push_str(&step?.chunk);
            }
            let report = generation.finish();
            response.push_str(&report.rest);
            let response_ids = report.context_ids[prompt_ids.len().min(report.context_ids.len())..]
                .to_vec();

            let reward = example::verify(&prompt.verifier, &response, &prompt.reference)?;
            attempted += 1;
            if reward > 0.0 {
                passed += 1;
            }
            rows.push(lib::nu::Value::record(
                lib::nu::record! {
                    "prompt_index" => v_int(prompt_index as i64),
                    "prompt_ids" => v_int_list(&prompt_ids),
                    "response_ids" => v_int_list(&response_ids),
                    "response" => v_str(&response),
                    "reward" => v_float(reward as f64),
                    "verifier" => v_str(&prompt.verifier),
                },
                span(),
            ));
        }
        eprintln!(
            "bquest rollout: prompt {}/{} - {passed}/{attempted} passing so far",
            prompt_index + 1,
            prompts.len()
        );
    }

    let table = lib::nu::Value::list(rows, span());
    lib::nu::conform(&table, &lib::nu::parse_typedef(&format!("list<{ROLLOUT_TYPEDEF}>"))?)?;
    lib::nu::save_value(&args.out, &table)?;

    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "prompts" => v_int(prompts.len() as i64),
            "group" => v_int(args.group as i64),
            "rollouts" => v_int(attempted as i64),
            "passing" => v_int(passed as i64),
            "pass_rate" => v_float(passed as f64 / attempted.max(1) as f64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// Read a scored-rollout artifact back into per-prompt groups, each
/// row padded to `width` with its response positions masked.
///
/// A rollout longer than the window is DROPPED rather than clipped: a
/// truncated reply would be graded on text the model did not finish
/// producing.
pub(crate) fn load_groups(
    path: &Path,
    width: usize,
    pad_id: u32,
) -> BquestResult<(RolloutGroups, usize)> {
    let value = lib::nu::load_value(path)?;
    lib::nu::conform(&value, &lib::nu::parse_typedef(&format!("list<{ROLLOUT_TYPEDEF}>"))?)?;
    let rows = match value.as_list() {
        Ok(rows) => rows,
        Err(e) => snafu::whatever!("{}: not a rollout table: {e}", path.display()),
    };
    let mut grouped: RolloutGroups = Vec::new();
    let mut index_of: HashMap<i64, usize> = HashMap::new();
    let mut dropped = 0usize;
    for row in rows {
        let record = match row.as_record() {
            Ok(record) => record,
            Err(e) => snafu::whatever!("rollout row: {e}"),
        };
        let prompt_ids = field_ids(record, "prompt_ids")?;
        let response_ids = field_ids(record, "response_ids")?;
        let reward = match field(record, "reward")?.as_float() {
            Ok(reward) => reward as f32,
            Err(e) => snafu::whatever!("rollout reward: {e}"),
        };
        if response_ids.is_empty() {
            dropped += 1;
            continue;
        }
        let mut ids = prompt_ids.clone();
        ids.extend_from_slice(&response_ids);
        let mut mask = vec![0u8; prompt_ids.len()];
        mask.extend(std::iter::repeat_n(1u8, response_ids.len()));
        let Some(packed) = example::pack_row(ids, mask, width, pad_id) else {
            dropped += 1;
            continue;
        };
        let key = field_int(record, "prompt_index")?;
        let slot = match index_of.get(&key) {
            Some(slot) => *slot,
            None => {
                grouped.push(Vec::new());
                index_of.insert(key, grouped.len() - 1);
                grouped.len() - 1
            }
        };
        grouped[slot].push((packed.ids, packed.mask, reward));
    }
    grouped.retain(|group| group.len() >= 2);
    snafu::ensure_whatever!(
        !grouped.is_empty(),
        "{}: no prompt kept two or more rollouts, so no group carries a \
         relative signal",
        path.display()
    );
    Ok((grouped, dropped))
}
