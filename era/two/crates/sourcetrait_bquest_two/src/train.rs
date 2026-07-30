//! The training verbs: thin drivers over the llm lib's trainer core.
use crate::*;

#[cfg(feature = "train-cuda")]
use llm::train::{
    EncodedPair,
    RolloutGroup,
    ScoredRollout,
    SupervisedBatch,
};
use llm::train::{
    LoopOptions,
    StepLog,
    TrainReport,
};

/// A loaded pack: the id rows and an optional per-position loss mask.
pub(crate) struct ChunkPack {
    pub(crate) rows: Vec<Vec<u32>>,
    pub(crate) masks: Option<Vec<Vec<u8>>>,
}

/// Read a packed artifact: u32 ids and an optional u8 loss mask.
fn load_chunks(path: &Path) -> BquestResult<ChunkPack> {
    let bytes = fs::read(path)?;
    let parsed = match safetensors::SafeTensors::deserialize(&bytes) {
        Ok(parsed) => parsed,
        Err(error) => snafu::whatever!("chunks parse failed: {error}"),
    };
    for name in parsed.names() {
        snafu::ensure_whatever!(
            name == example::TENSOR_IDS || name == example::TENSOR_LOSS_MASK,
            "pack carries unrecognised tensor {name:?} \
             (expected {:?} and optionally {:?})",
            example::TENSOR_IDS,
            example::TENSOR_LOSS_MASK
        );
    }
    let Ok(view) = parsed.tensor(example::TENSOR_IDS) else {
        snafu::whatever!("the artifact carries no {:?} tensor", example::TENSOR_IDS);
    };
    snafu::ensure_whatever!(
        view.dtype() == safetensors::Dtype::U32,
        "ids: expected u32, got {:?}",
        view.dtype()
    );
    let shape = view.shape();
    snafu::ensure_whatever!(shape.len() == 2, "ids: expected rank 2, got {shape:?}");
    let (row_count, width) = (shape[0], shape[1]);
    let values: Vec<u32> = view
        .data()
        .chunks_exact(4)
        .map(|quad| u32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
        .collect();
    let rows: Vec<Vec<u32>> = (0..row_count)
        .map(|row| values[row * width..(row + 1) * width].to_vec())
        .collect();

    let masks = match parsed.tensor(example::TENSOR_LOSS_MASK) {
        Ok(mask_view) => {
            snafu::ensure_whatever!(
                mask_view.dtype() == safetensors::Dtype::U8,
                "loss mask: expected u8, got {:?}",
                mask_view.dtype()
            );
            snafu::ensure_whatever!(
                mask_view.shape() == [row_count, width],
                "loss mask shape {:?} does not match the ids [{row_count}, {width}]",
                mask_view.shape()
            );
            let flat = mask_view.data().to_vec();
            Some(
                (0..row_count)
                    .map(|row| flat[row * width..(row + 1) * width].to_vec())
                    .collect(),
            )
        }
        Err(_) => None,
    };
    Ok(ChunkPack { rows, masks })
}

/// `bquest train gate`: run the locks and print the NUON verdict.
pub(crate) fn train_gate_verb() -> BquestResult<()> {
    let verdict = llm::train::run_train_gate()?;
    let passed = verdict.adapter_off_exact
        && verdict.init_grads.max_nmse <= llm::train::GATE_GRAD_NMSE_BAR
        && verdict.trained_grads.max_nmse <= llm::train::GATE_GRAD_NMSE_BAR
        && verdict.descended;
    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "passed" => v_bool(passed),
            "adapter_off_exact" => v_bool(verdict.adapter_off_exact),
            "init_grad_nmse_max" => v_float(verdict.init_grads.max_nmse),
            "init_grads_compared" => v_int(verdict.init_grads.compared as i64),
            "init_grads_zero_matched" => v_int(verdict.init_grads.zero_matched as i64),
            "trained_grad_nmse_max" => v_float(verdict.trained_grads.max_nmse),
            "trained_grads_compared" => v_int(verdict.trained_grads.compared as i64),
            "grad_nmse_bar" => v_float(llm::train::GATE_GRAD_NMSE_BAR),
            "first_loss" => v_float(verdict.first_loss as f64),
            "final_loss" => v_float(verdict.final_loss as f64),
            "descended" => v_bool(verdict.descended),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    snafu::ensure_whatever!(passed, "the train gate failed");
    Ok(())
}

/// The step log's default home: beside the adapter it belongs to.
#[cfg(feature = "train-cuda")]
fn stage_log_path(args: &StageArgs) -> PathBuf {
    match &args.log {
        Some(path) => path.clone(),
        None => {
            let stem = args
                .out
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| String::from("adapter"));
            args.out.with_file_name(format!("{stem}_steps.nuon"))
        }
    }
}

/// The cpt log home, from its own args shape.
fn cpt_log_path(out: &Path, log: &Option<PathBuf>) -> PathBuf {
    match log {
        Some(path) => path.clone(),
        None => {
            let stem = out
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| String::from("adapter"));
            out.with_file_name(format!("{stem}_steps.nuon"))
        }
    }
}

/// Append one step's NUON-lines row to the step log.
fn append_step_log(log_path: &Path, log: &StepLog) -> BquestResult<()> {
    let row = harness::nu::Value::record(
        harness::nu::record! {
            "step" => v_int(log.step as i64),
            "loss" => v_float(log.loss as f64),
            "learning_rate" => v_float(log.learning_rate),
            "tokens_per_second" => v_float(log.tokens_per_second),
            "elapsed_seconds" => v_float(log.elapsed_seconds),
        },
        span(),
    );
    harness::nu::append_line(log_path, &row)?;
    Ok(())
}

/// The closing record every stage prints.
fn print_stage_summary(
    stage: &str,
    report: &TrainReport,
    out: &Path,
    log_path: &Path,
) -> BquestResult<()> {
    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "stage" => v_str(stage),
            "steps" => v_int(report.steps as i64),
            "first_loss" => v_float(report.first_loss as f64),
            "final_loss" => v_float(report.final_loss as f64),
            "trained_tokens" => v_int(report.trained_tokens as i64),
            "seconds" => v_float(report.seconds),
            "out" => v_str(&out.display().to_string()),
            "log" => v_str(&log_path.display().to_string()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// Fresh adapters, or the previous stage's continued.
#[cfg(feature = "train-cuda")]
fn stage_adapters(
    stage: &StageArgs,
    config: &HybridCheckpointConfig,
    model_id: &str,
    device: &llm::CudaDevice,
) -> BquestResult<ModelAdapters<llm::TrainCudaAd>> {
    match &stage.resume {
        Some(path) => {
            eprintln!("bquest: resuming from adapter {}", path.display());
            Ok(ModelAdapters::<llm::TrainCudaAd>::load(path, config, model_id, device)?)
        }
        None => {
            let alpha = stage.alpha.unwrap_or(2.0 * stage.rank as f64);
            Ok(ModelAdapters::<llm::TrainCudaAd>::init(
                config, stage.rank, alpha, stage.seed, device,
            )?)
        }
    }
}

#[cfg(feature = "train-cuda")]
fn stage_options(args: &StageArgs, steps: usize) -> LoopOptions {
    LoopOptions {
        steps,
        learning_rate: args.learning_rate,
        warmup_steps: args.warmup_steps,
        loss_chunk: args.loss_chunk,
        log_every: args.log_every,
    }
}

/// `bquest train cpt`: LoRA CPT over a packed-chunks artifact.
pub(crate) fn train_cpt(cli: &Cli, args: &TrainCptArgs) -> BquestResult<()> {
    let config = llm::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let model_dir = config.model_dir();
    let model_id = config.model.clone();
    let alpha = args.alpha.unwrap_or(2.0 * args.rank as f64);
    let pack = load_chunks(&args.chunks)?;
    snafu::ensure_whatever!(
        pack.masks.is_none(),
        "{} carries a loss mask, so it is a supervised pack - use `train sft`",
        args.chunks.display()
    );
    let chunks = pack.rows;
    let steps = args.steps.unwrap_or(chunks.len());
    let log_path = cpt_log_path(&args.out, &args.log);
    eprintln!(
        "bquest train cpt: {} chunks of {}, {} steps, rank {} alpha {alpha}",
        chunks.len(),
        chunks[0].len() - 1,
        steps,
        args.rank
    );

    #[cfg(feature = "train-cuda")]
    {
        let device: llm::CudaDevice = Default::default();
        let hybrid_config = HybridCheckpointConfig::load(&model_dir)?;
        let load_start = std::time::Instant::now();
        let weights = HybridWeights::load(&model_dir)?;
        let model = HybridModel::<llm::CudaBack>::new(&hybrid_config, weights, device.clone())?;
        eprintln!(
            "bquest train cpt: model built in {:.1}s",
            load_start.elapsed().as_secs_f32()
        );
        let mut adapters = ModelAdapters::<llm::TrainCudaAd>::init(
            &hybrid_config,
            args.rank,
            alpha,
            args.seed,
            &device,
        )?;
        let options = LoopOptions {
            steps,
            learning_rate: args.learning_rate,
            warmup_steps: args.warmup_steps,
            loss_chunk: args.loss_chunk,
            log_every: args.log_every,
        };
        let report = llm::train::train_loop::<llm::TrainCudaAd>(
            &model,
            &mut adapters,
            &chunks,
            &options,
            &device,
            |log| Ok(append_step_log(&log_path, log)?),
        )?;
        adapters.save(&args.out, &model_id)?;
        let summary = harness::nu::Value::record(
            harness::nu::record! {
                "steps" => v_int(report.steps as i64),
                "first_loss" => v_float(report.first_loss as f64),
                "final_loss" => v_float(report.final_loss as f64),
                "trained_tokens" => v_int(report.trained_tokens as i64),
                "tokens_per_second" => v_float(report.trained_tokens as f64 / report.seconds),
                "seconds" => v_float(report.seconds),
                "out" => v_str(&args.out.display().to_string()),
                "log" => v_str(&log_path.display().to_string()),
            },
            span(),
        );
        println!("{}", harness::nu::to_nuon_text(&summary)?);
        Ok(())
    }
    #[cfg(not(feature = "train-cuda"))]
    {
        let _ = (model_dir, model_id, alpha, steps, log_path);
        snafu::whatever!("bquest train cpt runs on cuda (rebuild with --features train-cuda)")
    }
}

/// `bquest train sft`: supervised tuning over a packed artifact.
pub(crate) fn train_sft(cli: &Cli, args: &TrainSftArgs) -> BquestResult<()> {
    let config = llm::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let model_dir = config.model_dir();
    let model_id = config.model.clone();
    let pack = load_chunks(&args.chunks)?;
    let Some(masks) = pack.masks else {
        snafu::whatever!(
            "{} carries no loss mask, so it is a continued-pretraining pack - \
             pack instruction examples with `mix instruct`",
            args.chunks.display()
        );
    };
    let rows = pack.rows;
    let steps = args
        .stage
        .steps
        .unwrap_or_else(|| rows.len().div_ceil(args.accumulate.max(1)));
    eprintln!(
        "bquest train sft: {} rows of {}, {steps} steps, accumulate {}, rank {}",
        rows.len(),
        rows[0].len() - 1,
        args.accumulate,
        args.stage.rank
    );

    #[cfg(feature = "train-cuda")]
    {
        let log_path = stage_log_path(&args.stage);
        let device: llm::CudaDevice = Default::default();
        let hybrid_config = HybridCheckpointConfig::load(&model_dir)?;
        let weights = HybridWeights::load(&model_dir)?;
        let model = HybridModel::<llm::CudaBack>::new(&hybrid_config, weights, device.clone())?;
        let mut adapters = stage_adapters(&args.stage, &hybrid_config, &model_id, &device)?;
        let batch = SupervisedBatch {
            rows: &rows,
            masks: &masks,
            accumulate: args.accumulate,
        };
        let report = llm::train::sft_loop::<llm::TrainCudaAd>(
            &model,
            &mut adapters,
            &batch,
            &stage_options(&args.stage, steps),
            &device,
            |log| Ok(append_step_log(&log_path, log)?),
        )?;
        adapters.save(&args.stage.out, &model_id)?;
        print_stage_summary("sft", &report, &args.stage.out, &log_path)
    }
    #[cfg(not(feature = "train-cuda"))]
    {
        let _ = (model_dir, model_id, steps, masks, rows);
        snafu::whatever!("bquest train sft runs on cuda (rebuild with --features train-cuda)")
    }
}

/// `bquest train dpo`: preference tuning over pairs.
pub(crate) fn train_dpo(cli: &Cli, args: &TrainDpoArgs) -> BquestResult<()> {
    let config = llm::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let model_dir = config.model_dir();
    let model_id = config.model.clone();
    let pairs = example::load_dpo(&args.pairs)?;
    let width = args.seq_len + 1;
    eprintln!(
        "bquest train dpo: {} pairs, window {width}, beta {}, rank {}",
        pairs.len(),
        args.beta,
        args.stage.rank
    );

    #[cfg(feature = "train-cuda")]
    {
        let log_path = stage_log_path(&args.stage);
        let device: llm::CudaDevice = Default::default();
        let tokenizer = llm::load_tokenizer(&model_dir)?;
        llm::verify_token_map(&tokenizer)?;
        let hybrid_config = HybridCheckpointConfig::load(&model_dir)?;
        let weights = HybridWeights::load(&model_dir)?;
        let model = HybridModel::<llm::CudaBack>::new(&hybrid_config, weights, device.clone())?;

        let mut encoded: Vec<EncodedPair> = Vec::with_capacity(pairs.len());
        let mut dropped = 0usize;
        for pair in &pairs {
            let build = |reply: &str| -> BquestResult<Option<(Vec<u32>, Vec<u8>)>> {
                let mut messages = pair.prompt.clone();
                messages.push(llm::ChatMessage::assistant(reply));
                let (ids, mask) = example::encode_supervised(&tokenizer, &messages)?;
                Ok(example::pack_row(ids, mask, width, llm::consts::TOKEN_PAD)
                    .map(|row| (row.ids, row.mask)))
            };
            let (Some(chosen), Some(rejected)) = (build(&pair.chosen)?, build(&pair.rejected)?)
            else {
                dropped += 1;
                continue;
            };
            let mask_inner = llm::oracle::causal_mask::<llm::CudaBack>(args.seq_len, &device);
            let reference = |packed: &(Vec<u32>, Vec<u8>)| -> BquestResult<f32> {
                let (inputs, targets, target_mask) =
                    llm::train::split_row(&packed.0, &packed.1);
                Ok(llm::train::reference_logprob::<llm::CudaBack>(
                    &model,
                    &mask_inner,
                    inputs,
                    targets,
                    target_mask,
                    args.stage.loss_chunk,
                    &device,
                )?)
            };
            encoded.push(EncodedPair {
                chosen_reference: reference(&chosen)?,
                rejected_reference: reference(&rejected)?,
                chosen,
                rejected,
            });
        }
        if dropped > 0 {
            eprintln!("bquest train dpo: {dropped} pairs exceed the window and were DROPPED");
        }
        let steps = args.stage.steps.unwrap_or(encoded.len());
        let mut adapters = stage_adapters(&args.stage, &hybrid_config, &model_id, &device)?;
        let report = llm::train::dpo_loop::<llm::TrainCudaAd>(
            &model,
            &mut adapters,
            &encoded,
            args.beta,
            &stage_options(&args.stage, steps),
            &device,
            |log| Ok(append_step_log(&log_path, log)?),
        )?;
        adapters.save(&args.stage.out, &model_id)?;
        print_stage_summary("dpo", &report, &args.stage.out, &log_path)
    }
    #[cfg(not(feature = "train-cuda"))]
    {
        let _ = (model_dir, model_id, width, pairs);
        snafu::whatever!("bquest train dpo runs on cuda (rebuild with --features train-cuda)")
    }
}

/// `bquest train rlvr`: reinforcement tuning over scored groups.
pub(crate) fn train_rlvr(cli: &Cli, args: &TrainRlvrArgs) -> BquestResult<()> {
    let config = llm::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let model_dir = config.model_dir();
    let model_id = config.model.clone();
    let width = args.seq_len + 1;
    let (raw_groups, dropped) =
        rollout::load_groups(&args.rollouts, width, llm::consts::TOKEN_PAD)?;
    if dropped > 0 {
        eprintln!("bquest train rlvr: {dropped} rollouts exceed the window and were DROPPED");
    }
    let group_count = raw_groups.len();
    let steps = args.stage.steps.unwrap_or(group_count);
    eprintln!(
        "bquest train rlvr: {group_count} groups, window {width}, {steps} steps, rank {}",
        args.stage.rank
    );

    #[cfg(feature = "train-cuda")]
    {
        let log_path = stage_log_path(&args.stage);
        let groups: Vec<RolloutGroup> = raw_groups
            .into_iter()
            .map(|rollouts| RolloutGroup {
                rollouts: rollouts
                    .into_iter()
                    .map(|(ids, mask, reward)| ScoredRollout { ids, mask, reward })
                    .collect(),
            })
            .collect();
        let device: llm::CudaDevice = Default::default();
        let hybrid_config = HybridCheckpointConfig::load(&model_dir)?;
        let weights = HybridWeights::load(&model_dir)?;
        let model = HybridModel::<llm::CudaBack>::new(&hybrid_config, weights, device.clone())?;
        let mut adapters = stage_adapters(&args.stage, &hybrid_config, &model_id, &device)?;
        let report = llm::train::rlvr_loop::<llm::TrainCudaAd>(
            &model,
            &mut adapters,
            &groups,
            &stage_options(&args.stage, steps),
            &device,
            |log| Ok(append_step_log(&log_path, log)?),
        )?;
        adapters.save(&args.stage.out, &model_id)?;
        print_stage_summary("rlvr", &report, &args.stage.out, &log_path)
    }
    #[cfg(not(feature = "train-cuda"))]
    {
        let _ = (model_dir, model_id, steps, raw_groups);
        snafu::whatever!("bquest train rlvr runs on cuda (rebuild with --features train-cuda)")
    }
}
