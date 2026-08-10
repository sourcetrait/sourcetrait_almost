//! CptLoop trainer locks on the toy hybrid config (cpu f32): the
//! gate trio (adapter-off exactness, chain-vs-full gradient
//! equivalence, descent) plus the adapter artifact's placement
//! surface and metadata.
use crate::*;

/// The three toy locks, exactly what `bquest train gate` runs.
#[test]
fn train_gate_passes() {
    let verdict = train::run_train_gate().expect("the toy gate must run");
    assert!(
        verdict.adapter_off_exact,
        "zero-init adapters changed the forward"
    );
    assert!(
        verdict.init_grads.max_nmse <= 1e-9,
        "init chain-vs-full grad nmse {:.3e} over the 1e-9 bar",
        verdict.init_grads.max_nmse
    );
    assert!(
        verdict.init_grads.compared > 0,
        "no nonzero gradients compared at init"
    );
    assert!(
        verdict.init_grads.zero_matched > 0,
        "zero-init b implies zero a-gradients; none were seen"
    );
    assert!(
        verdict.trained_grads.max_nmse <= 1e-9,
        "trained chain-vs-full grad nmse {:.3e} over the 1e-9 bar",
        verdict.trained_grads.max_nmse
    );
    assert!(
        verdict.descended,
        "loss did not descend meaningfully: {} -> {}",
        verdict.first_loss,
        verdict.final_loss
    );
}

/// bf16 cuda chain-step gradient health on the TOY config at
/// increasing depths - the NaN-onset bisection instrument (the real
/// smoke's first GDN backward NaNs its input gradient; this splits
/// depth x dtype from real-weight magnitudes). Report-only.
#[cfg(feature = "train-cuda")]
#[test]
#[ignore]
fn diag_toy_chain_grad_health_cuda_bf16() {
    type DiagAd = burn::backend::Autodiff<CudaBack>;
    let device: <CudaBack as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let config = train::toy_config();
    let model = HybridModel::<CudaBack>::new(&config, train::toy_weights(&config), device.clone())
        .expect("toy model on cuda");
    let adapters = ModelAdapters::<DiagAd>::init(&config, 4, 8.0, 42, &device)
        .expect("toy adapters");
    for seq_len in [16usize, 256, 1024] {
        let mut rng = SplitMix64::new(1000 + seq_len as u64);
        let chunk: Vec<u32> = (0..seq_len + 1)
            .map(|_| (rng.next_u64() % config.vocab_size as u64) as u32)
            .collect();
        let mask = oracle::causal_mask::<CudaBack>(seq_len, &device);
        let outcome = train::chain_step::<DiagAd>(
            &model,
            &adapters,
            &mask,
            &chunk[..seq_len],
            &chunk[1..],
            8,
            Some(train::RECURRENCE_SEGMENT),
            &device,
        )
        .expect("chain step");
        let mut total = 0usize;
        let mut non_finite = 0usize;
        for grad in outcome.grads.values() {
            let values = grad
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .expect("grad extraction");
            total += values.len();
            non_finite += values.iter().filter(|v| !v.is_finite()).count();
        }
        println!(
            "toy bf16 chain @ seq {seq_len}: loss {} | {non_finite}/{total} non-finite grads",
            outcome.loss
        );
    }
}

/// REAL-checkpoint chain-step gradient health at short depths (cuda
/// bf16): pairs with the toy diag to split real-weight magnitudes
/// from depth. Prints per-layer non-finite counts per seq. Needs the
/// DPO checkpoint (QUEST_MODEL_DIR overrides) + a chunks artifact at
/// QUEST_DIAG_CHUNKS. Report-only.
#[cfg(feature = "train-cuda")]
#[test]
#[ignore]
fn diag_real_chain_grad_health_cuda_bf16() {
    type DiagAd = burn::backend::Autodiff<CudaBack>;
    let device: <CudaBack as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let model_dir = match std::env::var("QUEST_MODEL_DIR") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => crate::model_dir(crate::consts::DPO_MODEL_NAME).expect("the DPO checkpoint home"),
    };
    let config = HybridCheckpointConfig::load(&model_dir).expect("config");
    let weights = HybridWeights::load(&model_dir).expect("weights");
    let model =
        HybridModel::<CudaBack>::new(&config, weights, device.clone()).expect("model");
    let adapters = ModelAdapters::<DiagAd>::init(&config, 64, 128.0, 299_792_458, &device)
        .expect("adapters");
    let chunks_path = PathBuf::from(
        std::env::var("QUEST_DIAG_CHUNKS").expect("set QUEST_DIAG_CHUNKS to a pack artifact"),
    );
    let bytes = fs::read(&chunks_path).expect("chunks read");
    let parsed = safetensors::SafeTensors::deserialize(&bytes).expect("chunks parse");
    let view = parsed.tensor("chunks").expect("chunks tensor");
    let width = view.shape()[1];
    let first_chunk: Vec<u32> = view.data()[..width * 4]
        .chunks_exact(4)
        .map(|quad| u32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
        .collect();

    for seq_len in [16usize, 64, 256] {
        let mask = oracle::causal_mask::<CudaBack>(seq_len, &device);
        let outcome = train::chain_step::<DiagAd>(
            &model,
            &adapters,
            &mask,
            &first_chunk[..seq_len],
            &first_chunk[1..seq_len + 1],
            8,
            Some(train::RECURRENCE_SEGMENT),
            &device,
        )
        .expect("chain step");
        let mut by_layer: HashMap<usize, (usize, usize)> = HashMap::new();
        for (name, grad) in &outcome.grads {
            let layer: usize = name
                .strip_prefix("model.layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|index| index.parse().ok())
                .unwrap_or(usize::MAX);
            let values = grad
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .expect("grad extraction");
            let entry = by_layer.entry(layer).or_insert((0, 0));
            entry.0 += values.len();
            entry.1 += values.iter().filter(|v| !v.is_finite()).count();
        }
        let poisoned: Vec<usize> = {
            let mut layers: Vec<usize> = by_layer
                .iter()
                .filter(|(_, (_, bad))| *bad > 0)
                .map(|(layer, _)| *layer)
                .collect();
            layers.sort_unstable();
            layers
        };
        println!(
            "real bf16 chain @ seq {seq_len}: loss {} | poisoned layers {poisoned:?}",
            outcome.loss
        );
    }
}

/// SftLoop's token dump is the final pass decomposed: one entry per
/// visited row, positions exactly the supervised full-row mask
/// positions, and (single-row control) the dumped mean equal to the
/// last step-log loss.
#[test]
fn sft_token_losses_decompose_the_final_pass() {
    type ToyAd = burn::backend::Autodiff<CpuBack>;
    let device: <CpuBack as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let config = train::toy_config();
    let model = HybridModel::<CpuBack>::new(&config, train::toy_weights(&config), device)
        .expect("toy model");

    let seq_len = 12usize;
    let mut rng = SplitMix64::new(7);
    let rows: Vec<Vec<u32>> = (0..3)
        .map(|_| {
            (0..seq_len + 1)
                .map(|_| (rng.next_u64() % config.vocab_size as u64) as u32)
                .collect()
        })
        .collect();
    let mut masks: Vec<Vec<u8>> = vec![vec![0u8; seq_len + 1]; 3];
    for slot in masks[0].iter_mut().skip(4).take(5) {
        *slot = 1;
    }
    for position in [2usize, 3, 9] {
        masks[1][position] = 1;
    }
    for slot in masks[2].iter_mut().skip(1) {
        *slot = 1;
    }

    let options = train::LoopOptions {
        steps: 7,
        learning_rate: 1e-2,
        warmup_steps: 2,
        loss_chunk: 8,
        log_every: usize::MAX,
    };
    let mut adapters =
        ModelAdapters::<ToyAd>::init(&config, 4, 8.0, 42, &device).expect("toy adapters");
    let batch = train::SupervisedBatch { rows: &rows, masks: &masks, accumulate: 1 };
    let sft = train::sft_loop::<ToyAd>(
        &model,
        &mut adapters,
        &batch,
        train::SftObjective::ExampleMean,
        &options,
        &device,
        |_| Ok(()),
    )
    .expect("sft loop");

    assert_eq!(sft.token_losses.len(), rows.len(), "one dump entry per visited row");
    for dump in &sft.token_losses {
        let mask = &masks[dump.row];
        let expected: Vec<usize> = (1..mask.len()).filter(|i| mask[*i] != 0).collect();
        assert_eq!(dump.positions, expected, "row {} dump positions", dump.row);
        assert_eq!(dump.losses.len(), dump.positions.len());
        assert!(
            dump.losses.iter().all(|loss| loss.is_finite() && *loss >= 0.0),
            "row {} carries a non-finite or negative loss",
            dump.row
        );
    }

    let one_rows = vec![rows[0].clone()];
    let one_masks = vec![masks[0].clone()];
    let mut adapters =
        ModelAdapters::<ToyAd>::init(&config, 4, 8.0, 43, &device).expect("toy adapters");
    let batch = train::SupervisedBatch { rows: &one_rows, masks: &one_masks, accumulate: 1 };
    let mut last_loss = 0f32;
    let sft = train::sft_loop::<ToyAd>(
        &model,
        &mut adapters,
        &batch,
        train::SftObjective::ExampleMean,
        &train::LoopOptions { steps: 3, ..options },
        &device,
        |log| {
            last_loss = log.loss;
            Ok(())
        },
    )
    .expect("single-row sft loop");
    let dump = &sft.token_losses[0];
    let mean: f64 =
        dump.losses.iter().map(|loss| *loss as f64).sum::<f64>() / dump.losses.len() as f64;
    assert!(
        (mean - last_loss as f64).abs() <= 1e-4 * mean.abs().max(1.0),
        "dumped mean {mean} does not decompose the final step loss {last_loss}"
    );
}

/// The token-uniform objective rescales each row's loss by exactly
/// supervised/mean against the standing per-example mean: two frozen
/// (zero-rate) runs over rows of distinct supervised counts step the
/// same visit order, so each step's loss ratio must be one of the two
/// count ratios, and both ratios must appear.
#[test]
fn sft_token_uniform_rescales_by_supervised_over_mean() {
    type ToyAd = burn::backend::Autodiff<CpuBack>;
    let device: <CpuBack as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let config = train::toy_config();
    let model = HybridModel::<CpuBack>::new(&config, train::toy_weights(&config), device)
        .expect("toy model");

    let seq_len = 12usize;
    let mut rng = SplitMix64::new(11);
    let rows: Vec<Vec<u32>> = (0..2)
        .map(|_| {
            (0..seq_len + 1)
                .map(|_| (rng.next_u64() % config.vocab_size as u64) as u32)
                .collect()
        })
        .collect();
    let mut masks: Vec<Vec<u8>> = vec![vec![0u8; seq_len + 1]; 2];
    for slot in masks[0].iter_mut().skip(1).take(3) {
        *slot = 1;
    }
    for slot in masks[1].iter_mut().skip(1).take(9) {
        *slot = 1;
    }
    let mean = (3.0 + 9.0) / 2.0;
    let candidate_ratios = [3.0 / mean, 9.0 / mean];

    let options = train::LoopOptions {
        steps: 2,
        learning_rate: 0.0,
        warmup_steps: 1,
        loss_chunk: 8,
        log_every: usize::MAX,
    };
    let run = |objective: train::SftObjective| -> Vec<f32> {
        let mut adapters =
            ModelAdapters::<ToyAd>::init(&config, 4, 8.0, 42, &device).expect("toy adapters");
        let batch = train::SupervisedBatch { rows: &rows, masks: &masks, accumulate: 1 };
        let mut losses: Vec<f32> = Vec::new();
        train::sft_loop::<ToyAd>(&model, &mut adapters, &batch, objective, &options, &device, |log| {
            losses.push(log.loss);
            Ok(())
        })
        .expect("frozen sft loop");
        losses
    };
    let example = run(train::SftObjective::ExampleMean);
    let token = run(train::SftObjective::TokenUniform);
    assert_eq!(example.len(), 2);
    assert_eq!(token.len(), 2);

    let mut seen: Vec<f64> = Vec::new();
    for (step, (token_loss, example_loss)) in token.iter().zip(example.iter()).enumerate() {
        let ratio = *token_loss as f64 / *example_loss as f64;
        let matched = candidate_ratios
            .iter()
            .find(|candidate| (ratio - **candidate).abs() <= 1e-4)
            .copied();
        let Some(matched) = matched else {
            panic!("step {step} ratio {ratio} matches neither supervised/mean candidate");
        };
        seen.push(matched);
    }
    assert_ne!(seen[0], seen[1], "both rows must appear across the two steps");
}

/// The adapter artifact carries exactly the direction-3 surface: GDN
/// q/q_conv/g/o + MLP, attn q/o + MLP - and NEVER a state-carrying
/// tensor; the conv delta lands in the checkpoint's (channels, 1,
/// kernel) layout; rank/alpha/model identity ride the metadata.
#[test]
fn adapter_artifact_carries_the_direction_3_surface() {
    type Back = CpuBack;
    let device = Default::default();
    let config = train::toy_config();
    let adapters = ModelAdapters::<Back>::init(&config, 4, 8.0, 42, &device)
        .expect("toy adapters");
    let path = std::env::temp_dir().join(format!(
        "bquest_toy_adapter_{}.safetensors",
        std::process::id()
    ));
    adapters.save(&path, "toy/model").expect("adapter save");

    let bytes = fs::read(&path).expect("adapter read");
    let parsed = safetensors::SafeTensors::deserialize(&bytes).expect("adapter parse");
    let names: HashSet<String> = parsed.names().iter().map(|n| n.to_string()).collect();

    // Layer 0 is GDN: the sanctioned targets present...
    for target in [
        "linear_attn.q_proj",
        "linear_attn.g_proj",
        "linear_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ] {
        assert!(
            names.contains(&format!("model.layers.0.{target}.lora_a")),
            "missing layer-0 {target}.lora_a"
        );
        assert!(
            names.contains(&format!("model.layers.0.{target}.lora_b")),
            "missing layer-0 {target}.lora_b"
        );
    }
    let conv = parsed
        .tensor("model.layers.0.linear_attn.q_conv1d.delta")
        .expect("the q_conv1d delta");
    assert_eq!(conv.shape(), [24, 1, 4], "conv delta must ride the checkpoint layout");
    // ... and the state-carrying weights have NO adapter anywhere.
    for name in &names {
        for forbidden in [
            "k_proj", "v_proj", "a_proj", "b_proj", "k_conv1d", "v_conv1d",
        ] {
            assert!(
                !name.contains(forbidden),
                "state-carrying weight adapted: {name}"
            );
        }
    }
    // Layer 1 is attention: q/o + MLP.
    for target in ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj"] {
        assert!(
            names.contains(&format!("model.layers.1.{target}.lora_a")),
            "missing layer-1 {target}.lora_a"
        );
    }

    let (_, header) =
        safetensors::SafeTensors::read_metadata(&bytes).expect("adapter header");
    let metadata = header.metadata().as_ref().expect("adapter metadata");
    assert_eq!(metadata.get("version").map(String::as_str), Some("1"));
    assert_eq!(metadata.get("model_id").map(String::as_str), Some("toy/model"));
    assert_eq!(metadata.get("lora_rank").map(String::as_str), Some("4"));
    assert_eq!(metadata.get("lora_alpha").map(String::as_str), Some("8"));
    assert_eq!(metadata.get("seed").map(String::as_str), Some("42"));
    fs::remove_file(&path).ok();
}
