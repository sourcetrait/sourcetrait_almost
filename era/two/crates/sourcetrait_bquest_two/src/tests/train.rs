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
        let mask = causal_mask::<CudaBack>(seq_len, &device);
        let outcome = train::chain_step::<DiagAd>(
            &model,
            &adapters,
            &mask,
            &chunk[..seq_len],
            &chunk[1..],
            8,
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
        _ => lib::model_dir(lib::consts::DPO_MODEL_NAME).expect("the DPO checkpoint home"),
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
        let mask = causal_mask::<CudaBack>(seq_len, &device);
        let outcome = train::chain_step::<DiagAd>(
            &model,
            &adapters,
            &mask,
            &first_chunk[..seq_len],
            &first_chunk[1..seq_len + 1],
            8,
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
