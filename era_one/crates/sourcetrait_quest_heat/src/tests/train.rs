use crate::*;

use crate::train::{TrainOptions, train_on};
use crate::train_data::{Packed, SplitMix64};

/// A 2-layer toy Olmo 3 shape (one sliding + one full layer) small enough
/// for cpu-f32 autodiff.
fn toy_config() -> HeatConfig {
    HeatConfig {
        vocab_size: 96,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: None,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rope_scaling: None,
        sliding_window: 4,
        layer_types: Some(vec![
            String::from("sliding_attention"),
            String::from("full_attention"),
        ]),
        tie_word_embeddings: false,
    }
}

/// Matrix init at ~1/sqrt(fan_in) scale so activations neither explode
/// nor vanish through the toy depth.
fn toy_tensor(rng: &mut SplitMix64, shape: Vec<usize>) -> (Vec<usize>, Vec<f32>) {
    let count: usize = shape.iter().product();
    let values = (0..count)
        .map(|_| ((rng.next_u64() % 2000) as f32 / 1000.0 - 1.0) * 0.3)
        .collect();
    (shape, values)
}

/// Norm scales init near 1.0 (a small norm weight multiplies every
/// activation down and flattens the loss landscape - the fixture trap).
fn toy_norm(rng: &mut SplitMix64, width: usize) -> (Vec<usize>, Vec<f32>) {
    let values = (0..width)
        .map(|_| 1.0 + ((rng.next_u64() % 2000) as f32 / 1000.0 - 1.0) * 0.05)
        .collect();
    (vec![width], values)
}

fn toy_weights(config: &HeatConfig) -> Weights {
    let mut rng = SplitMix64(7);
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let vocab = config.vocab_size;
    let mut tensors = HashMap::new();
    tensors.insert(
        String::from("model.embed_tokens.weight"),
        toy_tensor(&mut rng, vec![vocab, hidden]),
    );
    for index in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{index}");
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            tensors.insert(
                format!("{prefix}.self_attn.{name}.weight"),
                toy_tensor(&mut rng, vec![hidden, hidden]),
            );
        }
        for name in ["q_norm", "k_norm"] {
            tensors.insert(
                format!("{prefix}.self_attn.{name}.weight"),
                toy_norm(&mut rng, hidden),
            );
        }
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            toy_tensor(&mut rng, vec![intermediate, hidden]),
        );
        tensors.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            toy_tensor(&mut rng, vec![intermediate, hidden]),
        );
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            toy_tensor(&mut rng, vec![hidden, intermediate]),
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            toy_norm(&mut rng, hidden),
        );
        tensors.insert(
            format!("{prefix}.post_feedforward_layernorm.weight"),
            toy_norm(&mut rng, hidden),
        );
    }
    tensors.insert(
        String::from("model.norm.weight"),
        toy_norm(&mut rng, hidden),
    );
    tensors.insert(
        String::from("lm_head.weight"),
        toy_tensor(&mut rng, vec![vocab, hidden]),
    );
    Weights::from_tensors(tensors)
}

/// Gradient flow end to end: on a fixed toy corpus the loss must descend
/// (the chunks repeat, so a working loop memorizes them) and the adapter
/// artifact must land on disk.
#[test]
fn toy_lora_training_descends() {
    let config = toy_config();
    let weights = toy_weights(&config);
    let seq_len = 16usize;
    let mut rng = SplitMix64(42);
    let chunks: Vec<Vec<u32>> = (0..4)
        .map(|_| {
            (0..seq_len + 1)
                .map(|_| (rng.next_u64() % config.vocab_size as u64) as u32)
                .collect()
        })
        .collect();
    let packed = Packed {
        total_tokens: chunks.len() * (seq_len + 1),
        documents: chunks.len(),
        chunks,
    };
    let out = std::env::temp_dir().join(format!(
        "heat_toy_adapter_{}.safetensors",
        std::process::id()
    ));
    let options = TrainOptions {
        data_roots: Vec::new(),
        exclude_dirs: Vec::new(),
        tokenizer: PathBuf::new(),
        model_dir: PathBuf::new(),
        out: out.clone(),
        seq_len,
        steps: 60,
        rank: 4,
        alpha: 16.0,
        learning_rate: 5e-2,
        warmup_steps: 2,
        seed: 42,
        loss_chunk: 8,
        log_every: 100,
    };

    type ToyBack = burn::backend::Autodiff<CpuBack>;
    let report = train_on::<ToyBack>(&config, weights, Default::default(), &packed, &options)
        .expect("toy training failed");
    // A real margin: 60 steps over 4 repeating chunks memorizes them;
    // a broken gradient path cannot cross it by noise.
    assert!(
        report.final_loss < report.first_loss - 0.05,
        "loss did not descend meaningfully: {} -> {}",
        report.first_loss,
        report.final_loss
    );
    assert!(out.exists(), "adapter file missing");
    std::fs::remove_file(&out).ok();
}
