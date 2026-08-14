//! Organism full-parameter trainer locks on a tied toy hybrid (cpu
//! f32): walk-vs-full-graph gradient equivalence (segmented too), the
//! touched-row embedding contract, AdamW math, descent, and the
//! checkpoint round trip.
use crate::*;

use crate::imagine_quest_train as organism;

type ToyAd = burn::backend::Autodiff<CpuBack>;

/// The toy hybrid with the organism's tied-embedding posture.
fn tied_toy_config() -> HybridCheckpointConfig {
    let mut config = train::toy_config();
    config.tie_word_embeddings = true;
    config
}

/// A config.json body the checkpoint loaders accept for the tied toy.
fn tied_toy_config_json() -> Vec<u8> {
    serde_json::to_vec_pretty(&serde_json::json!({
        "vocab_size": 96,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "max_position_embeddings": 65536,
        "layer_types": ["linear_attention", "full_attention"],
        "tie_word_embeddings": true,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 12,
        "linear_value_head_dim": 24,
        "linear_conv_kernel_dim": 4,
        "linear_allow_neg_eigval": true,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }))
    .expect("toy config renders")
}

fn tied_toy_trainer() -> organism::OrganismTrainer<ToyAd> {
    let device: <CpuBack as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let config = tied_toy_config();
    let model = HybridModel::<CpuBack>::new(&config, train::toy_weights(&config), device)
        .expect("tied toy model");
    organism::OrganismTrainer::<ToyAd>::from_parts(
        config,
        tied_toy_config_json(),
        model,
        &Default::default(),
    )
    .expect("tied toy trainer")
}

fn toy_chunk(seed: u64, seq_len: usize, vocab: usize) -> Vec<u32> {
    let mut rng = SplitMix64::new(seed);
    (0..seq_len + 1)
        .map(|_| (rng.next_u64() % vocab as u64) as u32)
        .collect()
}

/// Normalized mean squared error between two value sets, f64.
fn nmse(reference: &[f32], candidate: &[f32]) -> f64 {
    assert_eq!(reference.len(), candidate.len(), "value sets differ in size");
    let denominator: f64 = reference.iter().map(|v| (*v as f64).powi(2)).sum();
    let numerator: f64 = reference
        .iter()
        .zip(candidate)
        .map(|(r, c)| (*r as f64 - *c as f64).powi(2))
        .sum();
    if denominator == 0.0 {
        assert_eq!(numerator, 0.0, "reference is zero but the candidate is not");
        return 0.0;
    }
    numerator / denominator
}

/// The per-layer walk reproduces the full-graph gradients: every core
/// weight within the nmse bar at both segment widths, the embedding's
/// touched rows exact, and the dropped remainder demonstrably real
/// (untouched rows DO carry head gradient the sparse path skips).
#[test]
fn organism_walk_matches_the_full_graph() {
    let device: <CpuBack as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let trainer = tied_toy_trainer();
    let seq_len = 16usize;
    let chunk = toy_chunk(42, seq_len, 96);
    let inputs = &chunk[..seq_len];
    let targets = &chunk[1..];
    let hidden = trainer.model.hidden;

    let reference = organism::reference_step::<ToyAd>(&trainer.model, inputs, targets, &device)
        .expect("reference step");

    for segment in [seq_len, 4] {
        let (loss, grads) = trainer
            .grads_for_chunk(&chunk, 8, segment)
            .expect("walk step");
        assert!(
            (loss - reference.loss).abs() <= 1e-5,
            "segment {segment}: walk loss {loss} vs reference {}",
            reference.loss
        );

        let mut walk_names: Vec<&String> = grads.core_names();
        walk_names.sort();
        let mut reference_names: Vec<&String> = reference.core.keys().collect();
        reference_names.sort();
        assert_eq!(walk_names, reference_names, "segment {segment}: core name sets");
        for name in reference_names {
            let reference_values =
                organism::param_host_values(&reference.core[name]).expect("reference values");
            let walk_values = organism::param_host_values(
                grads.core_grad(name).expect("walk grad present"),
            )
            .expect("walk values");
            let error = nmse(&reference_values, &walk_values);
            assert!(
                error <= 1e-9,
                "segment {segment}: {name} nmse {error:.3e} over the 1e-9 bar"
            );
        }

        let mut touched: Vec<u32> = chunk.clone();
        touched.sort_unstable();
        touched.dedup();
        assert_eq!(
            grads.row_ids(),
            touched,
            "segment {segment}: touched rows must be exactly inputs plus targets"
        );
        for row in &touched {
            let reference_row =
                &reference.embedding[*row as usize * hidden..(*row as usize + 1) * hidden];
            let walk_row = grads.row_grad(*row).expect("touched row grad");
            let error = nmse(reference_row, walk_row);
            assert!(
                error <= 1e-9,
                "segment {segment}: embedding row {row} nmse {error:.3e} over the bar"
            );
        }
    }

    // The deliberate drop is a real approximation: some untouched row
    // carries head gradient in the dense reference.
    let mut touched: Vec<u32> = chunk.clone();
    touched.sort_unstable();
    touched.dedup();
    let untouched_norm: f64 = (0..96u32)
        .filter(|row| touched.binary_search(row).is_err())
        .map(|row| {
            reference.embedding[row as usize * hidden..(row as usize + 1) * hidden]
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
        })
        .sum();
    assert!(
        untouched_norm > 0.0,
        "untouched rows carry no reference gradient; the sparse drop would be vacuous"
    );
}

/// The manual AdamW matches an f64 mirror over two steps.
#[test]
fn organism_adamw_matches_a_hand_case() {
    let device: <CpuBack as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let tensor1 = |value: f32| {
        burn::tensor::Tensor::<CpuBack, 1>::from_data(
            burn::tensor::TensorData::new(vec![value], [1]),
            &device,
        )
    };
    let mut master = tensor1(1.0);
    let mut moment = tensor1(0.0);
    let mut velocity = tensor1(0.0);

    let (beta1, beta2, eps, rate) = (0.9f64, 0.95f64, 1e-8f64, 0.1f64);
    let mut mirror_master = 1.0f64;
    let mut mirror_moment = 0.0f64;
    let mut mirror_velocity = 0.0f64;
    for (step, grad) in [(1usize, 0.5f64), (2, -0.25)] {
        let bias1 = 1.0 - beta1.powi(step as i32);
        let bias2 = 1.0 - beta2.powi(step as i32);
        organism::adamw_tensor(
            &mut master,
            &mut moment,
            &mut velocity,
            tensor1(grad as f32),
            rate,
            bias1,
            bias2,
        );
        mirror_moment = beta1 * mirror_moment + (1.0 - beta1) * grad;
        mirror_velocity = beta2 * mirror_velocity + (1.0 - beta2) * grad * grad;
        let moment_hat = mirror_moment / bias1;
        let velocity_hat = mirror_velocity / bias2;
        mirror_master -= rate * moment_hat / (velocity_hat.sqrt() + eps);
    }
    let lived = master
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("master extraction")[0];
    assert!(
        (lived as f64 - mirror_master).abs() <= 1e-5,
        "adamw master {lived} vs the f64 mirror {mirror_master}"
    );
}

/// The stage loop descends on the tied toy, untouched embedding rows
/// stay frozen, and the saved checkpoint reloads to the same model.
#[test]
fn organism_descends_saves_and_reloads() {
    let mut trainer = tied_toy_trainer();
    let seq_len = 12usize;
    // Ids capped below 48 so rows 48..96 stay untouched by training.
    let chunks: Vec<Vec<u32>> = (0..4)
        .map(|index| toy_chunk(100 + index as u64, seq_len, 48))
        .collect();
    let hidden = trainer.model.hidden;
    let frozen_before: Vec<f32> = trainer.model.embed_rows[90 * hidden..91 * hidden].to_vec();

    let options = organism::OrganismLoopOptions {
        steps: 60,
        learning_rate: 5e-2,
        warmup_steps: 2,
        loss_chunk: 8,
        log_every: usize::MAX,
        accumulate: 1,
        recurrence_segment: 4,
        checkpoint_every: 0,
    };
    let report = trainer
        .train(&chunks, &options, None, |_| Ok(()))
        .expect("toy training run");
    assert!(
        report.final_loss < report.first_loss - 0.05,
        "loss did not descend meaningfully: {} -> {}",
        report.first_loss,
        report.final_loss
    );

    let frozen_after = &trainer.model.embed_rows[90 * hidden..91 * hidden];
    assert_eq!(
        frozen_before, frozen_after,
        "an untouched embedding row moved; row-sparsity is broken"
    );

    let out_dir = std::env::temp_dir().join(format!(
        "biquest_toy_organism_{}",
        std::process::id()
    ));
    trainer.save_checkpoint(&out_dir).expect("checkpoint save");
    let reloaded = organism::OrganismTrainer::<ToyAd>::load(&out_dir, &Default::default())
        .expect("checkpoint reload");

    let probe = toy_chunk(7, seq_len, 48);
    let (_, live_logits) = trainer
        .model
        .forward_all(&probe[..seq_len])
        .expect("live forward");
    let (_, reloaded_logits) = reloaded
        .model
        .forward_all(&probe[..seq_len])
        .expect("reloaded forward");
    let scale = live_logits
        .iter()
        .fold(0f32, |acc, value| acc.max(value.abs()))
        .max(1.0);
    let worst = live_logits
        .iter()
        .zip(&reloaded_logits)
        .fold(0f32, |acc, (live, back)| acc.max((live - back).abs()));
    assert!(
        worst <= 0.05 * scale,
        "reloaded logits diverge: worst {worst} against scale {scale}"
    );

    // The saved untouched row is exactly the frozen master, rounded.
    let reloaded_frozen = &reloaded.model.embed_rows[90 * hidden..91 * hidden];
    for (before, after) in frozen_before.iter().zip(reloaded_frozen) {
        let expected = half::bf16::from_f32(*before).to_f32();
        assert_eq!(
            expected, *after,
            "a frozen row's saved value drifted from its master"
        );
    }

    fs::remove_dir_all(&out_dir).ok();
}
