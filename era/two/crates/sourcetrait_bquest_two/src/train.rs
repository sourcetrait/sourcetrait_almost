//! The CptLoop trainer stage: LoRA CPT (plain LM loss over packed
//! chunks) under MANUAL PER-LAYER CHECKPOINTING - a no-grad pass over
//! the frozen model caches each layer's input, then the loss block
//! and each layer run as their own small autodiff graphs chained by
//! vector-Jacobian products, so at most one layer's graph is ever
//! live (the era-one method: full-graph autodiff over a 7B measured
//! ~7.5 GiB of retained forward on Tier A - card overflow beside the
//! weights). Burn autodiff over the sequential GDN recurrence is the
//! correctness grade; the chunked backward stays a later perf lever.
use crate::*;

use burn::optim::SimpleOptimizer;
use burn::tensor::{
    ElementConversion,
    Int,
    backend::AutodiffBackend,
    backend::Backend,
};

type FloatTensor<B, const D: usize> = burn::tensor::Tensor<B, D>;
type OptState<B> = <burn::optim::AdamW as SimpleOptimizer<B>>::State<2>;

/// The loop knobs (paths stay with the verb; the gate reuses the
/// loop on toy values).
pub(crate) struct LoopOptions {
    pub(crate) steps: usize,
    pub(crate) learning_rate: f64,
    pub(crate) warmup_steps: usize,
    pub(crate) loss_chunk: usize,
    pub(crate) log_every: usize,
}

/// One step's log row (the NUON-lines record).
#[cfg_attr(not(feature = "train-cuda"), allow(dead_code))]
pub(crate) struct StepLog {
    pub(crate) step: usize,
    pub(crate) loss: f32,
    pub(crate) learning_rate: f64,
    pub(crate) tokens_per_second: f64,
    pub(crate) elapsed_seconds: f64,
}

/// First/final loss plus run totals - the descent signal the gate
/// and the smoke assert on.
#[cfg_attr(not(feature = "train-cuda"), allow(dead_code))]
pub(crate) struct TrainReport {
    pub(crate) first_loss: f32,
    pub(crate) final_loss: f32,
    pub(crate) steps: usize,
    pub(crate) trained_tokens: usize,
    pub(crate) seconds: f64,
}

/// One step's chain outcome: the loss and every adapter gradient on
/// the inner backend, keyed by the artifact names.
pub(crate) struct StepOutcome<B: Backend> {
    pub(crate) loss: f32,
    pub(crate) grads: HashMap<String, FloatTensor<B, 2>>,
}

/// Zero-copy autodiff view of a frozen block (untracked constants).
fn lift_block<AD: AutodiffBackend>(
    block: &HybridBlock<AD::InnerBackend>,
) -> HybridBlock<AD> {
    match block {
        HybridBlock::Gdn(block) => HybridBlock::Gdn(GdnBlock {
            input_norm: FloatTensor::from_inner(block.input_norm.clone()),
            post_attention_norm: FloatTensor::from_inner(block.post_attention_norm.clone()),
            q_transposed: FloatTensor::from_inner(block.q_transposed.clone()),
            k_transposed: FloatTensor::from_inner(block.k_transposed.clone()),
            v_transposed: FloatTensor::from_inner(block.v_transposed.clone()),
            g_transposed: FloatTensor::from_inner(block.g_transposed.clone()),
            o_transposed: FloatTensor::from_inner(block.o_transposed.clone()),
            a_transposed: FloatTensor::from_inner(block.a_transposed.clone()),
            b_transposed: FloatTensor::from_inner(block.b_transposed.clone()),
            q_conv_taps: block.q_conv_taps.iter().map(|t| FloatTensor::from_inner(t.clone())).collect(),
            k_conv_taps: block.k_conv_taps.iter().map(|t| FloatTensor::from_inner(t.clone())).collect(),
            v_conv_taps: block.v_conv_taps.iter().map(|t| FloatTensor::from_inner(t.clone())).collect(),
            a_log_row: FloatTensor::from_inner(block.a_log_row.clone()),
            dt_bias_row: FloatTensor::from_inner(block.dt_bias_row.clone()),
            o_norm: FloatTensor::from_inner(block.o_norm.clone()),
            gate_transposed: FloatTensor::from_inner(block.gate_transposed.clone()),
            up_transposed: FloatTensor::from_inner(block.up_transposed.clone()),
            down_transposed: FloatTensor::from_inner(block.down_transposed.clone()),
        }),
        HybridBlock::Attn(block) => HybridBlock::Attn(AttnBlock {
            q_transposed: FloatTensor::from_inner(block.q_transposed.clone()),
            k_transposed: FloatTensor::from_inner(block.k_transposed.clone()),
            v_transposed: FloatTensor::from_inner(block.v_transposed.clone()),
            o_transposed: FloatTensor::from_inner(block.o_transposed.clone()),
            q_norm: FloatTensor::from_inner(block.q_norm.clone()),
            k_norm: FloatTensor::from_inner(block.k_norm.clone()),
            post_attention_norm: FloatTensor::from_inner(block.post_attention_norm.clone()),
            post_feedforward_norm: FloatTensor::from_inner(block.post_feedforward_norm.clone()),
            gate_transposed: FloatTensor::from_inner(block.gate_transposed.clone()),
            up_transposed: FloatTensor::from_inner(block.up_transposed.clone()),
            down_transposed: FloatTensor::from_inner(block.down_transposed.clone()),
        }),
    }
}

/// Frozen (inner-backend) view of one layer's current adapter values,
/// for the no-grad pass.
fn adapters_inner<AD: AutodiffBackend>(
    layer: &LayerAdapters<AD>,
) -> LayerAdapters<AD::InnerBackend> {
    fn pair_inner<AD: AutodiffBackend>(pair: &LoraPair<AD>) -> LoraPair<AD::InnerBackend> {
        LoraPair {
            a: pair.a.clone().inner(),
            b: pair.b.clone().inner(),
            scale: pair.scale,
        }
    }
    fn conv_inner<AD: AutodiffBackend>(delta: &ConvDelta<AD>) -> ConvDelta<AD::InnerBackend> {
        ConvDelta {
            taps: delta.taps.iter().map(|t| t.clone().inner()).collect(),
        }
    }
    match layer {
        LayerAdapters::Gdn(adapters) => LayerAdapters::Gdn(GdnAdapters {
            q: pair_inner(&adapters.q),
            q_conv: conv_inner(&adapters.q_conv),
            g: pair_inner(&adapters.g),
            o: pair_inner(&adapters.o),
            gate: pair_inner(&adapters.gate),
            up: pair_inner(&adapters.up),
            down: pair_inner(&adapters.down),
        }),
        LayerAdapters::Attn(adapters) => LayerAdapters::Attn(AttnAdapters {
            q: pair_inner(&adapters.q),
            o: pair_inner(&adapters.o),
            gate: pair_inner(&adapters.gate),
            up: pair_inner(&adapters.up),
            down: pair_inner(&adapters.down),
        }),
    }
}

/// The dimension bundle off the inner model, so one dispatcher
/// serves blocks on ANY backend (the AD-lifted calls reuse it).
fn dims_of<AD: AutodiffBackend>(model: &HybridModel<AD::InnerBackend>) -> HybridDims {
    HybridDims {
        attn_heads: model.attn_heads,
        attn_head_dim: model.attn_head_dim,
        gdn_heads: model.gdn_heads,
        gdn_key_dim: model.gdn_key_dim,
        gdn_value_dim: model.gdn_value_dim,
        eps: model.eps,
    }
}

/// The dimension bundle every block forward needs.
struct HybridDims {
    attn_heads: usize,
    attn_head_dim: usize,
    gdn_heads: usize,
    gdn_key_dim: usize,
    gdn_value_dim: usize,
    eps: f64,
}

fn dims_block_forward<B: Backend>(
    dims: &HybridDims,
    block: &HybridBlock<B>,
    x: FloatTensor<B, 2>,
    mask: &FloatTensor<B, 2>,
    adapters: Option<&LayerAdapters<B>>,
    device: &B::Device,
) -> FloatTensor<B, 2> {
    match (block, adapters) {
        (HybridBlock::Gdn(block), Some(LayerAdapters::Gdn(adapters))) => block.forward(
            x,
            dims.gdn_heads,
            dims.gdn_key_dim,
            dims.gdn_value_dim,
            dims.eps,
            device,
            Some(adapters),
        ),
        (HybridBlock::Gdn(block), None) => block.forward(
            x,
            dims.gdn_heads,
            dims.gdn_key_dim,
            dims.gdn_value_dim,
            dims.eps,
            device,
            None,
        ),
        (HybridBlock::Attn(block), Some(LayerAdapters::Attn(adapters))) => block.forward(
            x,
            mask,
            dims.attn_heads,
            dims.attn_head_dim,
            dims.eps,
            Some(adapters),
        ),
        (HybridBlock::Attn(block), None) => block.forward(
            x,
            mask,
            dims.attn_heads,
            dims.attn_head_dim,
            dims.eps,
            None,
        ),
        _ => unreachable!("adapter kind always matches its layer kind by construction"),
    }
}

/// Mean next-token cross-entropy, computed head-chunk by head-chunk
/// so the (n, vocab) logits never materialize as one tensor.
fn cross_entropy_chunked<AD: AutodiffBackend>(
    hidden: FloatTensor<AD, 2>,
    lm_head_transposed: &FloatTensor<AD, 2>,
    targets: &[u32],
    chunk: usize,
    device: &AD::Device,
) -> FloatTensor<AD, 1> {
    let n = targets.len();
    let chunk = chunk.max(1);
    let mut total: Option<FloatTensor<AD, 1>> = None;
    let mut start = 0usize;
    while start < n {
        let end = (start + chunk).min(n);
        let rows = hidden.clone().narrow(0, start, end - start);
        let logits = rows.matmul(lm_head_transposed.clone());
        let log_probs = burn::tensor::activation::log_softmax(logits, 1);
        let indices: Vec<i64> = targets[start..end].iter().map(|t| *t as i64).collect();
        let index_tensor = burn::tensor::Tensor::<AD, 2, Int>::from_data(
            burn::tensor::TensorData::new(indices, [end - start, 1]),
            device,
        );
        let picked = log_probs.gather(1, index_tensor);
        let chunk_loss = picked.sum().neg();
        total = Some(match total {
            Some(accumulated) => accumulated + chunk_loss,
            None => chunk_loss,
        });
        start = end;
    }
    total
        .expect("targets are never empty (packed chunks are seq_len + 1)")
        .div_scalar(n as f64)
}

/// One training step's gradients under the per-layer VJP chain: pass
/// 1 (no grad, frozen adapter views) caches each layer's input; the
/// loss block's input gradient then seeds a top-down chain where
/// sum(out * grad_out) per layer yields that layer's adapter
/// gradients and the next input gradient.
pub(crate) fn chain_step<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &ModelAdapters<AD>,
    mask_inner: &FloatTensor<AD::InnerBackend, 2>,
    inputs: &[u32],
    targets: &[u32],
    loss_chunk: usize,
    device: &AD::Device,
) -> BquestResult<StepOutcome<AD::InnerBackend>> {
    let dims = dims_of::<AD>(model);
    let frozen: Vec<LayerAdapters<AD::InnerBackend>> =
        adapters.layers.iter().map(adapters_inner::<AD>).collect();

    // Pass 1 (no grad): cache each layer's INPUT; x ends as the last
    // layer's output (pre final norm).
    let mut layer_inputs: Vec<FloatTensor<AD::InnerBackend, 2>> =
        Vec::with_capacity(model.layers.len());
    let mut x = model.embed(inputs);
    for (block, frozen_layer) in model.layers.iter().zip(frozen.iter()) {
        layer_inputs.push(x.clone());
        x = dims_block_forward(&dims, block, x, mask_inner, Some(frozen_layer), device);
    }

    // Loss block: final norm + chunked CE as a small autodiff graph;
    // its input gradient seeds the layer chain.
    let top_input = FloatTensor::<AD, 2>::from_inner(x).require_grad();
    let hidden = hybrid::rms_norm(
        top_input.clone(),
        &FloatTensor::from_inner(model.final_norm.clone()),
        model.eps,
    );
    let loss = cross_entropy_chunked::<AD>(
        hidden,
        &FloatTensor::from_inner(model.lm_head_transposed.clone()),
        targets,
        loss_chunk,
        device,
    );
    let loss_value: f32 = loss.clone().into_scalar().elem();
    let mut top_grads = loss.backward();
    let Some(mut grad_out) = top_input.grad_remove(&mut top_grads) else {
        snafu::whatever!("no gradient reached the loss block input");
    };
    drop(top_grads);

    // Pass 2: per-layer VJP, top down; at most one layer graph live.
    let mut lora_grads: HashMap<String, FloatTensor<AD::InnerBackend, 2>> = HashMap::new();
    for (index, (block, layer_adapters)) in model
        .layers
        .iter()
        .zip(adapters.layers.iter())
        .enumerate()
        .rev()
    {
        let block_ad = lift_block::<AD>(block);
        let mask_ad = FloatTensor::<AD, 2>::from_inner(mask_inner.clone());
        let block_input =
            FloatTensor::<AD, 2>::from_inner(layer_inputs[index].clone()).require_grad();
        let out = dims_block_forward(
            &dims,
            &block_ad,
            block_input.clone(),
            &mask_ad,
            Some(layer_adapters),
            device,
        );
        let pseudo_loss = (out * FloatTensor::from_inner(grad_out.clone())).sum();
        let mut grads = pseudo_loss.backward();
        let prefix = format!("model.layers.{index}");
        for (name, tensor) in layer_adapters.params(&prefix) {
            let Some(grad) = tensor.grad_remove(&mut grads) else {
                snafu::whatever!("no gradient for {name} (graph detached?)");
            };
            lora_grads.insert(name, grad);
        }
        let Some(next_grad) = block_input.grad_remove(&mut grads) else {
            snafu::whatever!("no input gradient at layer {index}");
        };
        grad_out = next_grad;
    }

    Ok(StepOutcome { loss: loss_value, grads: lora_grads })
}

/// The full-graph reference step (toy scale only): the identical
/// forward run end to end as ONE autodiff graph - the arbiter the
/// gate compares the chain against.
fn full_graph_step<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &ModelAdapters<AD>,
    mask_inner: &FloatTensor<AD::InnerBackend, 2>,
    inputs: &[u32],
    targets: &[u32],
    loss_chunk: usize,
    device: &AD::Device,
) -> BquestResult<StepOutcome<AD::InnerBackend>> {
    let dims = dims_of::<AD>(model);
    let mask_ad = FloatTensor::<AD, 2>::from_inner(mask_inner.clone());
    let mut x = FloatTensor::<AD, 2>::from_inner(model.embed(inputs));
    for (block, layer_adapters) in model.layers.iter().zip(adapters.layers.iter()) {
        let block_ad = lift_block::<AD>(block);
        x = dims_block_forward(&dims, &block_ad, x, &mask_ad, Some(layer_adapters), device);
    }
    let hidden = hybrid::rms_norm(
        x,
        &FloatTensor::from_inner(model.final_norm.clone()),
        model.eps,
    );
    let loss = cross_entropy_chunked::<AD>(
        hidden,
        &FloatTensor::from_inner(model.lm_head_transposed.clone()),
        targets,
        loss_chunk,
        device,
    );
    let loss_value: f32 = loss.clone().into_scalar().elem();
    let mut grads = loss.backward();
    let mut lora_grads: HashMap<String, FloatTensor<AD::InnerBackend, 2>> = HashMap::new();
    for (index, layer_adapters) in adapters.layers.iter().enumerate() {
        let prefix = format!("model.layers.{index}");
        for (name, tensor) in layer_adapters.params(&prefix) {
            let Some(grad) = tensor.grad_remove(&mut grads) else {
                snafu::whatever!("no full-graph gradient for {name}");
            };
            lora_grads.insert(name, grad);
        }
    }
    Ok(StepOutcome { loss: loss_value, grads: lora_grads })
}

/// The training loop: chain steps + AdamW per raw tensor under
/// linear warmup then constant lr; `on_step` receives every step's
/// log row (the cpt verb appends the NUON-lines log there).
pub(crate) fn train_loop<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &mut ModelAdapters<AD>,
    chunks: &[Vec<u32>],
    options: &LoopOptions,
    device: &AD::Device,
    mut on_step: impl FnMut(&StepLog) -> BquestResult<()>,
) -> BquestResult<TrainReport> {
    snafu::ensure_whatever!(!chunks.is_empty(), "no training chunks");
    let seq_len = chunks[0].len() - 1;
    let mask_inner = causal_mask::<AD::InnerBackend>(seq_len, device);
    let optimizer = burn::optim::AdamWConfig::new().with_weight_decay(0.0).build();
    let mut states: HashMap<String, OptState<AD::InnerBackend>> = HashMap::new();

    let probe = std::env::var("QUEST_TRAIN_PROBE").ok();
    let mut first_loss = 0f32;
    let mut last_loss = 0f32;
    let mut trained_tokens = 0usize;
    let started = std::time::Instant::now();
    for step in 0..options.steps {
        let sample = &chunks[step % chunks.len()];
        snafu::ensure_whatever!(
            sample.len() == seq_len + 1,
            "chunk {step} carries {} ids for seq_len {seq_len}",
            sample.len()
        );
        let inputs = &sample[..seq_len];
        let targets = &sample[1..];

        let mut outcome = chain_step::<AD>(
            model,
            adapters,
            &mask_inner,
            inputs,
            targets,
            options.loss_chunk,
            device,
        )?;
        if step == 0 {
            first_loss = outcome.loss;
            // Per-layer gradient health at step 0 (the NaN
            // localizer): QUEST_TRAIN_DEBUG_GRADS=1.
            if std::env::var("QUEST_TRAIN_DEBUG_GRADS").is_ok() {
                debug_grad_health(&outcome.grads)?;
            }
            // VRAM bisection park (an external poller reads the
            // level): QUEST_TRAIN_PROBE=grads|step.
            if probe.as_deref() == Some("grads") {
                eprintln!("bquest train probe: parked at grads");
                std::thread::sleep(std::time::Duration::from_secs(3));
                return Ok(TrainReport {
                    first_loss,
                    final_loss: outcome.loss,
                    steps: 1,
                    trained_tokens: seq_len,
                    seconds: started.elapsed().as_secs_f64(),
                });
            }
        }
        last_loss = outcome.loss;

        let learning_rate = options.learning_rate
            * (((step + 1) as f64 / options.warmup_steps.max(1) as f64).min(1.0));
        for (name, tensor) in adapters.params_mut() {
            let Some(grad) = outcome.grads.remove(&name) else {
                snafu::whatever!("missing accumulated gradient for {name}");
            };
            let inner = tensor.clone().inner();
            let state = states.remove(&name);
            let (updated, state) = optimizer.step(learning_rate, inner, grad, state);
            if let Some(state) = state {
                states.insert(name, state);
            }
            *tensor = FloatTensor::from_inner(updated).require_grad();
        }
        if step == 0 && probe.as_deref() == Some("step") {
            eprintln!("bquest train probe: parked at step");
            std::thread::sleep(std::time::Duration::from_secs(3));
            return Ok(TrainReport {
                first_loss,
                final_loss: last_loss,
                steps: 1,
                trained_tokens: seq_len,
                seconds: started.elapsed().as_secs_f64(),
            });
        }

        trained_tokens += seq_len;
        let elapsed = started.elapsed().as_secs_f64();
        let log = StepLog {
            step: step + 1,
            loss: last_loss,
            learning_rate,
            tokens_per_second: trained_tokens as f64 / elapsed,
            elapsed_seconds: elapsed,
        };
        on_step(&log)?;
        if (step + 1) % options.log_every.max(1) == 0 || step + 1 == options.steps {
            eprintln!(
                "bquest train: step {} | loss {:.4} | lr {:.2e} | {:.0} tok/s",
                log.step, log.loss, log.learning_rate, log.tokens_per_second
            );
        }
    }

    Ok(TrainReport {
        first_loss,
        final_loss: last_loss,
        steps: options.steps,
        trained_tokens,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Per-layer gradient health print (step 0, QUEST_TRAIN_DEBUG_GRADS):
/// non-finite counts localize a NaN backward to its layer.
fn debug_grad_health<B: Backend>(
    grads: &HashMap<String, FloatTensor<B, 2>>,
) -> BquestResult<()> {
    let mut by_layer: HashMap<usize, (usize, usize, f64)> = HashMap::new();
    for (name, grad) in grads {
        let layer: usize = name
            .strip_prefix("model.layers.")
            .and_then(|rest| rest.split('.').next())
            .and_then(|index| index.parse().ok())
            .unwrap_or(usize::MAX);
        let values = tensor_values(grad)?;
        let entry = by_layer.entry(layer).or_insert((0, 0, 0.0));
        entry.0 += values.len();
        entry.1 += values.iter().filter(|v| !v.is_finite()).count();
        entry.2 += values
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| v.abs() as f64)
            .sum::<f64>();
    }
    let mut layers: Vec<usize> = by_layer.keys().copied().collect();
    layers.sort_unstable();
    for layer in layers {
        let (total, non_finite, magnitude) = by_layer[&layer];
        eprintln!(
            "debug grads: layer {layer} | {non_finite}/{total} non-finite | sum|g| {magnitude:.3e}"
        );
    }
    Ok(())
}

/// Read a packed-chunks artifact ("chunks" u32 [n, seq_len + 1] -
/// the mix pack contract).
fn load_chunks(path: &Path) -> BquestResult<Vec<Vec<u32>>> {
    let bytes = fs::read(path)?;
    let parsed = match safetensors::SafeTensors::deserialize(&bytes) {
        Ok(parsed) => parsed,
        Err(error) => snafu::whatever!("chunks parse failed: {error}"),
    };
    let Ok(view) = parsed.tensor("chunks") else {
        snafu::whatever!("the artifact carries no chunks tensor");
    };
    snafu::ensure_whatever!(
        view.dtype() == safetensors::Dtype::U32,
        "chunks: expected u32, got {:?}",
        view.dtype()
    );
    let shape = view.shape();
    snafu::ensure_whatever!(shape.len() == 2, "chunks: expected rank 2, got {shape:?}");
    let (rows, width) = (shape[0], shape[1]);
    let values: Vec<u32> = view
        .data()
        .chunks_exact(4)
        .map(|quad| u32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
        .collect();
    Ok((0..rows)
        .map(|row| values[row * width..(row + 1) * width].to_vec())
        .collect())
}

// ---- The toy-config gate (cpu f32) --------------------------------

/// A 2-layer toy hybrid shape (one GDN + one attention layer) small
/// enough for cpu-f32 full-graph autodiff.
pub(crate) fn toy_config() -> HybridCheckpointConfig {
    HybridCheckpointConfig {
        vocab_size: 96,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        rms_norm_eps: 1e-6,
        layer_types: vec![
            String::from("linear_attention"),
            String::from("full_attention"),
        ],
        linear_num_key_heads: 2,
        linear_num_value_heads: 2,
        linear_key_head_dim: 12,
        linear_value_head_dim: 24,
        linear_conv_kernel_dim: 4,
        tie_word_embeddings: false,
    }
}

/// Matrix init at a bounded uniform scale so activations neither
/// explode nor vanish through the toy depth.
fn toy_tensor(rng: &mut SplitMix64, shape: Vec<usize>) -> (Vec<usize>, Vec<f32>) {
    let count: usize = shape.iter().product();
    let values = (0..count)
        .map(|_| ((rng.next_u64() % 2000) as f32 / 1000.0 - 1.0) * 0.3)
        .collect();
    (shape, values)
}

/// Norm scales init near 1.0 (a small norm weight multiplies every
/// activation down and flattens the loss landscape - the fixture
/// trap the era-one gate hit).
fn toy_norm(rng: &mut SplitMix64, width: usize) -> (Vec<usize>, Vec<f32>) {
    let values = (0..width)
        .map(|_| 1.0 + ((rng.next_u64() % 2000) as f32 / 1000.0 - 1.0) * 0.05)
        .collect();
    (vec![width], values)
}

pub(crate) fn toy_weights(config: &HybridCheckpointConfig) -> HybridWeights {
    let mut rng = SplitMix64::new(7);
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let vocab = config.vocab_size;
    let key_width = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_width = config.linear_num_value_heads * config.linear_value_head_dim;
    let kernel = config.linear_conv_kernel_dim;
    let gdn_heads = config.linear_num_key_heads;
    let mut tensors = HashMap::new();
    tensors.insert(
        String::from("model.embed_tokens.weight"),
        toy_tensor(&mut rng, vec![vocab, hidden]),
    );
    for index in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{index}");
        let gdn = config.is_gdn_layer(index).expect("toy layer kind");
        for name in ["gate_proj", "up_proj"] {
            tensors.insert(
                format!("{prefix}.mlp.{name}.weight"),
                toy_tensor(&mut rng, vec![intermediate, hidden]),
            );
        }
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            toy_tensor(&mut rng, vec![hidden, intermediate]),
        );
        if gdn {
            tensors.insert(
                format!("{prefix}.input_layernorm.weight"),
                toy_norm(&mut rng, hidden),
            );
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                toy_norm(&mut rng, hidden),
            );
            for name in ["q_proj", "k_proj"] {
                tensors.insert(
                    format!("{prefix}.linear_attn.{name}.weight"),
                    toy_tensor(&mut rng, vec![key_width, hidden]),
                );
            }
            for name in ["v_proj", "g_proj"] {
                tensors.insert(
                    format!("{prefix}.linear_attn.{name}.weight"),
                    toy_tensor(&mut rng, vec![value_width, hidden]),
                );
            }
            tensors.insert(
                format!("{prefix}.linear_attn.o_proj.weight"),
                toy_tensor(&mut rng, vec![hidden, value_width]),
            );
            for name in ["a_proj", "b_proj"] {
                tensors.insert(
                    format!("{prefix}.linear_attn.{name}.weight"),
                    toy_tensor(&mut rng, vec![gdn_heads, hidden]),
                );
            }
            for (name, channels) in [
                ("q_conv1d", key_width),
                ("k_conv1d", key_width),
                ("v_conv1d", value_width),
            ] {
                tensors.insert(
                    format!("{prefix}.linear_attn.{name}.weight"),
                    toy_tensor(&mut rng, vec![channels, 1, kernel]),
                );
            }
            tensors.insert(
                format!("{prefix}.linear_attn.A_log"),
                toy_tensor(&mut rng, vec![gdn_heads]),
            );
            tensors.insert(
                format!("{prefix}.linear_attn.dt_bias"),
                toy_tensor(&mut rng, vec![gdn_heads]),
            );
            tensors.insert(
                format!("{prefix}.linear_attn.o_norm.weight"),
                toy_norm(&mut rng, config.linear_value_head_dim),
            );
        } else {
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
                format!("{prefix}.post_attention_layernorm.weight"),
                toy_norm(&mut rng, hidden),
            );
            tensors.insert(
                format!("{prefix}.post_feedforward_layernorm.weight"),
                toy_norm(&mut rng, hidden),
            );
        }
    }
    tensors.insert(String::from("model.norm.weight"), toy_norm(&mut rng, hidden));
    tensors.insert(
        String::from("lm_head.weight"),
        toy_tensor(&mut rng, vec![vocab, hidden]),
    );
    HybridWeights::from_tensors(tensors)
}

/// One gradient-set comparison (chain vs full graph): worst per-param
/// nmse, with all-zero reference grads matched against all-zero chain
/// grads (b starts zero, so a-grads are exactly zero until b moves).
pub(crate) struct GradCompare {
    pub(crate) max_nmse: f64,
    pub(crate) compared: usize,
    pub(crate) zero_matched: usize,
}

fn compare_grads<B: Backend>(
    chain: &HashMap<String, FloatTensor<B, 2>>,
    full: &HashMap<String, FloatTensor<B, 2>>,
) -> BquestResult<GradCompare> {
    snafu::ensure_whatever!(
        chain.len() == full.len(),
        "gradient sets differ in size ({} chain vs {} full)",
        chain.len(),
        full.len()
    );
    let mut max_nmse = 0f64;
    let mut compared = 0usize;
    let mut zero_matched = 0usize;
    for (name, reference) in full {
        let Some(candidate) = chain.get(name) else {
            snafu::whatever!("chain gradients are missing {name}");
        };
        let reference_values = tensor_values(reference)?;
        let candidate_values = tensor_values(candidate)?;
        snafu::ensure_whatever!(
            reference_values.len() == candidate_values.len(),
            "{name}: gradient shapes differ"
        );
        let denominator: f64 = reference_values.iter().map(|v| (*v as f64).powi(2)).sum();
        let numerator: f64 = reference_values
            .iter()
            .zip(candidate_values.iter())
            .map(|(r, c)| (*r as f64 - *c as f64).powi(2))
            .sum();
        if denominator == 0.0 {
            snafu::ensure_whatever!(
                numerator == 0.0,
                "{name}: reference gradient is zero but the chain's is not"
            );
            zero_matched += 1;
            continue;
        }
        max_nmse = max_nmse.max(numerator / denominator);
        compared += 1;
    }
    Ok(GradCompare { max_nmse, compared, zero_matched })
}

fn tensor_values<B: Backend>(tensor: &FloatTensor<B, 2>) -> BquestResult<Vec<f32>> {
    let data = tensor.clone().into_data().convert::<f32>();
    match data.to_vec::<f32>() {
        Ok(values) => Ok(values),
        Err(error) => snafu::whatever!("gradient extraction failed: {error:?}"),
    }
}

/// The gate verdict: the three toy locks' readings.
pub(crate) struct GateVerdict {
    pub(crate) adapter_off_exact: bool,
    pub(crate) init_grads: GradCompare,
    pub(crate) trained_grads: GradCompare,
    pub(crate) first_loss: f32,
    pub(crate) final_loss: f32,
    pub(crate) descended: bool,
}

/// The chain-vs-full gradient bar (f32 reassociation class; the two
/// graph shapes compute the same math in different orders).
const GATE_GRAD_NMSE_BAR: f64 = 1e-9;
/// The descent margin (60 steps over 4 repeating chunks memorizes
/// them; a broken gradient path cannot cross it by noise).
const GATE_DESCENT_MARGIN: f32 = 0.05;

/// Run the toy-config locks on cpu f32: (1) zero-init adapters leave
/// the forward value-equal to the adapter-free oracle path; (2) the
/// per-layer VJP chain's gradients equal a full-graph backward's, at
/// init AND after five optimizer steps; (3) 60 steps descend.
pub(crate) fn run_train_gate() -> BquestResult<GateVerdict> {
    type ToyAd = burn::backend::Autodiff<CpuBack>;
    let device: <CpuBack as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let config = toy_config();
    let model = HybridModel::<CpuBack>::new(&config, toy_weights(&config), device)?;

    let seq_len = 16usize;
    let mut rng = SplitMix64::new(42);
    let chunks: Vec<Vec<u32>> = (0..4)
        .map(|_| {
            (0..seq_len + 1)
                .map(|_| (rng.next_u64() % config.vocab_size as u64) as u32)
                .collect()
        })
        .collect();
    let mask_inner = causal_mask::<CpuBack>(seq_len, &device);
    let dims = dims_of::<ToyAd>(&model);

    // Lock 1: adapter-off exactness - fresh zero-b adapters must
    // leave every hidden value equal to the None path.
    let adapters = ModelAdapters::<ToyAd>::init(&config, 4, 16.0, 99, &device)?;
    let frozen: Vec<LayerAdapters<CpuBack>> =
        adapters.layers.iter().map(adapters_inner::<ToyAd>).collect();
    let ids = &chunks[0][..seq_len];
    let mut plain = model.embed(ids);
    let mut adapted = plain.clone();
    for (block, frozen_layer) in model.layers.iter().zip(frozen.iter()) {
        plain = dims_block_forward(&dims, block, plain, &mask_inner, None, &device);
        adapted =
            dims_block_forward(&dims, block, adapted, &mask_inner, Some(frozen_layer), &device);
    }
    let plain_values = tensor_values(&plain)?;
    let adapted_values = tensor_values(&adapted)?;
    let adapter_off_exact = plain_values == adapted_values;

    // Lock 2: chain-vs-full gradient equivalence at init (b zero) and
    // after five real optimizer steps (b live).
    let targets = &chunks[0][1..];
    let chain = chain_step::<ToyAd>(
        &model, &adapters, &mask_inner, ids, targets, 8, &device,
    )?;
    let full = full_graph_step::<ToyAd>(
        &model, &adapters, &mask_inner, ids, targets, 8, &device,
    )?;
    snafu::ensure_whatever!(
        (chain.loss - full.loss).abs() <= 1e-6,
        "chain and full-graph losses diverge ({} vs {})",
        chain.loss,
        full.loss
    );
    let init_grads = compare_grads(&chain.grads, &full.grads)?;

    let mut adapters = adapters;
    let warm_options = LoopOptions {
        steps: 5,
        learning_rate: 5e-2,
        warmup_steps: 2,
        loss_chunk: 8,
        log_every: usize::MAX,
    };
    train_loop::<ToyAd>(&model, &mut adapters, &chunks, &warm_options, &device, |_| Ok(()))?;
    let chain = chain_step::<ToyAd>(
        &model, &adapters, &mask_inner, ids, targets, 8, &device,
    )?;
    let full = full_graph_step::<ToyAd>(
        &model, &adapters, &mask_inner, ids, targets, 8, &device,
    )?;
    let trained_grads = compare_grads(&chain.grads, &full.grads)?;

    // Lock 3: descent over fresh adapters.
    let mut adapters = ModelAdapters::<ToyAd>::init(&config, 4, 16.0, 42, &device)?;
    let descent_options = LoopOptions {
        steps: 60,
        learning_rate: 5e-2,
        warmup_steps: 2,
        loss_chunk: 8,
        log_every: usize::MAX,
    };
    let report = train_loop::<ToyAd>(
        &model,
        &mut adapters,
        &chunks,
        &descent_options,
        &device,
        |_| Ok(()),
    )?;

    Ok(GateVerdict {
        adapter_off_exact,
        init_grads,
        trained_grads,
        first_loss: report.first_loss,
        final_loss: report.final_loss,
        descended: report.final_loss < report.first_loss - GATE_DESCENT_MARGIN,
    })
}

/// `bquest train gate`: run the toy locks and print the NUON verdict;
/// any failing lock exits nonzero.
pub(crate) fn train_gate_verb() -> BquestResult<()> {
    let verdict = run_train_gate()?;
    let passed = verdict.adapter_off_exact
        && verdict.init_grads.max_nmse <= GATE_GRAD_NMSE_BAR
        && verdict.trained_grads.max_nmse <= GATE_GRAD_NMSE_BAR
        && verdict.descended;
    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "passed" => v_bool(passed),
            "adapter_off_exact" => v_bool(verdict.adapter_off_exact),
            "init_grad_nmse_max" => v_float(verdict.init_grads.max_nmse),
            "init_grads_compared" => v_int(verdict.init_grads.compared as i64),
            "init_grads_zero_matched" => v_int(verdict.init_grads.zero_matched as i64),
            "trained_grad_nmse_max" => v_float(verdict.trained_grads.max_nmse),
            "trained_grads_compared" => v_int(verdict.trained_grads.compared as i64),
            "grad_nmse_bar" => v_float(GATE_GRAD_NMSE_BAR),
            "first_loss" => v_float(verdict.first_loss as f64),
            "final_loss" => v_float(verdict.final_loss as f64),
            "descended" => v_bool(verdict.descended),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    snafu::ensure_whatever!(passed, "the train gate failed");
    Ok(())
}

/// `bquest train cpt`: LoRA CPT over a packed-chunks artifact on the
/// train-cuda grade; the adapter lands as one safetensors file and
/// every step appends to the NUON-lines log.
pub(crate) fn train_cpt(cli: &Cli, args: &TrainCptArgs) -> BquestResult<()> {
    let config = lib::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let model_dir = config.model_dir();
    let model_id = config.model.clone();
    let alpha = args.alpha.unwrap_or(2.0 * args.rank as f64);
    let chunks = load_chunks(&args.chunks)?;
    let steps = args.steps.unwrap_or(chunks.len());
    let log_path = match &args.log {
        Some(path) => path.clone(),
        None => {
            let stem = args
                .out
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| String::from("adapter"));
            args.out.with_file_name(format!("{stem}_steps.nuon"))
        }
    };
    eprintln!(
        "bquest train cpt: {} chunks of {}, {} steps, rank {} alpha {alpha}",
        chunks.len(),
        chunks[0].len() - 1,
        steps,
        args.rank
    );

    #[cfg(feature = "train-cuda")]
    {
        type TrainAd = burn::backend::Autodiff<CudaBack>;
        let device: <CudaBack as burn::tensor::backend::BackendTypes>::Device =
            Default::default();
        let hybrid_config = HybridCheckpointConfig::load(&model_dir)?;
        let load_start = std::time::Instant::now();
        let weights = HybridWeights::load(&model_dir)?;
        let model = HybridModel::<CudaBack>::new(&hybrid_config, weights, device.clone())?;
        eprintln!(
            "bquest train cpt: model built in {:.1}s",
            load_start.elapsed().as_secs_f32()
        );
        let mut adapters =
            ModelAdapters::<TrainAd>::init(&hybrid_config, args.rank, alpha, args.seed, &device)?;
        let options = LoopOptions {
            steps,
            learning_rate: args.learning_rate,
            warmup_steps: args.warmup_steps,
            loss_chunk: args.loss_chunk,
            log_every: args.log_every,
        };
        let report = train_loop::<TrainAd>(
            &model,
            &mut adapters,
            &chunks,
            &options,
            &device,
            |log| {
                let row = lib::nu::Value::record(
                    lib::nu::record! {
                        "step" => v_int(log.step as i64),
                        "loss" => v_float(log.loss as f64),
                        "learning_rate" => v_float(log.learning_rate),
                        "tokens_per_second" => v_float(log.tokens_per_second),
                        "elapsed_seconds" => v_float(log.elapsed_seconds),
                    },
                    span(),
                );
                lib::nu::append_line(&log_path, &row)?;
                Ok(())
            },
        )?;
        adapters.save(&args.out, &model_id)?;
        let summary = lib::nu::Value::record(
            lib::nu::record! {
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
        println!("{}", lib::nu::to_nuon_text(&summary)?);
        Ok(())
    }
    #[cfg(not(feature = "train-cuda"))]
    {
        let _ = (model_dir, model_id, alpha, steps, log_path);
        snafu::whatever!("bquest train cpt runs on cuda (rebuild with --features train-cuda)")
    }
}
