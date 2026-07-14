use crate::*;

use burn::optim::{
    AdamW,
    AdamWConfig,
    SimpleOptimizer,
};
use burn::tensor::{
    ElementConversion,
    Int,
    Tensor,
    TensorData,
    activation,
    backend::AutodiffBackend,
    backend::Backend,
};

/// Per-tensor AdamW state, keyed by the lora parameter name.
type OptState<B> = <AdamW as SimpleOptimizer<B>>::State<2>;

/// Knobs for one LoRA-CPT run (plain LM loss over packed chunks).
/// (Without cuda the cli path bails before the loop, so the loop-side
/// items read dead there; they stay live for cuda builds and the toy
/// cpu unit test.)
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct TrainOptions {
    pub(crate) data_roots: Vec<PathBuf>,
    pub(crate) exclude_dirs: Vec<String>,
    pub(crate) tokenizer: PathBuf,
    pub(crate) model_dir: PathBuf,
    pub(crate) out: PathBuf,
    pub(crate) seq_len: usize,
    pub(crate) steps: usize,
    pub(crate) rank: usize,
    pub(crate) alpha: f64,
    pub(crate) learning_rate: f64,
    pub(crate) warmup_steps: usize,
    pub(crate) seed: u64,
    pub(crate) loss_chunk: usize,
    pub(crate) log_every: usize,
}

/// First/final loss of a run - the descent sanity signal the smoke (and
/// the toy-config unit test) asserts on.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct TrainReport {
    pub(crate) first_loss: f32,
    pub(crate) final_loss: f32,
}

/// Load corpus + checkpoint, then train on the cuda backend.
pub(crate) fn train(options: &TrainOptions) -> HeatResult<()> {
    let tokenizer = match tokenizers::Tokenizer::from_file(&options.tokenizer) {
        Ok(tokenizer) => tokenizer,
        Err(error) => snafu::whatever!("loading tokenizer failed: {error}"),
    };
    let Some(eos) = tokenizer.token_to_id("<|endoftext|>") else {
        snafu::whatever!("tokenizer carries no <|endoftext|> token");
    };
    let packed = train_data::load_packed(
        &options.data_roots,
        &options.exclude_dirs,
        &tokenizer,
        options.seq_len,
        eos,
        options.seed,
    )?;
    eprintln!(
        "heat train: {} documents, {} tokens, {} chunks of {}",
        packed.documents,
        packed.total_tokens,
        packed.chunks.len(),
        options.seq_len
    );

    let config: HeatConfig = serde_json::from_reader(std::fs::File::open(
        options.model_dir.join("config.json"),
    )?)?;
    let shards = dump::shard_paths(&options.model_dir)?;
    let load_start = Instant::now();
    let weights = Weights::load(&shards)?;
    eprintln!(
        "heat train: weights read + converted to host f32 in {:.1}s",
        load_start.elapsed().as_secs_f32()
    );

    #[cfg(any(feature = "cuda", feature = "train-cuda"))]
    {
        use burn::backend::Autodiff;
        let report =
            train_on::<Autodiff<CudaBack>>(&config, weights, Default::default(), &packed, options)?;
        eprintln!(
            "heat train: loss {:.4} -> {:.4}",
            report.first_loss, report.final_loss
        );
        Ok(())
    }
    #[cfg(not(any(feature = "cuda", feature = "train-cuda")))]
    {
        let _ = (config, weights, packed);
        snafu::whatever!("heat train runs on cuda (rebuild with --features train-cuda)")
    }
}

/// The backend-generic loop under MANUAL PER-LAYER CHECKPOINTING: a
/// no-grad pass over the frozen model caches each layer's input, then
/// the loss block and each layer backward run as their own small
/// autodiff graphs chained by vector-Jacobian products - at most one
/// layer's graph is ever live, so residency stays near the weights
/// (full-graph autodiff over 7B measured ~7.5 GiB of retained forward
/// on this stack - card overflow beside 14 GiB of weights).
#[cfg_attr(not(any(feature = "cuda", feature = "train-cuda")), allow(dead_code))]
pub(crate) fn train_on<AD: AutodiffBackend>(
    config: &HeatConfig,
    weights: Weights,
    device: AD::Device,
    packed: &Packed,
    options: &TrainOptions,
) -> HeatResult<TrainReport> {
    // VRAM bisection probe: ALMOST_TRAIN_PROBE=build|forward|loss|step
    // completes that stage of step 0, parks 3s (an external poller reads
    // the level), and exits early with the report so far.
    let probe = std::env::var("ALMOST_TRAIN_PROBE").ok();
    let probe_stop = |stage: &str| -> bool {
        if probe.as_deref() == Some(stage) {
            eprintln!("heat train probe: parked at {stage}");
            std::thread::sleep(std::time::Duration::from_secs(3));
            true
        } else {
            false
        }
    };

    // The model lives on the INNER backend (frozen, no graphs); the
    // checkpointed blocks wrap the tensors they need per step.
    let build_start = Instant::now();
    let model = HeatModel::<AD::InnerBackend>::new(config, weights, device.clone())?;
    eprintln!(
        "heat train: model built in {:.1}s",
        build_start.elapsed().as_secs_f32()
    );
    if probe_stop("build") {
        return Ok(TrainReport {
            first_loss: 0.0,
            final_loss: 0.0,
        });
    }
    let mut lora = ModelLora::<AD>::new(config, options.rank, options.alpha, &device);
    let optimizer = AdamWConfig::new().with_weight_decay(0.0).build();
    let mut states: HashMap<String, OptState<AD::InnerBackend>> = HashMap::new();

    // Fixed sequence shape: masks and rope slices build once.
    let n = options.seq_len;
    let full_mask = model.mask(n, None);
    let sliding_mask = model.mask(n, Some(model.window));
    let (cos_full, sin_full) = model.table_slice(&model.cos_full, &model.sin_full, n);
    let (cos_sliding, sin_sliding) =
        model.table_slice(&model.cos_sliding, &model.sin_sliding, n);

    let mut first_loss = 0f32;
    let mut last_loss = 0f32;
    let mut trained_tokens = 0usize;
    let started = Instant::now();
    for step in 0..options.steps {
        let sample = &packed.chunks[step % packed.chunks.len()];
        let inputs = &sample[..n];
        let targets = &sample[1..];

        // Per-step frozen views of the current lora values for the
        // no-grad pass (params change every step).
        let lora_frozen: Vec<LayerLora<AD::InnerBackend>> = lora
            .layers
            .iter()
            .map(layer_lora_inner::<AD>)
            .collect();
        let lora_scale = lora.scale();

        // Pass 1 (no grad): cache each layer's INPUT; x becomes the last
        // layer's output (pre final norm).
        let mut layer_inputs: Vec<Tensor<AD::InnerBackend, 2>> =
            Vec::with_capacity(model.layers.len());
        let mut x = embed::<AD::InnerBackend>(&model, inputs);
        for (layer, frozen) in model.layers.iter().zip(lora_frozen.iter()) {
            layer_inputs.push(x.clone());
            let (mask, cos, sin) = pick_family::<AD::InnerBackend>(
                layer,
                &full_mask,
                &sliding_mask,
                &cos_full,
                &sin_full,
                &cos_sliding,
                &sin_sliding,
            );
            x = layer_forward_lora(
                layer, frozen, lora_scale, x, mask, cos, sin, model.heads, model.head_dim,
                model.eps,
            );
        }
        if step == 0 && probe.is_some() {
            let _ = x.clone().into_data();
            if probe_stop("forward") {
                return Ok(TrainReport {
                    first_loss: 0.0,
                    final_loss: 0.0,
                });
            }
        }

        // Loss block: final norm + chunked CE as a small autodiff graph;
        // its input gradient seeds the layer chain.
        let top_input = Tensor::<AD, 2>::from_inner(x).require_grad();
        let hidden = model::rms_norm(
            top_input.clone(),
            &Tensor::from_inner(model.final_norm.clone()),
            model.eps,
        );
        let loss = cross_entropy_chunked(
            hidden,
            &Tensor::from_inner(model.lm_head_transposed.clone()),
            targets,
            options.loss_chunk,
            &device,
        );
        let loss_value: f32 = loss.clone().into_scalar().elem();
        if step == 0 {
            first_loss = loss_value;
        }
        last_loss = loss_value;
        let mut top_grads = loss.backward();
        let Some(mut grad_out) = top_input.grad_remove(&mut top_grads) else {
            snafu::whatever!("no gradient reached the loss block input");
        };
        drop(top_grads);
        if step == 0 && probe_stop("loss") {
            return Ok(TrainReport {
                first_loss: loss_value,
                final_loss: loss_value,
            });
        }

        // Pass 2: per-layer VJP, top down. sum(out * grad_out) has the
        // gradients of the true loss wrt this layer's inputs and lora
        // params; at most one layer graph is live.
        let mut lora_grads: HashMap<String, Tensor<AD::InnerBackend, 2>> = HashMap::new();
        for (index, (layer, adapters)) in model
            .layers
            .iter()
            .zip(lora.layers.iter())
            .enumerate()
            .rev()
        {
            let (mask, cos, sin) = pick_family::<AD::InnerBackend>(
                layer,
                &full_mask,
                &sliding_mask,
                &cos_full,
                &sin_full,
                &cos_sliding,
                &sin_sliding,
            );
            let layer_ad = layer_to_autodiff::<AD>(layer);
            let block_input =
                Tensor::<AD, 2>::from_inner(layer_inputs[index].clone()).require_grad();
            let out = layer_forward_lora(
                &layer_ad,
                adapters,
                lora_scale,
                block_input.clone(),
                &Tensor::from_inner(mask.clone()),
                &Tensor::from_inner(cos.clone()),
                &Tensor::from_inner(sin.clone()),
                model.heads,
                model.head_dim,
                model.eps,
            );
            let pseudo_loss = (out * Tensor::from_inner(grad_out.clone())).sum();
            let mut grads = pseudo_loss.backward();
            for (name, tensor) in lora_layer_params(index, adapters) {
                let Some(grad) = tensor.grad_remove(&mut grads) else {
                    snafu::whatever!("no gradient for {name} (graph detached?)");
                };
                lora_grads.insert(name, grad);
            }
            let Some(next_grad) = block_input.grad_remove(&mut grads) else {
                snafu::whatever!("no input gradient at layer {index}");
            };
            if step == 0 && std::env::var("ALMOST_TRAIN_DEBUG_GRADS").is_ok() {
                let qb: f32 = lora_grads
                    .get(&format!("layers.{index}.q.b"))
                    .map(|g| g.clone().abs().sum().into_scalar().elem())
                    .unwrap_or(-1.0);
                let upstream: f32 = grad_out.clone().abs().sum().into_scalar().elem();
                let downstream: f32 = next_grad.clone().abs().sum().into_scalar().elem();
                eprintln!(
                    "debug grads: layer {index} | q.b {qb:.3e} | grad_in {upstream:.3e} | grad_out {downstream:.3e}"
                );
            }
            grad_out = next_grad;
        }

        let lr = options.learning_rate
            * (((step + 1) as f64 / options.warmup_steps.max(1) as f64).min(1.0));
        for (name, tensor) in lora.params_mut() {
            let Some(grad) = lora_grads.remove(&name) else {
                snafu::whatever!("missing accumulated gradient for {name}");
            };
            let inner = tensor.clone().inner();
            let state = states.remove(&name);
            let (updated, state) = optimizer.step(lr, inner, grad, state);
            if let Some(state) = state {
                states.insert(name, state);
            }
            *tensor = Tensor::from_inner(updated).require_grad();
        }
        if step == 0 && probe_stop("step") {
            return Ok(TrainReport {
                first_loss: loss_value,
                final_loss: loss_value,
            });
        }

        trained_tokens += n;
        if (step + 1) % options.log_every == 0 || step + 1 == options.steps {
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "heat train: step {} | loss {loss_value:.4} | lr {lr:.2e} | {:.0} tok/s",
                step + 1,
                trained_tokens as f64 / elapsed
            );
        }
    }

    lora.save(&options.out)?;
    eprintln!("heat train: adapter written to {}", options.out.display());
    Ok(TrainReport {
        first_loss,
        final_loss: last_loss,
    })
}

/// Host-side embedding gather (the table is frozen and host-resident).
fn embed<B: Backend>(model: &HeatModel<B>, ids: &[u32]) -> Tensor<B, 2> {
    let n = ids.len();
    let mut gathered = Vec::with_capacity(n * model.embed_hidden);
    for id in ids {
        let start = (*id as usize) * model.embed_hidden;
        gathered.extend_from_slice(&model.embed_rows[start..start + model.embed_hidden]);
    }
    Tensor::from_data(TensorData::new(gathered, [n, model.embed_hidden]), &model.device)
}

/// The visibility/rope family a layer attends under.
#[allow(clippy::too_many_arguments)]
fn pick_family<'t, B: Backend>(
    layer: &HeatLayer<B>,
    full_mask: &'t Tensor<B, 2>,
    sliding_mask: &'t Tensor<B, 2>,
    cos_full: &'t Tensor<B, 2>,
    sin_full: &'t Tensor<B, 2>,
    cos_sliding: &'t Tensor<B, 2>,
    sin_sliding: &'t Tensor<B, 2>,
) -> (&'t Tensor<B, 2>, &'t Tensor<B, 2>, &'t Tensor<B, 2>) {
    match layer.kind {
        LayerKind::Full => (full_mask, cos_full, sin_full),
        LayerKind::Sliding => (sliding_mask, cos_sliding, sin_sliding),
    }
}

/// Zero-copy autodiff view of a frozen layer (untracked constants).
fn layer_to_autodiff<AD: AutodiffBackend>(layer: &HeatLayer<AD::InnerBackend>) -> HeatLayer<AD> {
    HeatLayer {
        q_transposed: Tensor::from_inner(layer.q_transposed.clone()),
        k_transposed: Tensor::from_inner(layer.k_transposed.clone()),
        v_transposed: Tensor::from_inner(layer.v_transposed.clone()),
        o_transposed: Tensor::from_inner(layer.o_transposed.clone()),
        q_norm: Tensor::from_inner(layer.q_norm.clone()),
        k_norm: Tensor::from_inner(layer.k_norm.clone()),
        gate_transposed: Tensor::from_inner(layer.gate_transposed.clone()),
        up_transposed: Tensor::from_inner(layer.up_transposed.clone()),
        down_transposed: Tensor::from_inner(layer.down_transposed.clone()),
        post_attention_norm: Tensor::from_inner(layer.post_attention_norm.clone()),
        post_feedforward_norm: Tensor::from_inner(layer.post_feedforward_norm.clone()),
        kind: layer.kind,
    }
}

/// Frozen (inner-backend) view of one layer's current lora values, for
/// the no-grad pass.
fn layer_lora_inner<AD: AutodiffBackend>(
    adapters: &LayerLora<AD>,
) -> LayerLora<AD::InnerBackend> {
    fn pair_inner<AD: AutodiffBackend>(pair: &LoraPair<AD>) -> LoraPair<AD::InnerBackend> {
        LoraPair {
            a: pair.a.clone().inner(),
            b: pair.b.clone().inner(),
        }
    }
    LayerLora {
        q: pair_inner(&adapters.q),
        o: pair_inner(&adapters.o),
        gate: pair_inner(&adapters.gate),
        up: pair_inner(&adapters.up),
        down: pair_inner(&adapters.down),
    }
}

/// One layer's named lora tensors (matches ModelLora::params_mut keys).
fn lora_layer_params<B: Backend>(
    index: usize,
    adapters: &LayerLora<B>,
) -> Vec<(String, &Tensor<B, 2>)> {
    let targets = [
        ("q", &adapters.q),
        ("o", &adapters.o),
        ("gate", &adapters.gate),
        ("up", &adapters.up),
        ("down", &adapters.down),
    ];
    let mut params = Vec::with_capacity(10);
    for (target, pair) in targets {
        params.push((format!("layers.{index}.{target}.a"), &pair.a));
        params.push((format!("layers.{index}.{target}.b"), &pair.b));
    }
    params
}

/// One post-norm block with lora terms on q/o/gate/up/down; k and v ride
/// the frozen base only. Backend-generic: the no-grad pass runs it on
/// the inner backend, the VJP blocks on the autodiff backend.
#[allow(clippy::too_many_arguments)]
fn layer_forward_lora<B: Backend>(
    layer: &HeatLayer<B>,
    adapters: &LayerLora<B>,
    lora_scale: f64,
    x: Tensor<B, 2>,
    mask: &Tensor<B, 2>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
    heads: usize,
    head_dim: usize,
    eps: f64,
) -> Tensor<B, 2> {
    let n = x.dims()[0];
    let hidden = heads * head_dim;
    let residual = x.clone();

    let q = model::rms_norm(
        x.clone().matmul(layer.q_transposed.clone()) + adapters.q.contribution(&x, lora_scale),
        &layer.q_norm,
        eps,
    );
    let k = model::rms_norm(
        x.clone().matmul(layer.k_transposed.clone()),
        &layer.k_norm,
        eps,
    );
    let v = x.clone().matmul(layer.v_transposed.clone());

    let q = model::to_heads(q, n, heads, head_dim);
    let k = model::to_heads(k, n, heads, head_dim);
    let v = model::to_heads(v, n, heads, head_dim);

    let q = model::apply_rope(q, cos, sin, heads);
    let k = model::apply_rope(k, cos, sin, heads);

    let attn_scale = 1.0 / (head_dim as f64).sqrt();
    let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(attn_scale)
        + mask.clone().unsqueeze::<3>().expand([heads, n, n]);
    let attention = activation::softmax(scores, 2);
    let context = attention
        .matmul(v)
        .swap_dims(0, 1)
        .reshape([n, hidden]);
    let projected = context.clone().matmul(layer.o_transposed.clone())
        + adapters.o.contribution(&context, lora_scale);
    let x = residual + model::rms_norm(projected, &layer.post_attention_norm, eps);

    let residual = x.clone();
    let gate = activation::silu(
        x.clone().matmul(layer.gate_transposed.clone())
            + adapters.gate.contribution(&x, lora_scale),
    );
    let up = x.clone().matmul(layer.up_transposed.clone())
        + adapters.up.contribution(&x, lora_scale);
    let activated = gate * up;
    let feedforward = activated.clone().matmul(layer.down_transposed.clone())
        + adapters.down.contribution(&activated, lora_scale);
    residual + model::rms_norm(feedforward, &layer.post_feedforward_norm, eps)
}

/// Mean next-token cross-entropy, computed head-chunk by head-chunk so
/// the (n, vocab) logits never materialize as one tensor.
fn cross_entropy_chunked<AD: AutodiffBackend>(
    hidden: Tensor<AD, 2>,
    lm_head_transposed: &Tensor<AD, 2>,
    targets: &[u32],
    chunk: usize,
    device: &AD::Device,
) -> Tensor<AD, 1> {
    let n = targets.len();
    let chunk = chunk.max(1);
    let mut total: Option<Tensor<AD, 1>> = None;
    let mut start = 0usize;
    while start < n {
        let end = (start + chunk).min(n);
        let rows = hidden.clone().slice_dim(0, start..end);
        let logits = rows.matmul(lm_head_transposed.clone());
        let log_probs = activation::log_softmax(logits, 1);
        let indices: Vec<i64> = targets[start..end].iter().map(|t| *t as i64).collect();
        let index_tensor = Tensor::<AD, 2, Int>::from_data(
            TensorData::new(indices, [end - start, 1]),
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
