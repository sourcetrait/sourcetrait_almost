//! The trainer stages: LoRA training under manual per-layer checkpointing.
use crate::*;

use burn::optim::SimpleOptimizer;
use burn::tensor::{
    ElementConversion,
    Int,
    backend::AutodiffBackend,
    backend::Backend,
};

type FloatTensor<B, const D: usize> = burn::tensor::Tensor<B, D>;
pub type OptState<B> = <burn::optim::AdamW as SimpleOptimizer<B>>::State<2>;

/// The supervised loop's per-pass reshuffle seed (deterministic).
const SFT_SHUFFLE_SEED: u64 = 0x5EED_0001;

/// The knobs every stage loop shares.
pub struct LoopOptions {
    pub steps: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub loss_chunk: usize,
    pub log_every: usize,
}

/// One step's log row, as one NUON-lines record.
#[cfg_attr(not(feature = "train-cuda"), allow(dead_code))]
pub struct StepLog {
    pub step: usize,
    pub loss: f32,
    pub learning_rate: f64,
    pub tokens_per_second: f64,
    pub elapsed_seconds: f64,
}

/// First and final loss plus run totals.
#[cfg_attr(not(feature = "train-cuda"), allow(dead_code))]
pub struct TrainReport {
    pub first_loss: f32,
    pub final_loss: f32,
    pub steps: usize,
    pub trained_tokens: usize,
    pub seconds: f64,
}

/// One step's loss and its adapter gradients, keyed by artifact name.
pub struct StepOutcome<B: Backend> {
    pub loss: f32,
    pub grads: HashMap<String, FloatTensor<B, 2>>,
}

/// Zero-copy autodiff view of a frozen GDN block, as untracked constants.
fn lift_gdn<AD: AutodiffBackend>(block: &GdnBlock<AD::InnerBackend>) -> GdnBlock<AD> {
    GdnBlock {
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
    }
}

/// Zero-copy autodiff view of a frozen attention block.
fn lift_attn<AD: AutodiffBackend>(block: &AttnBlock<AD::InnerBackend>) -> AttnBlock<AD> {
    AttnBlock {
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
    }
}

/// Zero-copy autodiff view of a frozen block, as untracked constants.
fn lift_block<AD: AutodiffBackend>(
    block: &HybridBlock<AD::InnerBackend>,
) -> HybridBlock<AD> {
    match block {
        HybridBlock::Gdn(block) => HybridBlock::Gdn(lift_gdn::<AD>(block)),
        HybridBlock::Attn(block) => HybridBlock::Attn(lift_attn::<AD>(block)),
    }
}

/// Inner-backend view of one layer's adapters, for the no-grad pass.
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

/// The dimension bundle read off the inner model.
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

/// Mean next-token cross-entropy, chunked over the head.
fn cross_entropy_chunked<AD: AutodiffBackend>(
    hidden: FloatTensor<AD, 2>,
    lm_head_transposed: &FloatTensor<AD, 2>,
    head_delta: Option<&HeadDelta<AD>>,
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
        let logits = rows.clone().matmul(lm_head_transposed.clone());
        let logits = match head_delta {
            Some(delta) => objective::apply_head_delta::<AD>(logits, &rows, delta),
            None => logits,
        };
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

/// Pass one, no grad: cache each layer's input and the top output.
pub fn forward_cache<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &ModelAdapters<AD>,
    mask_inner: &FloatTensor<AD::InnerBackend, 2>,
    inputs: &[u32],
    device: &AD::Device,
) -> (
    Vec<FloatTensor<AD::InnerBackend, 2>>,
    FloatTensor<AD::InnerBackend, 2>,
) {
    let dims = dims_of::<AD>(model);
    let frozen: Vec<LayerAdapters<AD::InnerBackend>> =
        adapters.layers.iter().map(adapters_inner::<AD>).collect();
    let mut layer_inputs: Vec<FloatTensor<AD::InnerBackend, 2>> =
        Vec::with_capacity(model.layers.len());
    let mut x = model.embed(inputs);
    for (block, frozen_layer) in model.layers.iter().zip(frozen.iter()) {
        layer_inputs.push(x.clone());
        x = dims_block_forward(&dims, block, x, mask_inner, Some(frozen_layer), device);
    }
    (layer_inputs, x)
}

/// A loss block's yield: the loss, its seed gradient, and the head
/// delta's own gradients.
pub type SeedGrads<AD> = (
    f32,
    FloatTensor<<AD as AutodiffBackend>::InnerBackend, 2>,
    HashMap<String, FloatTensor<<AD as AutodiffBackend>::InnerBackend, 2>>,
);

/// The loss block: one sequence's scalar loss, its seed gradient, and
/// the head delta's own gradients when a delta is in the block.
pub fn seed_from_hidden<AD, F>(
    model: &HybridModel<AD::InnerBackend>,
    head_delta: Option<&HeadDelta<AD>>,
    top_x: FloatTensor<AD::InnerBackend, 2>,
    build_loss: F,
) -> LibQuestResult<SeedGrads<AD>>
where
    AD: AutodiffBackend,
    F: FnOnce(
        FloatTensor<AD, 2>,
        &FloatTensor<AD, 2>,
        Option<&HeadDelta<AD>>,
    ) -> LibQuestResult<FloatTensor<AD, 1>>,
{
    let top_input = FloatTensor::<AD, 2>::from_inner(top_x).require_grad();
    let hidden = oracle::rms_norm(
        top_input.clone(),
        &FloatTensor::from_inner(model.final_norm.clone()),
        model.eps,
    );
    let head = FloatTensor::<AD, 2>::from_inner(model.lm_head_transposed.clone());
    let loss = build_loss(hidden, &head, head_delta)?;
    let loss_value: f32 = loss.clone().into_scalar().elem();
    let mut top_grads = loss.backward();
    let head_grads = head_delta_grads::<AD>(head_delta, &mut top_grads)?;
    let Some(grad_out) = top_input.grad_remove(&mut top_grads) else {
        snafu::whatever!("no gradient reached the loss block input");
    };
    Ok((loss_value, grad_out, head_grads))
}

/// Pull the head delta's gradients out of a loss block's backward.
fn head_delta_grads<AD: AutodiffBackend>(
    head_delta: Option<&HeadDelta<AD>>,
    grads: &mut <AD as AutodiffBackend>::Gradients,
) -> LibQuestResult<HashMap<String, FloatTensor<AD::InnerBackend, 2>>> {
    let mut head_grads = HashMap::new();
    if let Some(delta) = head_delta {
        let (Some(config_grad), Some(run_grad)) = (
            delta.config_col.grad_remove(grads),
            delta.run_cols.grad_remove(grads),
        ) else {
            snafu::whatever!("no gradient reached the head delta");
        };
        head_grads.insert(String::from(consts::HEAD_DELTA_CONFIG_PARAM), config_grad);
        head_grads.insert(String::from(consts::HEAD_DELTA_RUN_PARAM), run_grad);
    }
    Ok(head_grads)
}

/// A paired loss block's yield: the loss, one seed per sequence, and
/// the head delta's gradients.
pub type PairedSeedGrads<AD> = (
    f32,
    FloatTensor<<AD as AutodiffBackend>::InnerBackend, 2>,
    FloatTensor<<AD as AutodiffBackend>::InnerBackend, 2>,
    HashMap<String, FloatTensor<<AD as AutodiffBackend>::InnerBackend, 2>>,
);

/// The paired loss block: two sequences under one coupled loss.
pub fn seed_pair_from_hidden<AD, F>(
    model: &HybridModel<AD::InnerBackend>,
    head_delta: Option<&HeadDelta<AD>>,
    left_x: FloatTensor<AD::InnerBackend, 2>,
    right_x: FloatTensor<AD::InnerBackend, 2>,
    build_loss: F,
) -> LibQuestResult<PairedSeedGrads<AD>>
where
    AD: AutodiffBackend,
    F: FnOnce(
        FloatTensor<AD, 2>,
        FloatTensor<AD, 2>,
        &FloatTensor<AD, 2>,
        Option<&HeadDelta<AD>>,
    ) -> LibQuestResult<FloatTensor<AD, 1>>,
{
    let left_input = FloatTensor::<AD, 2>::from_inner(left_x).require_grad();
    let right_input = FloatTensor::<AD, 2>::from_inner(right_x).require_grad();
    let norm = FloatTensor::<AD, 1>::from_inner(model.final_norm.clone());
    let left_hidden = oracle::rms_norm(left_input.clone(), &norm, model.eps);
    let right_hidden = oracle::rms_norm(right_input.clone(), &norm, model.eps);
    let head = FloatTensor::<AD, 2>::from_inner(model.lm_head_transposed.clone());
    let loss = build_loss(left_hidden, right_hidden, &head, head_delta)?;
    let loss_value: f32 = loss.clone().into_scalar().elem();
    let mut top_grads = loss.backward();
    let head_grads = head_delta_grads::<AD>(head_delta, &mut top_grads)?;
    let (Some(left_grad), Some(right_grad)) = (
        left_input.grad_remove(&mut top_grads),
        right_input.grad_remove(&mut top_grads),
    ) else {
        snafu::whatever!("a paired loss left one sequence without a gradient");
    };
    Ok((loss_value, left_grad, right_grad, head_grads))
}

/// The recurrence-segment width the production training loops run under.
pub const RECURRENCE_SEGMENT: usize = 64;

/// Pass two: a per-layer VJP top down from a seed gradient.
pub fn chain_from_seed<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &ModelAdapters<AD>,
    mask_inner: &FloatTensor<AD::InnerBackend, 2>,
    layer_inputs: &[FloatTensor<AD::InnerBackend, 2>],
    seed: FloatTensor<AD::InnerBackend, 2>,
    segment: Option<usize>,
    device: &AD::Device,
) -> LibQuestResult<HashMap<String, FloatTensor<AD::InnerBackend, 2>>> {
    let dims = dims_of::<AD>(model);
    let mut grad_out = seed;
    let mut lora_grads: HashMap<String, FloatTensor<AD::InnerBackend, 2>> = HashMap::new();
    for (index, (block, layer_adapters)) in model
        .layers
        .iter()
        .zip(adapters.layers.iter())
        .enumerate()
        .rev()
    {
        let prefix = format!("model.layers.{index}");
        if let (HybridBlock::Gdn(gdn), Some(width)) = (block, segment) {
            let (grads, grad_in) = gdn_chain_segmented::<AD>(
                gdn,
                layer_adapters,
                &dims,
                &layer_inputs[index],
                &grad_out,
                width,
                &prefix,
                device,
            )?;
            for (name, grad) in grads {
                lora_grads.insert(name, grad);
            }
            grad_out = grad_in;
            continue;
        }
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
    Ok(lora_grads)
}

/// A segmented GDN gradient set plus the gradient handed to the layer below.
type SegmentedLayerGrads<AD> = (
    HashMap<String, FloatTensor<<AD as AutodiffBackend>::InnerBackend, 2>>,
    FloatTensor<<AD as AutodiffBackend>::InnerBackend, 2>,
);

/// One GDN layer's VJP with the recurrence checkpointed by segment.
#[allow(clippy::too_many_arguments)]
fn gdn_chain_segmented<AD: AutodiffBackend>(
    block: &oracle::GdnBlock<AD::InnerBackend>,
    layer_adapters: &LayerAdapters<AD>,
    dims: &HybridDims,
    block_input: &FloatTensor<AD::InnerBackend, 2>,
    grad_out: &FloatTensor<AD::InnerBackend, 2>,
    segment: usize,
    prefix: &str,
    device: &AD::Device,
) -> LibQuestResult<SegmentedLayerGrads<AD>> {
    let LayerAdapters::Gdn(gdn_adapters) = layer_adapters else {
        snafu::whatever!("a segmented GDN chain reached a non-GDN adapter set at {prefix}");
    };
    let segment = segment.max(1);
    let n = block_input.dims()[0];

    // Pass one, no grad: pre products, boundary states, the whole output.
    let frozen = adapters_inner::<AD>(layer_adapters);
    let LayerAdapters::Gdn(frozen_gdn) = &frozen else {
        unreachable!("frozen views keep their layer kind by construction")
    };
    let pre = block.gdn_pre(
        block_input.clone(),
        dims.gdn_heads,
        dims.gdn_key_dim,
        dims.gdn_value_dim,
        dims.eps,
        device,
        Some(frozen_gdn),
    );
    let mut state = FloatTensor::<AD::InnerBackend, 3>::zeros(
        [dims.gdn_heads, dims.gdn_key_dim, dims.gdn_value_dim],
        device,
    )
    .cast(burn::tensor::FloatDType::F32);
    let mut boundaries: Vec<FloatTensor<AD::InnerBackend, 3>> = Vec::new();
    let mut y_parts: Vec<FloatTensor<AD::InnerBackend, 3>> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let len = segment.min(n - start);
        boundaries.push(state.clone());
        let (y_seg, next) = oracle::gdn_recurrence::<AD::InnerBackend>(
            pre.q.clone().narrow(0, start, len),
            pre.k.clone().narrow(0, start, len),
            pre.v.clone().narrow(0, start, len),
            pre.decay.clone().narrow(0, start, len),
            pre.beta.clone().narrow(0, start, len),
            state,
        );
        y_parts.push(y_seg);
        state = next;
        start += len;
    }
    let y_full = burn::tensor::Tensor::cat(y_parts, 0);

    // The post half's backward: output-side adapters, y and gate grads.
    let block_ad = lift_gdn::<AD>(block);
    let x_post = FloatTensor::<AD, 2>::from_inner(block_input.clone()).require_grad();
    let y_leaf = FloatTensor::<AD, 3>::from_inner(y_full).require_grad();
    let gate_leaf = FloatTensor::<AD, 2>::from_inner(pre.gate_rows.clone()).require_grad();
    let out = block_ad.gdn_post(
        x_post.clone(),
        y_leaf.clone(),
        gate_leaf.clone(),
        dims.gdn_heads,
        dims.gdn_value_dim,
        dims.eps,
        Some(gdn_adapters),
    );
    let pseudo_loss = (out * FloatTensor::from_inner(grad_out.clone())).sum();
    let mut post_grads = pseudo_loss.backward();
    let mut collected: HashMap<String, FloatTensor<AD::InnerBackend, 2>> = HashMap::new();
    for (name, tensor) in layer_adapters.params(prefix) {
        if let Some(grad) = tensor.grad_remove(&mut post_grads) {
            collected.insert(name, grad);
        }
    }
    let Some(grad_y) = y_leaf.grad_remove(&mut post_grads) else {
        snafu::whatever!("no gradient reached the recurrence output at {prefix}");
    };
    let Some(grad_gate_rows) = gate_leaf.grad_remove(&mut post_grads) else {
        snafu::whatever!("no gradient reached the gate rows at {prefix}");
    };
    let Some(grad_x_post) = x_post.grad_remove(&mut post_grads) else {
        snafu::whatever!("no post-half input gradient at {prefix}");
    };

    // The recurrence backward, one segment at a time from the top: each
    // takes its output-grad slice plus the boundary adjoint above it, and
    // emits rule-input grads plus the adjoint for the segment below.
    let seg_count = boundaries.len();
    let mut adjoint: Option<FloatTensor<AD::InnerBackend, 3>> = None;
    let mut grad_q_parts: Vec<FloatTensor<AD::InnerBackend, 3>> = Vec::with_capacity(seg_count);
    let mut grad_k_parts: Vec<FloatTensor<AD::InnerBackend, 3>> = Vec::with_capacity(seg_count);
    let mut grad_v_parts: Vec<FloatTensor<AD::InnerBackend, 3>> = Vec::with_capacity(seg_count);
    let mut grad_d_parts: Vec<FloatTensor<AD::InnerBackend, 2>> = Vec::with_capacity(seg_count);
    let mut grad_b_parts: Vec<FloatTensor<AD::InnerBackend, 2>> = Vec::with_capacity(seg_count);
    for index in (0..seg_count).rev() {
        let start = index * segment;
        let len = segment.min(n - start);
        let state_in =
            FloatTensor::<AD, 3>::from_inner(boundaries[index].clone()).require_grad();
        let q_leaf =
            FloatTensor::<AD, 3>::from_inner(pre.q.clone().narrow(0, start, len)).require_grad();
        let k_leaf =
            FloatTensor::<AD, 3>::from_inner(pre.k.clone().narrow(0, start, len)).require_grad();
        let v_leaf =
            FloatTensor::<AD, 3>::from_inner(pre.v.clone().narrow(0, start, len)).require_grad();
        let d_leaf = FloatTensor::<AD, 2>::from_inner(pre.decay.clone().narrow(0, start, len))
            .require_grad();
        let b_leaf = FloatTensor::<AD, 2>::from_inner(pre.beta.clone().narrow(0, start, len))
            .require_grad();
        let (y_seg, state_out) = oracle::gdn_recurrence::<AD>(
            q_leaf.clone(),
            k_leaf.clone(),
            v_leaf.clone(),
            d_leaf.clone(),
            b_leaf.clone(),
            state_in.clone(),
        );
        let mut pseudo_loss =
            (y_seg * FloatTensor::from_inner(grad_y.clone().narrow(0, start, len))).sum();
        if let Some(above) = &adjoint {
            pseudo_loss = pseudo_loss + (state_out * FloatTensor::from_inner(above.clone())).sum();
        }
        let mut seg_grads = pseudo_loss.backward();
        let Some(grad_q) = q_leaf.grad_remove(&mut seg_grads) else {
            snafu::whatever!("no query gradient in segment {index} at {prefix}");
        };
        let Some(grad_k) = k_leaf.grad_remove(&mut seg_grads) else {
            snafu::whatever!("no key gradient in segment {index} at {prefix}");
        };
        let Some(grad_v) = v_leaf.grad_remove(&mut seg_grads) else {
            snafu::whatever!("no value gradient in segment {index} at {prefix}");
        };
        let Some(grad_d) = d_leaf.grad_remove(&mut seg_grads) else {
            snafu::whatever!("no decay gradient in segment {index} at {prefix}");
        };
        let Some(grad_b) = b_leaf.grad_remove(&mut seg_grads) else {
            snafu::whatever!("no beta gradient in segment {index} at {prefix}");
        };
        let Some(grad_state) = state_in.grad_remove(&mut seg_grads) else {
            snafu::whatever!("no state adjoint in segment {index} at {prefix}");
        };
        grad_q_parts.push(grad_q);
        grad_k_parts.push(grad_k);
        grad_v_parts.push(grad_v);
        grad_d_parts.push(grad_d);
        grad_b_parts.push(grad_b);
        adjoint = Some(grad_state);
    }
    grad_q_parts.reverse();
    grad_k_parts.reverse();
    grad_v_parts.reverse();
    grad_d_parts.reverse();
    grad_b_parts.reverse();
    let grad_q = burn::tensor::Tensor::cat(grad_q_parts, 0);
    let grad_k = burn::tensor::Tensor::cat(grad_k_parts, 0);
    let grad_v = burn::tensor::Tensor::cat(grad_v_parts, 0);
    let grad_d = burn::tensor::Tensor::cat(grad_d_parts, 0);
    let grad_b = burn::tensor::Tensor::cat(grad_b_parts, 0);

    // The pre half's backward: input-side adapters and the input gradient.
    let x_pre = FloatTensor::<AD, 2>::from_inner(block_input.clone()).require_grad();
    let pre_ad = block_ad.gdn_pre(
        x_pre.clone(),
        dims.gdn_heads,
        dims.gdn_key_dim,
        dims.gdn_value_dim,
        dims.eps,
        device,
        Some(gdn_adapters),
    );
    let pseudo_loss = (pre_ad.q * FloatTensor::from_inner(grad_q)).sum()
        + (pre_ad.k * FloatTensor::from_inner(grad_k)).sum()
        + (pre_ad.v * FloatTensor::from_inner(grad_v)).sum()
        + (pre_ad.decay * FloatTensor::from_inner(grad_d)).sum()
        + (pre_ad.beta * FloatTensor::from_inner(grad_b)).sum()
        + (pre_ad.gate_rows * FloatTensor::from_inner(grad_gate_rows)).sum();
    let mut pre_grads = pseudo_loss.backward();
    for (name, tensor) in layer_adapters.params(prefix) {
        if let Some(grad) = tensor.grad_remove(&mut pre_grads) {
            match collected.remove(&name) {
                Some(existing) => collected.insert(name, existing + grad),
                None => collected.insert(name, grad),
            };
        }
    }
    let Some(grad_x_pre) = x_pre.grad_remove(&mut pre_grads) else {
        snafu::whatever!("no pre-half input gradient at {prefix}");
    };
    for (name, _) in layer_adapters.params(prefix) {
        snafu::ensure_whatever!(
            collected.contains_key(&name),
            "no segmented gradient for {name}"
        );
    }
    Ok((collected, grad_x_post + grad_x_pre))
}

/// Add one gradient set into an accumulator, summing on name.
pub fn accumulate_grads<B: Backend>(
    into: &mut HashMap<String, FloatTensor<B, 2>>,
    from: HashMap<String, FloatTensor<B, 2>>,
) {
    for (name, grad) in from {
        match into.remove(&name) {
            Some(existing) => into.insert(name, existing + grad),
            None => into.insert(name, grad),
        };
    }
}

/// Scale an accumulated gradient set.
pub fn scale_grads<B: Backend>(
    grads: &mut HashMap<String, FloatTensor<B, 2>>,
    factor: f64,
) {
    let names: Vec<String> = grads.keys().cloned().collect();
    for name in names {
        if let Some(grad) = grads.remove(&name) {
            grads.insert(name, grad.mul_scalar(factor));
        }
    }
}

/// One continued-pretraining step: the plain LM objective.
#[allow(clippy::too_many_arguments)]
pub fn chain_step<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &ModelAdapters<AD>,
    mask_inner: &FloatTensor<AD::InnerBackend, 2>,
    inputs: &[u32],
    targets: &[u32],
    loss_chunk: usize,
    segment: Option<usize>,
    device: &AD::Device,
) -> LibQuestResult<StepOutcome<AD::InnerBackend>> {
    let (layer_inputs, top_x) =
        forward_cache::<AD>(model, adapters, mask_inner, inputs, device);
    let (loss, seed, head_grads) = seed_from_hidden::<AD, _>(
        model,
        Some(&adapters.head),
        top_x,
        |hidden, head, delta| {
            Ok(cross_entropy_chunked::<AD>(
                hidden, head, delta, targets, loss_chunk, device,
            ))
        },
    )?;
    let mut grads = chain_from_seed::<AD>(
        model,
        adapters,
        mask_inner,
        &layer_inputs,
        seed,
        segment,
        device,
    )?;
    accumulate_grads(&mut grads, head_grads);
    Ok(StepOutcome { loss, grads })
}

/// The full-graph reference step, at toy scale only.
fn full_graph_step<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &ModelAdapters<AD>,
    mask_inner: &FloatTensor<AD::InnerBackend, 2>,
    inputs: &[u32],
    targets: &[u32],
    loss_chunk: usize,
    device: &AD::Device,
) -> LibQuestResult<StepOutcome<AD::InnerBackend>> {
    let dims = dims_of::<AD>(model);
    let mask_ad = FloatTensor::<AD, 2>::from_inner(mask_inner.clone());
    let mut x = FloatTensor::<AD, 2>::from_inner(model.embed(inputs));
    for (block, layer_adapters) in model.layers.iter().zip(adapters.layers.iter()) {
        let block_ad = lift_block::<AD>(block);
        x = dims_block_forward(&dims, &block_ad, x, &mask_ad, Some(layer_adapters), device);
    }
    let hidden = oracle::rms_norm(
        x,
        &FloatTensor::from_inner(model.final_norm.clone()),
        model.eps,
    );
    let loss = cross_entropy_chunked::<AD>(
        hidden,
        &FloatTensor::from_inner(model.lm_head_transposed.clone()),
        Some(&adapters.head),
        targets,
        loss_chunk,
        device,
    );
    let loss_value: f32 = loss.clone().into_scalar().elem();
    let mut grads = loss.backward();
    let mut lora_grads = head_delta_grads::<AD>(Some(&adapters.head), &mut grads)?;
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

/// The continued-pretraining loop: chain steps under AdamW.
pub fn train_loop<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &mut ModelAdapters<AD>,
    chunks: &[Vec<u32>],
    options: &LoopOptions,
    device: &AD::Device,
    mut on_step: impl FnMut(&StepLog) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
) -> LibQuestResult<TrainReport> {
    snafu::ensure_whatever!(!chunks.is_empty(), "no training chunks");
    let seq_len = chunks[0].len() - 1;
    let mask_inner = oracle::causal_mask::<AD::InnerBackend>(seq_len, device);
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
            Some(RECURRENCE_SEGMENT),
            device,
        )?;
        if step == 0 {
            first_loss = outcome.loss;
            if std::env::var("QUEST_TRAIN_DEBUG_GRADS").is_ok() {
                debug_grad_health(&outcome.grads)?;
            }
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

        let learning_rate = warmup_rate(options.learning_rate, options.warmup_steps, step);
        optimizer_step::<AD>(
            &optimizer,
            &mut states,
            adapters,
            &mut outcome.grads,
            learning_rate,
        )?;
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
        on_step(&log).map_err(|source| LibQuestError::Whatever {
            message: String::from("the step callback failed"),
            source: Some(source),
        })?;
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

/// The shared learning-rate schedule: linear warmup, then constant.
pub fn warmup_rate(peak: f64, warmup_steps: usize, step: usize) -> f64 {
    peak * (((step + 1) as f64 / warmup_steps.max(1) as f64).min(1.0))
}

/// Apply one AdamW update per adapter tensor, consuming the gradients.
pub fn optimizer_step<AD: AutodiffBackend>(
    optimizer: &burn::optim::AdamW,
    states: &mut HashMap<String, OptState<AD::InnerBackend>>,
    adapters: &mut ModelAdapters<AD>,
    grads: &mut HashMap<String, FloatTensor<AD::InnerBackend, 2>>,
    learning_rate: f64,
) -> LibQuestResult<()> {
    for (name, tensor) in adapters.params_mut() {
        let Some(grad) = grads.remove(&name) else {
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
    Ok(())
}

/// Per-layer gradient health print, for localizing a NaN backward.
fn debug_grad_health<B: Backend>(
    grads: &HashMap<String, FloatTensor<B, 2>>,
) -> LibQuestResult<()> {
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


/// A 2-layer toy hybrid shape, small enough for cpu-f32 autodiff.
pub fn toy_config() -> HybridCheckpointConfig {
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

/// Matrix init at a bounded uniform scale.
fn toy_tensor(rng: &mut SplitMix64, shape: Vec<usize>) -> (Vec<usize>, Vec<f32>) {
    let count: usize = shape.iter().product();
    let values = (0..count)
        .map(|_| ((rng.next_u64() % 2000) as f32 / 1000.0 - 1.0) * 0.3)
        .collect();
    (shape, values)
}

/// Norm scales init near 1.0.
fn toy_norm(rng: &mut SplitMix64, width: usize) -> (Vec<usize>, Vec<f32>) {
    let values = (0..width)
        .map(|_| 1.0 + ((rng.next_u64() % 2000) as f32 / 1000.0 - 1.0) * 0.05)
        .collect();
    (vec![width], values)
}

pub fn toy_weights(config: &HybridCheckpointConfig) -> HybridWeights {
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

/// One gradient-set comparison: the worst per-parameter nmse.
pub struct GradCompare {
    pub max_nmse: f64,
    pub compared: usize,
    pub zero_matched: usize,
}

fn compare_grads<B: Backend>(
    chain: &HashMap<String, FloatTensor<B, 2>>,
    full: &HashMap<String, FloatTensor<B, 2>>,
) -> LibQuestResult<GradCompare> {
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

fn tensor_values<B: Backend>(tensor: &FloatTensor<B, 2>) -> LibQuestResult<Vec<f32>> {
    let data = tensor.clone().into_data().convert::<f32>();
    match data.to_vec::<f32>() {
        Ok(values) => Ok(values),
        Err(error) => snafu::whatever!("gradient extraction failed: {error:?}"),
    }
}

/// The gate verdict: the toy locks' readings, segmented legs included.
pub struct GateVerdict {
    pub adapter_off_exact: bool,
    pub init_grads: GradCompare,
    pub trained_grads: GradCompare,
    pub segmented_init: GradCompare,
    pub segmented_trained: GradCompare,
    pub first_loss: f32,
    pub final_loss: f32,
    pub descended: bool,
}

/// The chain-versus-full gradient bar.
pub const GATE_GRAD_NMSE_BAR: f64 = 1e-9;
/// The descent margin the 60-step lock must clear.
const GATE_DESCENT_MARGIN: f32 = 0.05;
/// The gate's segment width: four segments over the toy sequence.
const GATE_SEGMENT: usize = 4;

/// Run the three toy-config locks on cpu f32.
pub fn run_train_gate() -> LibQuestResult<GateVerdict> {
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
    let mask_inner = oracle::causal_mask::<CpuBack>(seq_len, &device);
    let dims = dims_of::<ToyAd>(&model);

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

    let targets = &chunks[0][1..];
    let chain = chain_step::<ToyAd>(
        &model, &adapters, &mask_inner, ids, targets, 8, None, &device,
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
    let segmented = chain_step::<ToyAd>(
        &model, &adapters, &mask_inner, ids, targets, 8, Some(GATE_SEGMENT), &device,
    )?;
    snafu::ensure_whatever!(
        (segmented.loss - full.loss).abs() <= 1e-6,
        "segmented and full-graph losses diverge ({} vs {})",
        segmented.loss,
        full.loss
    );
    let segmented_init = compare_grads(&segmented.grads, &full.grads)?;
    snafu::ensure_whatever!(
        segmented_init.max_nmse <= GATE_GRAD_NMSE_BAR,
        "the segmented chain misses the init gradient bar ({} vs {})",
        segmented_init.max_nmse,
        GATE_GRAD_NMSE_BAR
    );

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
        &model, &adapters, &mask_inner, ids, targets, 8, None, &device,
    )?;
    let full = full_graph_step::<ToyAd>(
        &model, &adapters, &mask_inner, ids, targets, 8, &device,
    )?;
    let trained_grads = compare_grads(&chain.grads, &full.grads)?;
    let segmented = chain_step::<ToyAd>(
        &model, &adapters, &mask_inner, ids, targets, 8, Some(GATE_SEGMENT), &device,
    )?;
    let segmented_trained = compare_grads(&segmented.grads, &full.grads)?;
    snafu::ensure_whatever!(
        segmented_trained.max_nmse <= GATE_GRAD_NMSE_BAR,
        "the segmented chain misses the trained gradient bar ({} vs {})",
        segmented_trained.max_nmse,
        GATE_GRAD_NMSE_BAR
    );

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
        segmented_init,
        segmented_trained,
        first_loss: report.first_loss,
        final_loss: report.final_loss,
        descended: report.final_loss < report.first_loss - GATE_DESCENT_MARGIN,
    })
}


/// The adapter-off forward: the frozen base, with no adapters.
fn forward_plain<B: Backend>(
    model: &HybridModel<B>,
    mask_inner: &FloatTensor<B, 2>,
    inputs: &[u32],
    device: &B::Device,
) -> FloatTensor<B, 2> {
    let dims = HybridDims {
        attn_heads: model.attn_heads,
        attn_head_dim: model.attn_head_dim,
        gdn_heads: model.gdn_heads,
        gdn_key_dim: model.gdn_key_dim,
        gdn_value_dim: model.gdn_value_dim,
        eps: model.eps,
    };
    let mut x = model.embed(inputs);
    for block in model.layers.iter() {
        x = dims_block_forward(&dims, block, x, mask_inner, None, device);
    }
    x
}

/// A sequence's masked log-probability under the frozen base.
pub fn reference_logprob<B: Backend>(
    model: &HybridModel<B>,
    mask_inner: &FloatTensor<B, 2>,
    inputs: &[u32],
    targets: &[u32],
    mask: &[u8],
    chunk: usize,
    device: &B::Device,
) -> LibQuestResult<f32> {
    let top = forward_plain::<B>(model, mask_inner, inputs, device);
    let hidden = oracle::rms_norm(top, &model.final_norm, model.eps);
    let n = targets.len();
    let chunk = chunk.max(1);
    let mut total = 0f64;
    let mut start = 0usize;
    while start < n {
        let end = (start + chunk).min(n);
        let rows = end - start;
        let logits = hidden
            .clone()
            .narrow(0, start, rows)
            .matmul(model.lm_head_transposed.clone());
        let log_probs = burn::tensor::activation::log_softmax(logits, 1);
        let indices: Vec<i64> = targets[start..end].iter().map(|t| *t as i64).collect();
        let index_tensor = burn::tensor::Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(indices, [rows, 1]),
            device,
        );
        let picked = log_probs.gather(1, index_tensor);
        let values = match picked.into_data().convert::<f32>().to_vec::<f32>() {
            Ok(values) => values,
            Err(error) => snafu::whatever!("reference logprob extraction failed: {error:?}"),
        };
        for (offset, value) in values.iter().enumerate() {
            if mask[start + offset] != 0 {
                total += *value as f64;
            }
        }
        start = end;
    }
    Ok(total as f32)
}

/// Split a packed row into inputs, targets, and the target mask.
pub fn split_row<'a>(ids: &'a [u32], mask: &'a [u8]) -> (&'a [u32], &'a [u32], &'a [u8]) {
    let width = ids.len();
    (&ids[..width - 1], &ids[1..], &mask[1..])
}

/// The supervised loop's input: rows, masks, and accumulation.
pub struct SupervisedBatch<'a> {
    pub rows: &'a [Vec<u32>],
    pub masks: &'a [Vec<u8>],
    pub accumulate: usize,
}

/// The supervised loss normalization the loop applies per example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SftObjective {
    /// The standing per-example mean over the row's own count.
    ExampleMean,
    /// Token-uniform: the sum over the pack-mean supervised count.
    TokenUniform,
}

/// One row's last-visit losses at its supervised full-row positions.
pub struct RowTokenLosses {
    pub row: usize,
    pub positions: Vec<usize>,
    pub losses: Vec<f32>,
}

/// The supervised loop's report: the shared totals plus the token dump.
pub struct SftReport {
    pub report: TrainReport,
    pub token_losses: Vec<RowTokenLosses>,
}

/// The supervised loop: masked cross-entropy over example rows.
pub fn sft_loop<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &mut ModelAdapters<AD>,
    batch: &SupervisedBatch<'_>,
    objective: SftObjective,
    options: &LoopOptions,
    device: &AD::Device,
    mut on_step: impl FnMut(&StepLog) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
) -> LibQuestResult<SftReport> {
    let (rows, masks, accumulate) = (batch.rows, batch.masks, batch.accumulate);
    snafu::ensure_whatever!(!rows.is_empty(), "no supervised rows");
    let seq_len = rows[0].len() - 1;
    // The token-uniform denominator is a RUN constant over the whole
    // pack, visited or not, so a partial run prices tokens identically.
    let mean_supervised = {
        let total: usize = masks
            .iter()
            .map(|mask| objective::supervised_count(&mask[1..]))
            .sum();
        total as f64 / rows.len() as f64
    };
    let mask_inner = oracle::causal_mask::<AD::InnerBackend>(seq_len, device);
    let optimizer = burn::optim::AdamWConfig::new().with_weight_decay(0.0).build();
    let mut states: HashMap<String, OptState<AD::InnerBackend>> = HashMap::new();
    let accumulate = accumulate.max(1);

    let mut first_loss = 0f32;
    let mut last_loss = 0f32;
    let mut trained_tokens = 0usize;
    // Per-visit overwrite, so each slot ends as its row's LAST visit.
    let mut final_token_losses: Vec<Option<RowTokenLosses>> =
        (0..rows.len()).map(|_| None).collect();
    // Per-pass seeded reshuffle: a fixed order replays the same neighbor
    // sequence every pass, compounding order effects at batch 1.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let mut order_epoch = usize::MAX;
    let started = std::time::Instant::now();
    for step in 0..options.steps {
        let mut accumulated: HashMap<String, FloatTensor<AD::InnerBackend, 2>> = HashMap::new();
        let mut batch_loss = 0f32;
        for slot in 0..accumulate {
            let flat = step * accumulate + slot;
            let epoch = flat / rows.len();
            if epoch != order_epoch {
                let mut rng = SplitMix64::new(SFT_SHUFFLE_SEED ^ epoch as u64);
                for high in (1..order.len()).rev() {
                    let pick = rng.next_below(high + 1);
                    order.swap(high, pick);
                }
                order_epoch = epoch;
            }
            let index = order[flat % rows.len()];
            let (ids, mask) = (&rows[index], &masks[index]);
            snafu::ensure_whatever!(
                ids.len() == seq_len + 1 && mask.len() == seq_len + 1,
                "row {index} is {} wide against seq_len {seq_len}",
                ids.len()
            );
            let (inputs, targets, target_mask) = split_row(ids, mask);
            let supervised = objective::supervised_count(target_mask);
            snafu::ensure_whatever!(
                supervised > 0,
                "row {index} supervises no position - packing should have dropped it"
            );
            let (layer_inputs, top_x) =
                forward_cache::<AD>(model, adapters, &mask_inner, inputs, device);
            let mut visit_capture: Vec<objective::TokenLoss> = Vec::new();
            let (loss, seed, head_grads) = seed_from_hidden::<AD, _>(
                model,
                Some(&adapters.head),
                top_x,
                |hidden, head, delta| match objective {
                    SftObjective::ExampleMean => objective::masked_cross_entropy::<AD>(
                        hidden,
                        head,
                        delta,
                        targets,
                        target_mask,
                        options.loss_chunk,
                        device,
                        Some(&mut visit_capture),
                    ),
                    SftObjective::TokenUniform => objective::masked_cross_entropy_fixed::<AD>(
                        hidden,
                        head,
                        delta,
                        targets,
                        target_mask,
                        options.loss_chunk,
                        device,
                        mean_supervised,
                        Some(&mut visit_capture),
                    ),
                },
            )?;
            // Target index + 1 is the token's own full-row position.
            final_token_losses[index] = Some(RowTokenLosses {
                row: index,
                positions: visit_capture
                    .iter()
                    .map(|(target_index, _)| target_index + 1)
                    .collect(),
                losses: visit_capture.iter().map(|(_, loss)| *loss).collect(),
            });
            let grads = chain_from_seed::<AD>(
                model,
                adapters,
                &mask_inner,
                &layer_inputs,
                seed,
                Some(RECURRENCE_SEGMENT),
                device,
            )?;
            accumulate_grads(&mut accumulated, grads);
            accumulate_grads(&mut accumulated, head_grads);
            batch_loss += loss;
            trained_tokens += supervised;
        }
        scale_grads(&mut accumulated, 1.0 / accumulate as f64);
        let batch_loss = batch_loss / accumulate as f32;
        if step == 0 {
            first_loss = batch_loss;
        }
        last_loss = batch_loss;

        let learning_rate = warmup_rate(options.learning_rate, options.warmup_steps, step);
        optimizer_step::<AD>(
            &optimizer,
            &mut states,
            adapters,
            &mut accumulated,
            learning_rate,
        )?;

        let elapsed = started.elapsed().as_secs_f64();
        let log = StepLog {
            step: step + 1,
            loss: last_loss,
            learning_rate,
            tokens_per_second: trained_tokens as f64 / elapsed,
            elapsed_seconds: elapsed,
        };
        on_step(&log).map_err(|source| LibQuestError::Whatever {
            message: String::from("the step callback failed"),
            source: Some(source),
        })?;
        if (step + 1) % options.log_every.max(1) == 0 || step + 1 == options.steps {
            eprintln!(
                "bquest train sft: step {} | loss {:.4} | lr {:.2e} | {:.0} supervised tok/s",
                log.step, log.loss, log.learning_rate, log.tokens_per_second
            );
        }
    }
    Ok(SftReport {
        report: TrainReport {
            first_loss,
            final_loss: last_loss,
            steps: options.steps,
            trained_tokens,
            seconds: started.elapsed().as_secs_f64(),
        },
        // A short run visits only some rows; only visited rows dump.
        token_losses: final_token_losses.into_iter().flatten().collect(),
    })
}

/// One preference pair, encoded, with its frozen-base log-probs.
pub struct EncodedPair {
    pub chosen: (Vec<u32>, Vec<u8>),
    pub rejected: (Vec<u32>, Vec<u8>),
    pub chosen_reference: f32,
    pub rejected_reference: f32,
}

/// The preference loop: one coupled backward per pair.
pub fn dpo_loop<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &mut ModelAdapters<AD>,
    pairs: &[EncodedPair],
    beta: f64,
    options: &LoopOptions,
    device: &AD::Device,
    mut on_step: impl FnMut(&StepLog) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
) -> LibQuestResult<TrainReport> {
    snafu::ensure_whatever!(!pairs.is_empty(), "no preference pairs");
    let optimizer = burn::optim::AdamWConfig::new().with_weight_decay(0.0).build();
    let mut states: HashMap<String, OptState<AD::InnerBackend>> = HashMap::new();
    let mut first_loss = 0f32;
    let mut last_loss = 0f32;
    let mut trained_tokens = 0usize;
    let started = std::time::Instant::now();

    for step in 0..options.steps {
        let pair = &pairs[step % pairs.len()];
        let seq_len = pair.chosen.0.len() - 1;
        snafu::ensure_whatever!(
            pair.rejected.0.len() == pair.chosen.0.len(),
            "a preference pair's candidates are padded to different widths"
        );
        let mask_inner = oracle::causal_mask::<AD::InnerBackend>(seq_len, device);
        let (chosen_in, chosen_targets, chosen_mask) =
            split_row(&pair.chosen.0, &pair.chosen.1);
        let (rejected_in, rejected_targets, rejected_mask) =
            split_row(&pair.rejected.0, &pair.rejected.1);

        let (chosen_layers, chosen_top) =
            forward_cache::<AD>(model, adapters, &mask_inner, chosen_in, device);
        let (rejected_layers, rejected_top) =
            forward_cache::<AD>(model, adapters, &mask_inner, rejected_in, device);

        let loss_chunk = options.loss_chunk;
        let (loss, chosen_seed, rejected_seed, head_grads) = seed_pair_from_hidden::<AD, _>(
            model,
            Some(&adapters.head),
            chosen_top,
            rejected_top,
            |chosen_hidden, rejected_hidden, head, delta| {
                let chosen_logprob = objective::sequence_logprob::<AD>(
                    chosen_hidden,
                    head,
                    delta,
                    chosen_targets,
                    chosen_mask,
                    loss_chunk,
                    device,
                )?;
                let rejected_logprob = objective::sequence_logprob::<AD>(
                    rejected_hidden,
                    head,
                    delta,
                    rejected_targets,
                    rejected_mask,
                    loss_chunk,
                    device,
                )?;
                Ok(objective::dpo_loss::<AD>(
                    chosen_logprob,
                    rejected_logprob,
                    pair.chosen_reference,
                    pair.rejected_reference,
                    beta,
                ))
            },
        )?;

        let mut accumulated = chain_from_seed::<AD>(
            model,
            adapters,
            &mask_inner,
            &chosen_layers,
            chosen_seed,
            Some(RECURRENCE_SEGMENT),
            device,
        )?;
        let rejected_grads = chain_from_seed::<AD>(
            model,
            adapters,
            &mask_inner,
            &rejected_layers,
            rejected_seed,
            Some(RECURRENCE_SEGMENT),
            device,
        )?;
        accumulate_grads(&mut accumulated, rejected_grads);
        accumulate_grads(&mut accumulated, head_grads);

        if step == 0 {
            first_loss = loss;
        }
        last_loss = loss;
        trained_tokens += objective::supervised_count(chosen_mask)
            + objective::supervised_count(rejected_mask);

        let learning_rate = warmup_rate(options.learning_rate, options.warmup_steps, step);
        optimizer_step::<AD>(
            &optimizer,
            &mut states,
            adapters,
            &mut accumulated,
            learning_rate,
        )?;

        let elapsed = started.elapsed().as_secs_f64();
        let log = StepLog {
            step: step + 1,
            loss: last_loss,
            learning_rate,
            tokens_per_second: trained_tokens as f64 / elapsed,
            elapsed_seconds: elapsed,
        };
        on_step(&log).map_err(|source| LibQuestError::Whatever {
            message: String::from("the step callback failed"),
            source: Some(source),
        })?;
        if (step + 1) % options.log_every.max(1) == 0 || step + 1 == options.steps {
            eprintln!(
                "bquest train dpo: step {} | loss {:.4} | lr {:.2e}",
                log.step, log.loss, log.learning_rate
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

/// One scored rollout: the window, its mask, and the reward.
pub struct ScoredRollout {
    pub ids: Vec<u32>,
    pub mask: Vec<u8>,
    pub reward: f32,
}

/// One prompt's group of rollouts.
pub struct RolloutGroup {
    pub rollouts: Vec<ScoredRollout>,
}

/// The reinforcement loop: one optimizer step per scoring group.
pub fn rlvr_loop<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    adapters: &mut ModelAdapters<AD>,
    groups: &[RolloutGroup],
    options: &LoopOptions,
    device: &AD::Device,
    mut on_step: impl FnMut(&StepLog) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
) -> LibQuestResult<TrainReport> {
    snafu::ensure_whatever!(!groups.is_empty(), "no rollout groups");
    let optimizer = burn::optim::AdamWConfig::new().with_weight_decay(0.0).build();
    let mut states: HashMap<String, OptState<AD::InnerBackend>> = HashMap::new();
    let mut first_loss = 0f32;
    let mut last_loss = 0f32;
    let mut trained_tokens = 0usize;
    let mut stepped = 0usize;
    let started = std::time::Instant::now();

    for step in 0..options.steps {
        let group = &groups[step % groups.len()];
        let rewards: Vec<f32> = group.rollouts.iter().map(|r| r.reward).collect();
        let advantages = objective::group_advantages(&rewards);
        if advantages.iter().all(|a| *a == 0.0) {
            eprintln!(
                "bquest train rlvr: group {} scored uniformly ({:?}) - no signal, skipped",
                step % groups.len(),
                rewards.first()
            );
            continue;
        }

        let mut accumulated: HashMap<String, FloatTensor<AD::InnerBackend, 2>> = HashMap::new();
        let mut group_loss = 0f32;
        for (rollout, advantage) in group.rollouts.iter().zip(&advantages) {
            if *advantage == 0.0 {
                continue;
            }
            let seq_len = rollout.ids.len() - 1;
            let mask_inner = oracle::causal_mask::<AD::InnerBackend>(seq_len, device);
            let (inputs, targets, target_mask) = split_row(&rollout.ids, &rollout.mask);
            if objective::supervised_count(target_mask) == 0 {
                continue;
            }
            let (layer_inputs, top_x) =
                forward_cache::<AD>(model, adapters, &mask_inner, inputs, device);
            let advantage = *advantage;
            let (loss, seed, head_grads) = seed_from_hidden::<AD, _>(
                model,
                Some(&adapters.head),
                top_x,
                |hidden, head, delta| {
                    let logprob = objective::sequence_logprob::<AD>(
                        hidden,
                        head,
                        delta,
                        targets,
                        target_mask,
                        options.loss_chunk,
                        device,
                    )?;
                    Ok(objective::policy_gradient_loss::<AD>(logprob, advantage))
                },
            )?;
            let grads = chain_from_seed::<AD>(
                model,
                adapters,
                &mask_inner,
                &layer_inputs,
                seed,
                Some(RECURRENCE_SEGMENT),
                device,
            )?;
            accumulate_grads(&mut accumulated, grads);
            accumulate_grads(&mut accumulated, head_grads);
            group_loss += loss;
            trained_tokens += objective::supervised_count(target_mask);
        }
        if accumulated.is_empty() {
            continue;
        }
        scale_grads(&mut accumulated, 1.0 / group.rollouts.len() as f64);
        let group_loss = group_loss / group.rollouts.len() as f32;
        if stepped == 0 {
            first_loss = group_loss;
        }
        last_loss = group_loss;
        stepped += 1;

        let learning_rate = warmup_rate(options.learning_rate, options.warmup_steps, step);
        optimizer_step::<AD>(
            &optimizer,
            &mut states,
            adapters,
            &mut accumulated,
            learning_rate,
        )?;

        let elapsed = started.elapsed().as_secs_f64();
        let log = StepLog {
            step: step + 1,
            loss: last_loss,
            learning_rate,
            tokens_per_second: trained_tokens as f64 / elapsed,
            elapsed_seconds: elapsed,
        };
        on_step(&log).map_err(|source| LibQuestError::Whatever {
            message: String::from("the step callback failed"),
            source: Some(source),
        })?;
        if (step + 1) % options.log_every.max(1) == 0 || step + 1 == options.steps {
            eprintln!(
                "bquest train rlvr: step {} | mean reward {:.3} | objective {:.4} | lr {:.2e}",
                log.step,
                rewards.iter().sum::<f32>() / rewards.len() as f32,
                log.loss,
                log.learning_rate
            );
        }
    }
    Ok(TrainReport {
        first_loss,
        final_loss: last_loss,
        steps: stepped,
        trained_tokens,
        seconds: started.elapsed().as_secs_f64(),
    })
}
