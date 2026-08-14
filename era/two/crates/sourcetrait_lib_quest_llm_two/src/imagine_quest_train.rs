//! From-scratch full-parameter training for the ImagineQuest organism.
use crate::*;

use burn::tensor::{
    Int,
    backend::AutodiffBackend,
    backend::Backend,
};

type FloatTensor<B, const D: usize> = burn::tensor::Tensor<B, D>;

/// AdamW moments (0.9, 0.95): the from-scratch pretraining convention.
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.95;
const ADAM_EPS: f64 = 1e-8;
/// The organism loop's per-epoch chunk reshuffle seed (deterministic).
const CHUNK_SHUFFLE_SEED: u64 = 0x5EED_0002;

/// The knobs the organism loop takes beyond the shared LoopOptions.
pub struct OrganismLoopOptions {
    pub steps: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub loss_chunk: usize,
    pub log_every: usize,
    /// Microbatches (chunks) summed into one optimizer step.
    pub accumulate: usize,
    /// GDN recurrence checkpoint width for the per-layer walk.
    pub recurrence_segment: usize,
    /// Steps between mid-run checkpoints; 0 saves the final one only.
    pub checkpoint_every: usize,
}

impl Default for OrganismLoopOptions {
    fn default() -> Self {
        Self {
            steps: 1,
            learning_rate: 3e-4,
            warmup_steps: 100,
            loss_chunk: 128,
            log_every: 10,
            accumulate: 1,
            recurrence_segment: train::RECURRENCE_SEGMENT,
            checkpoint_every: 0,
        }
    }
}

/// A named core parameter's tensor, at the oracle's own field rank.
pub(crate) enum ParamTensor<B: Backend> {
    Vector(FloatTensor<B, 1>),
    Matrix(FloatTensor<B, 2>),
}

/// A mutable view of one core parameter during the model walk.
enum ParamField<'a, B: Backend> {
    Vector(&'a mut FloatTensor<B, 1>),
    Matrix(&'a mut FloatTensor<B, 2>),
}

/// One core parameter's f32 master and Adam moments.
struct MasterState<B: Backend> {
    master: ParamTensor<B>,
    moment: ParamTensor<B>,
    velocity: ParamTensor<B>,
}

/// One touched embedding row's Adam moments, host-side.
struct EmbeddingRowMoments {
    moment: Vec<f32>,
    velocity: Vec<f32>,
}

/// One step's gradients: the core map plus touched embedding rows.
pub struct OrganismGrads<B: Backend> {
    core: HashMap<String, ParamTensor<B>>,
    /// Touched row id to its summed f32 gradient row (hidden wide).
    rows: HashMap<u32, Vec<f32>>,
}

/// The full-parameter organism trainer over the burn oracle stack.
pub struct OrganismTrainer<AD: AutodiffBackend> {
    pub model: HybridModel<AD::InnerBackend>,
    config: HybridCheckpointConfig,
    /// The source config.json bytes, written back verbatim on save.
    config_json: Vec<u8>,
    masters: HashMap<String, MasterState<AD::InnerBackend>>,
    /// Adam state exists only for rows a step touched (TheUser lever).
    embedding_moments: HashMap<u32, EmbeddingRowMoments>,
    step: usize,
    /// The backend's default float, probed once: masters cast back to
    /// it after every step so forwards stay on the compute dtype.
    compute_dtype: burn::tensor::FloatDType,
    device: <AD::InnerBackend as burn::tensor::backend::BackendTypes>::Device,
}

/// Walk every core parameter (blocks plus final norm; embedding and
/// tied head ride the row-sparse path instead), checkpoint-named.
fn for_each_core_param<B: Backend>(
    model: &mut HybridModel<B>,
    visit: &mut dyn FnMut(String, ParamField<'_, B>) -> LibQuestResult<()>,
) -> LibQuestResult<()> {
    for (index, block) in model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{index}");
        match block {
            HybridBlock::Gdn(block) => {
                visit(
                    format!("{prefix}.input_layernorm.weight"),
                    ParamField::Vector(&mut block.input_norm),
                )?;
                visit(
                    format!("{prefix}.post_attention_layernorm.weight"),
                    ParamField::Vector(&mut block.post_attention_norm),
                )?;
                for (name, tensor) in [
                    ("q_proj", &mut block.q_transposed),
                    ("k_proj", &mut block.k_transposed),
                    ("v_proj", &mut block.v_transposed),
                    ("g_proj", &mut block.g_transposed),
                    ("o_proj", &mut block.o_transposed),
                    ("a_proj", &mut block.a_transposed),
                    ("b_proj", &mut block.b_transposed),
                ] {
                    visit(
                        format!("{prefix}.linear_attn.{name}.weight"),
                        ParamField::Matrix(tensor),
                    )?;
                }
                for (name, taps) in [
                    ("q_conv1d", &mut block.q_conv_taps),
                    ("k_conv1d", &mut block.k_conv_taps),
                    ("v_conv1d", &mut block.v_conv_taps),
                ] {
                    for (tap, row) in taps.iter_mut().enumerate() {
                        visit(
                            format!("{prefix}.linear_attn.{name}.weight.tap{tap}"),
                            ParamField::Matrix(row),
                        )?;
                    }
                }
                visit(
                    format!("{prefix}.linear_attn.A_log"),
                    ParamField::Matrix(&mut block.a_log_row),
                )?;
                visit(
                    format!("{prefix}.linear_attn.dt_bias"),
                    ParamField::Matrix(&mut block.dt_bias_row),
                )?;
                visit(
                    format!("{prefix}.linear_attn.o_norm.weight"),
                    ParamField::Vector(&mut block.o_norm),
                )?;
                for (name, tensor) in [
                    ("gate_proj", &mut block.gate_transposed),
                    ("up_proj", &mut block.up_transposed),
                    ("down_proj", &mut block.down_transposed),
                ] {
                    visit(
                        format!("{prefix}.mlp.{name}.weight"),
                        ParamField::Matrix(tensor),
                    )?;
                }
            }
            HybridBlock::Attn(block) => {
                for (name, tensor) in [
                    ("q_proj", &mut block.q_transposed),
                    ("k_proj", &mut block.k_transposed),
                    ("v_proj", &mut block.v_transposed),
                    ("o_proj", &mut block.o_transposed),
                ] {
                    visit(
                        format!("{prefix}.self_attn.{name}.weight"),
                        ParamField::Matrix(tensor),
                    )?;
                }
                visit(
                    format!("{prefix}.self_attn.q_norm.weight"),
                    ParamField::Vector(&mut block.q_norm),
                )?;
                visit(
                    format!("{prefix}.self_attn.k_norm.weight"),
                    ParamField::Vector(&mut block.k_norm),
                )?;
                visit(
                    format!("{prefix}.post_attention_layernorm.weight"),
                    ParamField::Vector(&mut block.post_attention_norm),
                )?;
                visit(
                    format!("{prefix}.post_feedforward_layernorm.weight"),
                    ParamField::Vector(&mut block.post_feedforward_norm),
                )?;
                for (name, tensor) in [
                    ("gate_proj", &mut block.gate_transposed),
                    ("up_proj", &mut block.up_transposed),
                    ("down_proj", &mut block.down_transposed),
                ] {
                    visit(
                        format!("{prefix}.mlp.{name}.weight"),
                        ParamField::Matrix(tensor),
                    )?;
                }
            }
        }
    }
    visit(
        String::from("model.norm.weight"),
        ParamField::Vector(&mut model.final_norm),
    )?;
    Ok(())
}

/// A tracked or constant lift of one weight into the autodiff backend.
fn lift1<AD: AutodiffBackend>(
    tensor: &FloatTensor<AD::InnerBackend, 1>,
    leaf: bool,
) -> FloatTensor<AD, 1> {
    let lifted = FloatTensor::<AD, 1>::from_inner(tensor.clone());
    if leaf { lifted.require_grad() } else { lifted }
}

fn lift2<AD: AutodiffBackend>(
    tensor: &FloatTensor<AD::InnerBackend, 2>,
    leaf: bool,
) -> FloatTensor<AD, 2> {
    let lifted = FloatTensor::<AD, 2>::from_inner(tensor.clone());
    if leaf { lifted.require_grad() } else { lifted }
}

/// A lifted layer's named leaves, kept for grad_remove after backward.
enum LeafTensor<AD: AutodiffBackend> {
    Vector(FloatTensor<AD, 1>),
    Matrix(FloatTensor<AD, 2>),
}

type NamedLeaves<AD> = Vec<(String, LeafTensor<AD>)>;

fn push_leaf1<AD: AutodiffBackend>(
    leaves: &mut NamedLeaves<AD>,
    name: String,
    tensor: &FloatTensor<AD::InnerBackend, 1>,
) -> FloatTensor<AD, 1> {
    let lifted = lift1::<AD>(tensor, true);
    leaves.push((name, LeafTensor::Vector(lifted.clone())));
    lifted
}

fn push_leaf2<AD: AutodiffBackend>(
    leaves: &mut NamedLeaves<AD>,
    name: String,
    tensor: &FloatTensor<AD::InnerBackend, 2>,
) -> FloatTensor<AD, 2> {
    let lifted = lift2::<AD>(tensor, true);
    leaves.push((name, LeafTensor::Matrix(lifted.clone())));
    lifted
}

/// Which of a GDN block's weights lift as leaves for one backward.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GdnLeafSet {
    /// gdn_pre's weights: norms-in, projections, convs, gate scalars.
    Pre,
    /// gdn_post's weights: output norm and mix, MLP half.
    Post,
    /// Everything: the unsegmented (test-reference) form.
    #[cfg_attr(not(test), allow(dead_code))]
    All,
}

impl GdnLeafSet {
    fn pre(self) -> bool {
        matches!(self, Self::Pre | Self::All)
    }

    fn post(self) -> bool {
        matches!(self, Self::Post | Self::All)
    }
}

/// Lift a GDN block with the chosen weight set tracked as leaves.
fn lift_gdn_trainable<AD: AutodiffBackend>(
    block: &GdnBlock<AD::InnerBackend>,
    set: GdnLeafSet,
    prefix: &str,
    leaves: &mut NamedLeaves<AD>,
) -> GdnBlock<AD> {
    let pre1 = |leaves: &mut NamedLeaves<AD>, name: &str, tensor| {
        if set.pre() {
            push_leaf1::<AD>(leaves, format!("{prefix}.{name}"), tensor)
        } else {
            lift1::<AD>(tensor, false)
        }
    };
    let pre2 = |leaves: &mut NamedLeaves<AD>, name: &str, tensor| {
        if set.pre() {
            push_leaf2::<AD>(leaves, format!("{prefix}.{name}"), tensor)
        } else {
            lift2::<AD>(tensor, false)
        }
    };
    let post1 = |leaves: &mut NamedLeaves<AD>, name: &str, tensor| {
        if set.post() {
            push_leaf1::<AD>(leaves, format!("{prefix}.{name}"), tensor)
        } else {
            lift1::<AD>(tensor, false)
        }
    };
    let post2 = |leaves: &mut NamedLeaves<AD>, name: &str, tensor| {
        if set.post() {
            push_leaf2::<AD>(leaves, format!("{prefix}.{name}"), tensor)
        } else {
            lift2::<AD>(tensor, false)
        }
    };
    let taps = |leaves: &mut NamedLeaves<AD>, name: &str, taps: &Vec<_>| {
        taps.iter()
            .enumerate()
            .map(|(tap, row)| {
                if set.pre() {
                    push_leaf2::<AD>(leaves, format!("{prefix}.{name}.tap{tap}"), row)
                } else {
                    lift2::<AD>(row, false)
                }
            })
            .collect()
    };
    GdnBlock {
        input_norm: pre1(leaves, "input_layernorm.weight", &block.input_norm),
        post_attention_norm: post1(
            leaves,
            "post_attention_layernorm.weight",
            &block.post_attention_norm,
        ),
        q_transposed: pre2(leaves, "linear_attn.q_proj.weight", &block.q_transposed),
        k_transposed: pre2(leaves, "linear_attn.k_proj.weight", &block.k_transposed),
        v_transposed: pre2(leaves, "linear_attn.v_proj.weight", &block.v_transposed),
        g_transposed: pre2(leaves, "linear_attn.g_proj.weight", &block.g_transposed),
        o_transposed: post2(leaves, "linear_attn.o_proj.weight", &block.o_transposed),
        a_transposed: pre2(leaves, "linear_attn.a_proj.weight", &block.a_transposed),
        b_transposed: pre2(leaves, "linear_attn.b_proj.weight", &block.b_transposed),
        q_conv_taps: taps(leaves, "linear_attn.q_conv1d.weight", &block.q_conv_taps),
        k_conv_taps: taps(leaves, "linear_attn.k_conv1d.weight", &block.k_conv_taps),
        v_conv_taps: taps(leaves, "linear_attn.v_conv1d.weight", &block.v_conv_taps),
        a_log_row: pre2(leaves, "linear_attn.A_log", &block.a_log_row),
        dt_bias_row: pre2(leaves, "linear_attn.dt_bias", &block.dt_bias_row),
        o_norm: post1(leaves, "linear_attn.o_norm.weight", &block.o_norm),
        gate_transposed: post2(leaves, "mlp.gate_proj.weight", &block.gate_transposed),
        up_transposed: post2(leaves, "mlp.up_proj.weight", &block.up_transposed),
        down_transposed: post2(leaves, "mlp.down_proj.weight", &block.down_transposed),
    }
}

/// Lift an attention block with every weight tracked as a leaf.
fn lift_attn_trainable<AD: AutodiffBackend>(
    block: &AttnBlock<AD::InnerBackend>,
    prefix: &str,
    leaves: &mut NamedLeaves<AD>,
) -> AttnBlock<AD> {
    AttnBlock {
        q_transposed: push_leaf2::<AD>(
            leaves,
            format!("{prefix}.self_attn.q_proj.weight"),
            &block.q_transposed,
        ),
        k_transposed: push_leaf2::<AD>(
            leaves,
            format!("{prefix}.self_attn.k_proj.weight"),
            &block.k_transposed,
        ),
        v_transposed: push_leaf2::<AD>(
            leaves,
            format!("{prefix}.self_attn.v_proj.weight"),
            &block.v_transposed,
        ),
        o_transposed: push_leaf2::<AD>(
            leaves,
            format!("{prefix}.self_attn.o_proj.weight"),
            &block.o_transposed,
        ),
        q_norm: push_leaf1::<AD>(
            leaves,
            format!("{prefix}.self_attn.q_norm.weight"),
            &block.q_norm,
        ),
        k_norm: push_leaf1::<AD>(
            leaves,
            format!("{prefix}.self_attn.k_norm.weight"),
            &block.k_norm,
        ),
        post_attention_norm: push_leaf1::<AD>(
            leaves,
            format!("{prefix}.post_attention_layernorm.weight"),
            &block.post_attention_norm,
        ),
        post_feedforward_norm: push_leaf1::<AD>(
            leaves,
            format!("{prefix}.post_feedforward_layernorm.weight"),
            &block.post_feedforward_norm,
        ),
        gate_transposed: push_leaf2::<AD>(
            leaves,
            format!("{prefix}.mlp.gate_proj.weight"),
            &block.gate_transposed,
        ),
        up_transposed: push_leaf2::<AD>(
            leaves,
            format!("{prefix}.mlp.up_proj.weight"),
            &block.up_transposed,
        ),
        down_transposed: push_leaf2::<AD>(
            leaves,
            format!("{prefix}.mlp.down_proj.weight"),
            &block.down_transposed,
        ),
    }
}

/// Pull every named leaf's gradient after one backward, all required.
fn remove_leaf_grads<AD: AutodiffBackend>(
    leaves: NamedLeaves<AD>,
    grads: &mut <AD as AutodiffBackend>::Gradients,
    into: &mut HashMap<String, ParamTensor<AD::InnerBackend>>,
) -> LibQuestResult<()> {
    for (name, leaf) in leaves {
        let grad = match leaf {
            LeafTensor::Vector(tensor) => match tensor.grad_remove(grads) {
                Some(grad) => ParamTensor::Vector(grad),
                None => snafu::whatever!("no full-parameter gradient for {name}"),
            },
            LeafTensor::Matrix(tensor) => match tensor.grad_remove(grads) {
                Some(grad) => ParamTensor::Matrix(grad),
                None => snafu::whatever!("no full-parameter gradient for {name}"),
            },
        };
        match into.remove(&name) {
            Some(existing) => {
                into.insert(name, add_param(existing, grad));
            }
            None => {
                into.insert(name, grad);
            }
        }
    }
    Ok(())
}

fn add_param<B: Backend>(left: ParamTensor<B>, right: ParamTensor<B>) -> ParamTensor<B> {
    match (left, right) {
        (ParamTensor::Vector(a), ParamTensor::Vector(b)) => ParamTensor::Vector(a + b),
        (ParamTensor::Matrix(a), ParamTensor::Matrix(b)) => ParamTensor::Matrix(a + b),
        _ => unreachable!("a parameter's rank never changes between backwards"),
    }
}

/// Pass one, no grad: each layer's input plus the top hidden state.
fn forward_cache_plain<B: Backend>(
    model: &HybridModel<B>,
    mask: &FloatTensor<B, 2>,
    inputs: &[u32],
    device: &B::Device,
) -> (Vec<FloatTensor<B, 2>>, FloatTensor<B, 2>) {
    let mut layer_inputs = Vec::with_capacity(model.layers.len());
    let mut x = model.embed(inputs);
    for block in &model.layers {
        layer_inputs.push(x.clone());
        x = match block {
            HybridBlock::Gdn(block) => block.forward(
                x,
                model.gdn_heads,
                model.gdn_key_dim,
                model.gdn_value_dim,
                model.eps,
                device,
                None,
            ),
            HybridBlock::Attn(block) => block.forward(
                x,
                mask,
                model.attn_heads,
                model.attn_head_dim,
                model.eps,
                None,
            ),
        };
    }
    (layer_inputs, x)
}

/// The loss block's yield: loss, seed, the final norm's grad, and the
/// per-position log-sum-exp trail the head formula reads.
struct OrganismLoss<B: Backend> {
    loss: f32,
    seed: FloatTensor<B, 2>,
    final_norm_grad: FloatTensor<B, 1>,
    lse: Vec<f32>,
}

/// The chunk-independent loss: each chunk is its own autodiff block,
/// so the vocabulary-wide logits never sit whole in one graph.
fn organism_loss_block<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    top_x: &FloatTensor<AD::InnerBackend, 2>,
    targets: &[u32],
    loss_chunk: usize,
    device: &AD::Device,
) -> LibQuestResult<OrganismLoss<AD::InnerBackend>> {
    let n = targets.len();
    snafu::ensure_whatever!(n > 0, "empty target sequence");
    let chunk = loss_chunk.max(1);
    let mut loss = 0f32;
    let mut seed_parts: Vec<FloatTensor<AD::InnerBackend, 2>> = Vec::new();
    let mut final_norm_grad: Option<FloatTensor<AD::InnerBackend, 1>> = None;
    let mut lse = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let end = (start + chunk).min(n);
        let rows = end - start;
        let top_leaf =
            FloatTensor::<AD, 2>::from_inner(top_x.clone().narrow(0, start, rows))
                .require_grad();
        let norm_leaf =
            FloatTensor::<AD, 1>::from_inner(model.final_norm.clone()).require_grad();
        let hidden = oracle::rms_norm(top_leaf.clone(), &norm_leaf, model.eps);
        let logits = hidden
            .matmul(FloatTensor::<AD, 2>::from_inner(model.lm_head_transposed.clone()))
            .cast(burn::tensor::FloatDType::F32);
        let log_probs = burn::tensor::activation::log_softmax(logits.clone(), 1);
        let indices: Vec<i64> = targets[start..end].iter().map(|t| *t as i64).collect();
        let index_tensor = burn::tensor::Tensor::<AD, 2, Int>::from_data(
            burn::tensor::TensorData::new(indices, [rows, 1]),
            device,
        );
        let picked_logit = logits.gather(1, index_tensor.clone());
        let picked_logprob = log_probs.gather(1, index_tensor);
        let lse_chunk = (picked_logit - picked_logprob.clone()).inner();
        let lse_values = match lse_chunk.into_data().convert::<f32>().to_vec::<f32>() {
            Ok(values) => values,
            Err(error) => snafu::whatever!("lse extraction failed: {error:?}"),
        };
        lse.extend(lse_values);
        let chunk_loss = picked_logprob.sum().neg().div_scalar(n as f64);
        loss += objective::scalar_of::<AD>(&chunk_loss);
        let mut grads = chunk_loss.backward();
        let Some(seed_chunk) = top_leaf.grad_remove(&mut grads) else {
            snafu::whatever!("no gradient reached the loss chunk at {start}");
        };
        seed_parts.push(seed_chunk);
        let Some(norm_grad) = norm_leaf.grad_remove(&mut grads) else {
            snafu::whatever!("no gradient reached the final norm at {start}");
        };
        final_norm_grad = Some(match final_norm_grad {
            Some(existing) => existing + norm_grad,
            None => norm_grad,
        });
        start = end;
    }
    Ok(OrganismLoss {
        loss,
        seed: burn::tensor::Tensor::cat(seed_parts, 0),
        final_norm_grad: final_norm_grad.expect("targets are non-empty"),
        lse,
    })
}

/// Pass two: the full-parameter per-layer VJP, top down from the seed.
/// Returns the core gradients and the embedding-output gradient.
#[allow(clippy::type_complexity)]
fn organism_chain_from_seed<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    mask_inner: &FloatTensor<AD::InnerBackend, 2>,
    layer_inputs: &[FloatTensor<AD::InnerBackend, 2>],
    seed: FloatTensor<AD::InnerBackend, 2>,
    segment: usize,
    device: &AD::Device,
) -> LibQuestResult<(
    HashMap<String, ParamTensor<AD::InnerBackend>>,
    FloatTensor<AD::InnerBackend, 2>,
)> {
    let mut grad_out = seed;
    let mut collected: HashMap<String, ParamTensor<AD::InnerBackend>> = HashMap::new();
    for (index, block) in model.layers.iter().enumerate().rev() {
        let prefix = format!("model.layers.{index}");
        match block {
            HybridBlock::Gdn(block) => {
                let grad_in = gdn_layer_backward::<AD>(
                    block,
                    model,
                    &layer_inputs[index],
                    &grad_out,
                    segment,
                    &prefix,
                    &mut collected,
                    device,
                )?;
                grad_out = grad_in;
            }
            HybridBlock::Attn(block) => {
                let mut leaves: NamedLeaves<AD> = Vec::new();
                let lifted = lift_attn_trainable::<AD>(block, &prefix, &mut leaves);
                let x_leaf =
                    FloatTensor::<AD, 2>::from_inner(layer_inputs[index].clone())
                        .require_grad();
                let mask_ad = FloatTensor::<AD, 2>::from_inner(mask_inner.clone());
                let out = lifted.forward(
                    x_leaf.clone(),
                    &mask_ad,
                    model.attn_heads,
                    model.attn_head_dim,
                    model.eps,
                    None,
                );
                let pseudo_loss =
                    (out * FloatTensor::from_inner(grad_out.clone())).sum();
                let mut grads = pseudo_loss.backward();
                remove_leaf_grads::<AD>(leaves, &mut grads, &mut collected)?;
                let Some(next_grad) = x_leaf.grad_remove(&mut grads) else {
                    snafu::whatever!("no input gradient at layer {index}");
                };
                grad_out = next_grad;
            }
        }
    }
    Ok((collected, grad_out))
}

/// One GDN layer's full-parameter VJP with the recurrence segmented.
#[allow(clippy::too_many_arguments)]
fn gdn_layer_backward<AD: AutodiffBackend>(
    block: &GdnBlock<AD::InnerBackend>,
    model: &HybridModel<AD::InnerBackend>,
    block_input: &FloatTensor<AD::InnerBackend, 2>,
    grad_out: &FloatTensor<AD::InnerBackend, 2>,
    segment: usize,
    prefix: &str,
    collected: &mut HashMap<String, ParamTensor<AD::InnerBackend>>,
    device: &AD::Device,
) -> LibQuestResult<FloatTensor<AD::InnerBackend, 2>> {
    let heads = model.gdn_heads;
    let key_dim = model.gdn_key_dim;
    let value_dim = model.gdn_value_dim;
    let eps = model.eps;
    let segment = segment.max(1);
    let n = block_input.dims()[0];

    // Pass one, no grad: pre products, boundary states, the output.
    let pre = block.gdn_pre(
        block_input.clone(),
        heads,
        key_dim,
        value_dim,
        eps,
        device,
        None,
    );
    let mut state = FloatTensor::<AD::InnerBackend, 3>::zeros(
        [heads, key_dim, value_dim],
        device,
    )
    .cast(burn::tensor::FloatDType::F32);
    let mut boundaries: Vec<FloatTensor<AD::InnerBackend, 3>> = Vec::new();
    let mut y_parts: Vec<FloatTensor<AD::InnerBackend, 3>> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let len = segment.min(n - start);
        boundaries.push(state.clone());
        let (y_segment, next) = oracle::gdn_recurrence::<AD::InnerBackend>(
            pre.q.clone().narrow(0, start, len),
            pre.k.clone().narrow(0, start, len),
            pre.v.clone().narrow(0, start, len),
            pre.decay.clone().narrow(0, start, len),
            pre.beta.clone().narrow(0, start, len),
            state,
        );
        y_parts.push(y_segment);
        state = next;
        start += len;
    }
    let y_full = burn::tensor::Tensor::cat(y_parts, 0);

    // The post half's backward: output-side weights, y and gate grads.
    let mut post_leaves: NamedLeaves<AD> = Vec::new();
    let post_block =
        lift_gdn_trainable::<AD>(block, GdnLeafSet::Post, prefix, &mut post_leaves);
    let x_post = FloatTensor::<AD, 2>::from_inner(block_input.clone()).require_grad();
    let y_leaf = FloatTensor::<AD, 3>::from_inner(y_full).require_grad();
    let gate_leaf =
        FloatTensor::<AD, 2>::from_inner(pre.gate_rows.clone()).require_grad();
    let out = post_block.gdn_post(
        x_post.clone(),
        y_leaf.clone(),
        gate_leaf.clone(),
        heads,
        value_dim,
        eps,
        None,
    );
    let pseudo_loss = (out * FloatTensor::from_inner(grad_out.clone())).sum();
    let mut post_grads = pseudo_loss.backward();
    remove_leaf_grads::<AD>(post_leaves, &mut post_grads, collected)?;
    let Some(grad_y) = y_leaf.grad_remove(&mut post_grads) else {
        snafu::whatever!("no gradient reached the recurrence output at {prefix}");
    };
    let Some(grad_gate_rows) = gate_leaf.grad_remove(&mut post_grads) else {
        snafu::whatever!("no gradient reached the gate rows at {prefix}");
    };
    let Some(grad_x_post) = x_post.grad_remove(&mut post_grads) else {
        snafu::whatever!("no post-half input gradient at {prefix}");
    };

    // The recurrence backward, one segment at a time from the top.
    let segment_count = boundaries.len();
    let mut adjoint: Option<FloatTensor<AD::InnerBackend, 3>> = None;
    let mut grad_q_parts = Vec::with_capacity(segment_count);
    let mut grad_k_parts = Vec::with_capacity(segment_count);
    let mut grad_v_parts = Vec::with_capacity(segment_count);
    let mut grad_d_parts = Vec::with_capacity(segment_count);
    let mut grad_b_parts = Vec::with_capacity(segment_count);
    for index in (0..segment_count).rev() {
        let start = index * segment;
        let len = segment.min(n - start);
        let state_in =
            FloatTensor::<AD, 3>::from_inner(boundaries[index].clone()).require_grad();
        let q_leaf = FloatTensor::<AD, 3>::from_inner(pre.q.clone().narrow(0, start, len))
            .require_grad();
        let k_leaf = FloatTensor::<AD, 3>::from_inner(pre.k.clone().narrow(0, start, len))
            .require_grad();
        let v_leaf = FloatTensor::<AD, 3>::from_inner(pre.v.clone().narrow(0, start, len))
            .require_grad();
        let d_leaf =
            FloatTensor::<AD, 2>::from_inner(pre.decay.clone().narrow(0, start, len))
                .require_grad();
        let b_leaf =
            FloatTensor::<AD, 2>::from_inner(pre.beta.clone().narrow(0, start, len))
                .require_grad();
        let (y_segment, state_out) = oracle::gdn_recurrence::<AD>(
            q_leaf.clone(),
            k_leaf.clone(),
            v_leaf.clone(),
            d_leaf.clone(),
            b_leaf.clone(),
            state_in.clone(),
        );
        let mut pseudo_loss = (y_segment
            * FloatTensor::from_inner(grad_y.clone().narrow(0, start, len)))
        .sum();
        if let Some(above) = &adjoint {
            pseudo_loss =
                pseudo_loss + (state_out * FloatTensor::from_inner(above.clone())).sum();
        }
        let mut segment_grads = pseudo_loss.backward();
        let mut take3 = |leaf: FloatTensor<AD, 3>,
                         what: &str|
         -> LibQuestResult<FloatTensor<AD::InnerBackend, 3>> {
            match leaf.grad_remove(&mut segment_grads) {
                Some(grad) => Ok(grad),
                None => snafu::whatever!("no {what} gradient in segment {index} at {prefix}"),
            }
        };
        grad_q_parts.push(take3(q_leaf, "query")?);
        grad_k_parts.push(take3(k_leaf, "key")?);
        grad_v_parts.push(take3(v_leaf, "value")?);
        let Some(grad_d) = d_leaf.grad_remove(&mut segment_grads) else {
            snafu::whatever!("no decay gradient in segment {index} at {prefix}");
        };
        let Some(grad_b) = b_leaf.grad_remove(&mut segment_grads) else {
            snafu::whatever!("no beta gradient in segment {index} at {prefix}");
        };
        let Some(grad_state) = state_in.grad_remove(&mut segment_grads) else {
            snafu::whatever!("no state adjoint in segment {index} at {prefix}");
        };
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

    // The pre half's backward: input-side weights and the input grad.
    let mut pre_leaves: NamedLeaves<AD> = Vec::new();
    let pre_block =
        lift_gdn_trainable::<AD>(block, GdnLeafSet::Pre, prefix, &mut pre_leaves);
    let x_pre = FloatTensor::<AD, 2>::from_inner(block_input.clone()).require_grad();
    let pre_ad = pre_block.gdn_pre(
        x_pre.clone(),
        heads,
        key_dim,
        value_dim,
        eps,
        device,
        None,
    );
    let pseudo_loss = (pre_ad.q * FloatTensor::from_inner(grad_q)).sum()
        + (pre_ad.k * FloatTensor::from_inner(grad_k)).sum()
        + (pre_ad.v * FloatTensor::from_inner(grad_v)).sum()
        + (pre_ad.decay * FloatTensor::from_inner(grad_d)).sum()
        + (pre_ad.beta * FloatTensor::from_inner(grad_b)).sum()
        + (pre_ad.gate_rows * FloatTensor::from_inner(grad_gate_rows)).sum();
    let mut pre_grads = pseudo_loss.backward();
    remove_leaf_grads::<AD>(pre_leaves, &mut pre_grads, collected)?;
    let Some(grad_x_pre) = x_pre.grad_remove(&mut pre_grads) else {
        snafu::whatever!("no pre-half input gradient at {prefix}");
    };
    Ok(grad_x_post + grad_x_pre)
}

/// The sorted unique row set one chunk touches: inputs plus targets.
fn touched_rows(inputs: &[u32], targets: &[u32]) -> Vec<u32> {
    let mut rows: Vec<u32> = inputs.iter().chain(targets).copied().collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The embedding's touched-row gradients: the input-side scatter plus
/// the head-side softmax formula, exact for every touched row. The
/// dense remainder (the softmax denominator's push on untouched rows)
/// is deliberately dropped: only touched rows update, so only touched
/// rows carry Adam state (TheUser's row-sparse lever).
#[allow(clippy::too_many_arguments)]
fn embedding_row_grads<B: Backend>(
    model: &HybridModel<B>,
    top_x: &FloatTensor<B, 2>,
    seed_bottom: &FloatTensor<B, 2>,
    lse: &[f32],
    inputs: &[u32],
    targets: &[u32],
    device: &B::Device,
) -> LibQuestResult<HashMap<u32, Vec<f32>>> {
    let hidden = model.hidden;
    let n = targets.len();
    let touched = touched_rows(inputs, targets);
    let touched_count = touched.len();
    let position_of = |row: u32| -> usize {
        touched.binary_search(&row).expect("touched rows cover inputs and targets")
    };

    // Head side: p = exp(logit - lse) over the touched columns alone.
    let normed = oracle::rms_norm(top_x.clone(), &model.final_norm, model.eps)
        .cast(burn::tensor::FloatDType::F32);
    let indices: Vec<i64> = touched.iter().map(|row| *row as i64).collect();
    let index_tensor = burn::tensor::Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(indices, [touched_count]),
        device,
    );
    let w_touched = model
        .lm_head_transposed
        .clone()
        .select(1, index_tensor)
        .cast(burn::tensor::FloatDType::F32);
    let logits_touched = normed.clone().matmul(w_touched);
    let lse_tensor = FloatTensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(lse.to_vec(), [n, 1]),
        device,
    )
    .cast(burn::tensor::FloatDType::F32)
    .expand([n, touched_count]);
    let probabilities = (logits_touched - lse_tensor).exp();
    let mut one_hot = vec![0f32; n * touched_count];
    for (position, target) in targets.iter().enumerate() {
        one_hot[position * touched_count + position_of(*target)] = 1.0;
    }
    let one_hot_tensor = FloatTensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(one_hot, [n, touched_count]),
        device,
    )
    .cast(burn::tensor::FloatDType::F32);
    let logit_grads = (probabilities - one_hot_tensor).div_scalar(n as f64);
    let head_grads = logit_grads.swap_dims(0, 1).matmul(normed);
    let head_values = match head_grads.into_data().convert::<f32>().to_vec::<f32>() {
        Ok(values) => values,
        Err(error) => snafu::whatever!("head gradient extraction failed: {error:?}"),
    };

    // Input side: scatter the embedding-output gradient by input id.
    let seed_values = match seed_bottom
        .clone()
        .cast(burn::tensor::FloatDType::F32)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
    {
        Ok(values) => values,
        Err(error) => snafu::whatever!("seed gradient extraction failed: {error:?}"),
    };
    let mut rows: HashMap<u32, Vec<f32>> = HashMap::with_capacity(touched_count);
    for (position, row) in touched.iter().enumerate() {
        rows.insert(
            *row,
            head_values[position * hidden..(position + 1) * hidden].to_vec(),
        );
    }
    for (position, input) in inputs.iter().enumerate() {
        let row = rows.get_mut(input).expect("touched rows cover inputs");
        for (slot, value) in row
            .iter_mut()
            .zip(&seed_values[position * hidden..(position + 1) * hidden])
        {
            *slot += *value;
        }
    }
    Ok(rows)
}

/// Merge one microbatch's gradients into the accumulator.
fn accumulate_organism_grads<B: Backend>(
    into: &mut OrganismGrads<B>,
    core: HashMap<String, ParamTensor<B>>,
    rows: HashMap<u32, Vec<f32>>,
) {
    for (name, grad) in core {
        match into.core.remove(&name) {
            Some(existing) => {
                into.core.insert(name, add_param(existing, grad));
            }
            None => {
                into.core.insert(name, grad);
            }
        }
    }
    for (row, values) in rows {
        match into.rows.get_mut(&row) {
            Some(existing) => {
                for (slot, value) in existing.iter_mut().zip(&values) {
                    *slot += *value;
                }
            }
            None => {
                into.rows.insert(row, values);
            }
        }
    }
}

/// One AdamW update on an f32 master tensor pair, in place.
pub(crate) fn adamw_tensor<B: Backend, const D: usize>(
    master: &mut FloatTensor<B, D>,
    moment: &mut FloatTensor<B, D>,
    velocity: &mut FloatTensor<B, D>,
    grad: FloatTensor<B, D>,
    learning_rate: f64,
    bias1: f64,
    bias2: f64,
) {
    let grad = grad.cast(burn::tensor::FloatDType::F32);
    *moment = moment.clone().mul_scalar(ADAM_BETA1) + grad.clone().mul_scalar(1.0 - ADAM_BETA1);
    *velocity = velocity.clone().mul_scalar(ADAM_BETA2)
        + grad.clone().mul_scalar(1.0 - ADAM_BETA2) * grad;
    let moment_hat = moment.clone().div_scalar(bias1);
    let velocity_hat = velocity.clone().div_scalar(bias2);
    *master = master.clone()
        - moment_hat.div(velocity_hat.sqrt().add_scalar(ADAM_EPS)).mul_scalar(learning_rate);
}

impl<AD: AutodiffBackend> OrganismTrainer<AD> {
    /// Load an organism checkpoint into a trainable session.
    pub fn load(
        model_dir: &Path,
        device: &<AD::InnerBackend as burn::tensor::backend::BackendTypes>::Device,
    ) -> LibQuestResult<Self> {
        let config = HybridCheckpointConfig::load(model_dir)?;
        let config_json = fs::read(model_dir.join("config.json"))?;
        let weights = HybridWeights::load(model_dir)?;
        let model = HybridModel::<AD::InnerBackend>::new(&config, weights, device.clone())?;
        Self::from_parts(config, config_json, model, device)
    }

    /// Assemble a session over an already-built model.
    pub(crate) fn from_parts(
        config: HybridCheckpointConfig,
        config_json: Vec<u8>,
        mut model: HybridModel<AD::InnerBackend>,
        device: &<AD::InnerBackend as burn::tensor::backend::BackendTypes>::Device,
    ) -> LibQuestResult<Self> {
        snafu::ensure_whatever!(
            config.tie_word_embeddings,
            "the organism trainer wants tied embeddings (the organism posture)"
        );
        let mut masters = HashMap::new();
        for_each_core_param(&mut model, &mut |name, field| {
            let state = match field {
                ParamField::Vector(tensor) => {
                    let master = tensor.clone().cast(burn::tensor::FloatDType::F32);
                    MasterState {
                        moment: ParamTensor::Vector(master.zeros_like()),
                        velocity: ParamTensor::Vector(master.zeros_like()),
                        master: ParamTensor::Vector(master),
                    }
                }
                ParamField::Matrix(tensor) => {
                    let master = tensor.clone().cast(burn::tensor::FloatDType::F32);
                    MasterState {
                        moment: ParamTensor::Matrix(master.zeros_like()),
                        velocity: ParamTensor::Matrix(master.zeros_like()),
                        master: ParamTensor::Matrix(master),
                    }
                }
            };
            masters.insert(name, state);
            Ok(())
        })?;
        let probed = FloatTensor::<AD::InnerBackend, 1>::zeros([1], device)
            .into_data()
            .dtype;
        Ok(Self {
            model,
            config,
            config_json,
            masters,
            embedding_moments: HashMap::new(),
            step: 0,
            compute_dtype: float_dtype_of(probed)?,
            device: device.clone(),
        })
    }

    /// Optimizer steps taken so far.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// One microbatch's gradients over a packed chunk, no update.
    pub fn grads_for_chunk(
        &self,
        chunk: &[u32],
        loss_chunk: usize,
        segment: usize,
    ) -> LibQuestResult<(f32, OrganismGrads<AD::InnerBackend>)> {
        snafu::ensure_whatever!(
            chunk.len() >= 2,
            "a training chunk wants at least two ids, got {}",
            chunk.len()
        );
        let seq_len = chunk.len() - 1;
        let inputs = &chunk[..seq_len];
        let targets = &chunk[1..];
        let vocab = self.config.vocab_size as u32;
        if let Some(bad) = chunk.iter().find(|id| **id >= vocab) {
            snafu::whatever!("token id {bad} is outside the {vocab}-row vocabulary");
        }
        let mask = oracle::causal_mask::<AD::InnerBackend>(seq_len, &self.device);
        let (layer_inputs, top_x) =
            forward_cache_plain(&self.model, &mask, inputs, &self.device);
        let loss =
            organism_loss_block::<AD>(&self.model, &top_x, targets, loss_chunk, &self.device)?;
        let (mut core, seed_bottom) = organism_chain_from_seed::<AD>(
            &self.model,
            &mask,
            &layer_inputs,
            loss.seed,
            segment,
            &self.device,
        )?;
        match core.remove("model.norm.weight") {
            Some(existing) => {
                core.insert(
                    String::from("model.norm.weight"),
                    add_param(existing, ParamTensor::Vector(loss.final_norm_grad)),
                );
            }
            None => {
                core.insert(
                    String::from("model.norm.weight"),
                    ParamTensor::Vector(loss.final_norm_grad),
                );
            }
        }
        let rows = embedding_row_grads(
            &self.model,
            &top_x,
            &seed_bottom,
            &loss.lse,
            inputs,
            targets,
            &self.device,
        )?;
        Ok((loss.loss, OrganismGrads { core, rows }))
    }

    /// Apply one AdamW step from accumulated gradients, then write the
    /// updated masters back into the model's compute tensors.
    pub fn apply_step(
        &mut self,
        mut grads: OrganismGrads<AD::InnerBackend>,
        learning_rate: f64,
    ) -> LibQuestResult<()> {
        self.step += 1;
        let bias1 = 1.0 - ADAM_BETA1.powi(self.step as i32);
        let bias2 = 1.0 - ADAM_BETA2.powi(self.step as i32);

        for (name, state) in self.masters.iter_mut() {
            let Some(grad) = grads.core.remove(name) else {
                snafu::whatever!("missing accumulated gradient for {name}");
            };
            match (state, grad) {
                (
                    MasterState {
                        master: ParamTensor::Vector(master),
                        moment: ParamTensor::Vector(moment),
                        velocity: ParamTensor::Vector(velocity),
                    },
                    ParamTensor::Vector(grad),
                ) => adamw_tensor(master, moment, velocity, grad, learning_rate, bias1, bias2),
                (
                    MasterState {
                        master: ParamTensor::Matrix(master),
                        moment: ParamTensor::Matrix(moment),
                        velocity: ParamTensor::Matrix(velocity),
                    },
                    ParamTensor::Matrix(grad),
                ) => adamw_tensor(master, moment, velocity, grad, learning_rate, bias1, bias2),
                _ => snafu::whatever!("gradient rank mismatch for {name}"),
            }
        }
        snafu::ensure_whatever!(
            grads.core.is_empty(),
            "a gradient arrived for no master: {:?}",
            grads.core.keys().next()
        );

        // Write the masters back at the model's own compute dtype.
        let compute = self.compute_dtype;
        let masters = &self.masters;
        for_each_core_param(&mut self.model, &mut |name, field| {
            let state = masters.get(&name).expect("every parameter has a master");
            match (field, &state.master) {
                (ParamField::Vector(tensor), ParamTensor::Vector(master)) => {
                    *tensor = master.clone().cast(compute);
                }
                (ParamField::Matrix(tensor), ParamTensor::Matrix(master)) => {
                    *tensor = master.clone().cast(compute);
                }
                _ => unreachable!("the walk and the master store share one shape"),
            }
            Ok(())
        })?;

        // Row-sparse embedding update: host master rows, then the head
        // columns re-anchored to the master so rounding never compounds.
        let hidden = self.model.hidden;
        let mut touched: Vec<u32> = grads.rows.keys().copied().collect();
        touched.sort_unstable();
        for row in &touched {
            let grad = &grads.rows[row];
            let state = self
                .embedding_moments
                .entry(*row)
                .or_insert_with(|| EmbeddingRowMoments {
                    moment: vec![0f32; hidden],
                    velocity: vec![0f32; hidden],
                });
            let master =
                &mut self.model.embed_rows[*row as usize * hidden..(*row as usize + 1) * hidden];
            for slot in 0..hidden {
                let g = grad[slot];
                state.moment[slot] =
                    (ADAM_BETA1 * state.moment[slot] as f64 + (1.0 - ADAM_BETA1) * g as f64) as f32;
                state.velocity[slot] = (ADAM_BETA2 * state.velocity[slot] as f64
                    + (1.0 - ADAM_BETA2) * (g as f64) * (g as f64))
                    as f32;
                let moment_hat = state.moment[slot] as f64 / bias1;
                let velocity_hat = state.velocity[slot] as f64 / bias2;
                master[slot] -=
                    (learning_rate * moment_hat / (velocity_hat.sqrt() + ADAM_EPS)) as f32;
            }
        }
        if !touched.is_empty() {
            let touched_count = touched.len();
            let mut new_rows = Vec::with_capacity(touched_count * hidden);
            for row in &touched {
                new_rows.extend_from_slice(
                    &self.model.embed_rows
                        [*row as usize * hidden..(*row as usize + 1) * hidden],
                );
            }
            let indices: Vec<i64> = touched.iter().map(|row| *row as i64).collect();
            let index_tensor = burn::tensor::Tensor::<AD::InnerBackend, 1, Int>::from_data(
                burn::tensor::TensorData::new(indices, [touched_count]),
                &self.device,
            );
            let new_columns = FloatTensor::<AD::InnerBackend, 2>::from_data(
                burn::tensor::TensorData::new(new_rows, [touched_count, hidden]),
                &self.device,
            )
            .swap_dims(0, 1);
            // Float select_assign implements only Add, so the columns
            // re-anchor as old + (new - old): equal to the master
            // within one rounding, re-derived whole every step.
            let old_columns = self
                .model
                .lm_head_transposed
                .clone()
                .select(1, index_tensor.clone());
            let delta = new_columns - old_columns;
            self.model.lm_head_transposed = self.model.lm_head_transposed.clone().select_assign(
                1,
                index_tensor,
                delta,
                burn::tensor::IndexingUpdateOp::Add,
            );
        }
        Ok(())
    }

    /// The stage loop: shuffled chunk cycling under warmup AdamW.
    pub fn train(
        &mut self,
        chunks: &[Vec<u32>],
        options: &OrganismLoopOptions,
        checkpoint_dir: Option<&Path>,
        mut on_step: impl FnMut(&train::StepLog) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    ) -> LibQuestResult<train::TrainReport> {
        snafu::ensure_whatever!(!chunks.is_empty(), "no training chunks");
        let seq_len = chunks[0].len() - 1;
        let accumulate = options.accumulate.max(1);
        let mut order: Vec<usize> = (0..chunks.len()).collect();
        let mut order_epoch = usize::MAX;
        let mut first_loss = 0f32;
        let mut last_loss = 0f32;
        let mut trained_tokens = 0usize;
        let started = std::time::Instant::now();
        for step in 0..options.steps {
            let mut accumulated = OrganismGrads {
                core: HashMap::new(),
                rows: HashMap::new(),
            };
            let mut batch_loss = 0f32;
            for slot in 0..accumulate {
                let flat = step * accumulate + slot;
                let epoch = flat / chunks.len();
                if epoch != order_epoch {
                    let mut rng = SplitMix64::new(CHUNK_SHUFFLE_SEED ^ epoch as u64);
                    for high in (1..order.len()).rev() {
                        let pick = rng.next_below(high + 1);
                        order.swap(high, pick);
                    }
                    order_epoch = epoch;
                }
                let chunk = &chunks[order[flat % chunks.len()]];
                snafu::ensure_whatever!(
                    chunk.len() == seq_len + 1,
                    "chunk {flat} carries {} ids for seq_len {seq_len}",
                    chunk.len()
                );
                let (loss, grads) = self.grads_for_chunk(
                    chunk,
                    options.loss_chunk,
                    options.recurrence_segment,
                )?;
                accumulate_organism_grads(&mut accumulated, grads.core, grads.rows);
                batch_loss += loss;
                trained_tokens += seq_len;
            }
            if accumulate > 1 {
                let scale = 1.0 / accumulate as f64;
                for grad in accumulated.core.values_mut() {
                    *grad = match grad {
                        ParamTensor::Vector(tensor) => {
                            ParamTensor::Vector(tensor.clone().mul_scalar(scale))
                        }
                        ParamTensor::Matrix(tensor) => {
                            ParamTensor::Matrix(tensor.clone().mul_scalar(scale))
                        }
                    };
                }
                for row in accumulated.rows.values_mut() {
                    for value in row.iter_mut() {
                        *value *= scale as f32;
                    }
                }
            }
            let batch_loss = batch_loss / accumulate as f32;
            if step == 0 {
                first_loss = batch_loss;
            }
            last_loss = batch_loss;

            let learning_rate =
                train::warmup_rate(options.learning_rate, options.warmup_steps, step);
            self.apply_step(accumulated, learning_rate)?;

            let elapsed = started.elapsed().as_secs_f64();
            let log = train::StepLog {
                step: step + 1,
                loss: batch_loss,
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
                    "organism train: step {} | loss {:.4} | lr {:.2e} | {:.0} tok/s",
                    log.step, log.loss, log.learning_rate, log.tokens_per_second
                );
            }
            if options.checkpoint_every > 0
                && (step + 1) % options.checkpoint_every == 0
                && step + 1 < options.steps
                && let Some(dir) = checkpoint_dir
            {
                self.save_checkpoint(&dir.join(format!("step_{}", step + 1)))?;
            }
        }
        Ok(train::TrainReport {
            first_loss,
            final_loss: last_loss,
            steps: options.steps,
            trained_tokens,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// Write the trained organism as a checkpoint the loaders read:
    /// the verbatim config.json plus bf16 tensors from the masters,
    /// the tied embedding stored under both names.
    pub fn save_checkpoint(&self, out_dir: &Path) -> LibQuestResult<()> {
        fs::create_dir_all(out_dir)?;
        fs::write(out_dir.join("config.json"), &self.config_json)?;

        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let embed_bytes: Vec<u8> = self
            .model
            .embed_rows
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
            .collect();
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = vec![
            (
                String::from("model.embed_tokens.weight"),
                vec![vocab, hidden],
                embed_bytes.clone(),
            ),
            (
                String::from("lm_head.weight"),
                vec![vocab, hidden],
                embed_bytes,
            ),
        ];

        let mut host: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (name, state) in &self.masters {
            let (shape, values) = match &state.master {
                ParamTensor::Vector(tensor) => {
                    let dims = tensor.dims().to_vec();
                    (dims, tensor_host_values1(tensor)?)
                }
                ParamTensor::Matrix(tensor) => {
                    let dims = tensor.dims().to_vec();
                    (dims, tensor_host_values2(tensor)?)
                }
            };
            host.insert(name.clone(), (shape, values));
        }
        let mut taken: HashSet<String> = HashSet::new();
        for name in self.masters.keys() {
            if let Some(tap_at) = name.find(".tap") {
                let base = &name[..tap_at];
                if !taken.insert(base.to_string()) {
                    continue;
                }
                let kernel = self.config.linear_conv_kernel_dim;
                let channels = host[&format!("{base}.tap0")].0[1];
                let mut values = vec![0f32; channels * kernel];
                for tap in 0..kernel {
                    let (_, row) = &host[&format!("{base}.tap{tap}")];
                    for (channel, value) in row.iter().enumerate() {
                        values[channel * kernel + tap] = *value;
                    }
                }
                tensors.push((
                    base.to_string(),
                    vec![channels, 1, kernel],
                    f32_bf16_bytes(&values),
                ));
                continue;
            }
            let (shape, values) = &host[name];
            if name.ends_with(".A_log") || name.ends_with(".dt_bias") {
                tensors.push((name.clone(), vec![shape[1]], f32_bf16_bytes(values)));
            } else if shape.len() == 1 {
                tensors.push((name.clone(), shape.clone(), f32_bf16_bytes(values)));
            } else {
                // Masters hold the transposed (in, out) orientation the
                // oracle computes with; the checkpoint stores (out, in).
                let (rows, columns) = (shape[0], shape[1]);
                let mut transposed = vec![0f32; values.len()];
                for row in 0..rows {
                    for column in 0..columns {
                        transposed[column * rows + row] = values[row * columns + column];
                    }
                }
                tensors.push((
                    name.clone(),
                    vec![columns, rows],
                    f32_bf16_bytes(&transposed),
                ));
            }
        }

        tensors.sort_by(|left, right| left.0.cmp(&right.0));
        let views: Vec<(String, safetensors::tensor::TensorView)> = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                match safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    shape.clone(),
                    bytes,
                ) {
                    Ok(view) => Ok((name.clone(), view)),
                    Err(error) => snafu::whatever!("tensor view {name} failed: {error}"),
                }
            })
            .collect::<LibQuestResult<Vec<_>>>()?;
        let mut metadata: HashMap<String, String> = HashMap::new();
        metadata.insert(String::from("kind"), String::from("imagine_quest_organism"));
        metadata.insert(String::from("trained_steps"), self.step.to_string());
        metadata.insert(
            String::from("tool_version"),
            String::from(env!("CARGO_PKG_VERSION")),
        );
        match safetensors::serialize_to_file(
            views,
            Some(metadata),
            &out_dir.join("model.safetensors"),
        ) {
            Ok(()) => {}
            Err(error) => snafu::whatever!("checkpoint write failed: {error}"),
        }
        crate::load_config(out_dir)?;
        Ok(())
    }
}

/// A rank-1 tensor's host f32 values.
fn tensor_host_values1<B: Backend>(
    tensor: &FloatTensor<B, 1>,
) -> LibQuestResult<Vec<f32>> {
    match tensor.clone().into_data().convert::<f32>().to_vec::<f32>() {
        Ok(values) => Ok(values),
        Err(error) => snafu::whatever!("tensor extraction failed: {error:?}"),
    }
}

/// A rank-2 tensor's host f32 values, row-major.
fn tensor_host_values2<B: Backend>(
    tensor: &FloatTensor<B, 2>,
) -> LibQuestResult<Vec<f32>> {
    match tensor.clone().into_data().convert::<f32>().to_vec::<f32>() {
        Ok(values) => Ok(values),
        Err(error) => snafu::whatever!("tensor extraction failed: {error:?}"),
    }
}

fn f32_bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
        .collect()
}

/// The FloatDType a backend's probed DType corresponds to.
fn float_dtype_of(dtype: burn::tensor::DType) -> LibQuestResult<burn::tensor::FloatDType> {
    Ok(match dtype {
        burn::tensor::DType::F64 => burn::tensor::FloatDType::F64,
        burn::tensor::DType::F32 => burn::tensor::FloatDType::F32,
        burn::tensor::DType::F16 => burn::tensor::FloatDType::F16,
        burn::tensor::DType::BF16 => burn::tensor::FloatDType::BF16,
        other => snafu::whatever!("no float compute dtype for {other:?}"),
    })
}

/// The unsegmented reference step: every weight a leaf in one graph,
/// the head tracked densely. Toy scale only; the gate's oracle.
#[cfg(test)]
pub(crate) struct ReferenceGrads<B: Backend> {
    pub(crate) loss: f32,
    pub(crate) core: HashMap<String, ParamTensor<B>>,
    /// The dense (vocab, hidden) embedding gradient: the input-side
    /// scatter plus the whole head-side term, untouched rows included.
    pub(crate) embedding: Vec<f32>,
}

#[cfg(test)]
pub(crate) fn reference_step<AD: AutodiffBackend>(
    model: &HybridModel<AD::InnerBackend>,
    inputs: &[u32],
    targets: &[u32],
    device: &AD::Device,
) -> LibQuestResult<ReferenceGrads<AD::InnerBackend>> {
    let seq_len = inputs.len();
    let hidden = model.hidden;
    let vocab = model.lm_head_transposed.dims()[1];
    let mask = oracle::causal_mask::<AD::InnerBackend>(seq_len, device);
    let mask_ad = FloatTensor::<AD, 2>::from_inner(mask);
    let mut leaves: NamedLeaves<AD> = Vec::new();
    let mut lifted: Vec<HybridBlock<AD>> = Vec::new();
    for (index, block) in model.layers.iter().enumerate() {
        let prefix = format!("model.layers.{index}");
        lifted.push(match block {
            HybridBlock::Gdn(block) => HybridBlock::Gdn(lift_gdn_trainable::<AD>(
                block,
                GdnLeafSet::All,
                &prefix,
                &mut leaves,
            )),
            HybridBlock::Attn(block) => {
                HybridBlock::Attn(lift_attn_trainable::<AD>(block, &prefix, &mut leaves))
            }
        });
    }
    let embed_leaf =
        FloatTensor::<AD, 2>::from_inner(model.embed(inputs)).require_grad();
    let norm_leaf = push_leaf1::<AD>(
        &mut leaves,
        String::from("model.norm.weight"),
        &model.final_norm,
    );
    let head_leaf =
        FloatTensor::<AD, 2>::from_inner(model.lm_head_transposed.clone()).require_grad();

    let mut x = embed_leaf.clone();
    for block in &lifted {
        x = match block {
            HybridBlock::Gdn(block) => block.forward(
                x,
                model.gdn_heads,
                model.gdn_key_dim,
                model.gdn_value_dim,
                model.eps,
                device,
                None,
            ),
            HybridBlock::Attn(block) => block.forward(
                x,
                &mask_ad,
                model.attn_heads,
                model.attn_head_dim,
                model.eps,
                None,
            ),
        };
    }
    let hidden_rows = oracle::rms_norm(x, &norm_leaf, model.eps);
    let logits = hidden_rows
        .matmul(head_leaf.clone())
        .cast(burn::tensor::FloatDType::F32);
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);
    let indices: Vec<i64> = targets.iter().map(|t| *t as i64).collect();
    let index_tensor = burn::tensor::Tensor::<AD, 2, Int>::from_data(
        burn::tensor::TensorData::new(indices, [targets.len(), 1]),
        device,
    );
    let loss = log_probs
        .gather(1, index_tensor)
        .sum()
        .neg()
        .div_scalar(targets.len() as f64);
    let loss_value = objective::scalar_of::<AD>(&loss);
    let mut grads = loss.backward();
    let mut core = HashMap::new();
    remove_leaf_grads::<AD>(leaves, &mut grads, &mut core)?;

    let Some(head_grad) = head_leaf.grad_remove(&mut grads) else {
        snafu::whatever!("no reference head gradient");
    };
    let Some(embed_out_grad) = embed_leaf.grad_remove(&mut grads) else {
        snafu::whatever!("no reference embedding-output gradient");
    };
    // (hidden, vocab) transposed to per-row order, then the scatter.
    let head_values = tensor_host_values2(&head_grad.swap_dims(0, 1))?;
    let seed_values = tensor_host_values2(&embed_out_grad)?;
    let mut embedding = head_values;
    for (position, input) in inputs.iter().enumerate() {
        for slot in 0..hidden {
            embedding[*input as usize * hidden + slot] +=
                seed_values[position * hidden + slot];
        }
    }
    debug_assert_eq!(embedding.len(), vocab * hidden);
    Ok(ReferenceGrads {
        loss: loss_value,
        core,
        embedding,
    })
}

#[cfg(test)]
pub(crate) fn param_host_values<B: Backend>(
    tensor: &ParamTensor<B>,
) -> LibQuestResult<Vec<f32>> {
    match tensor {
        ParamTensor::Vector(tensor) => tensor_host_values1(tensor),
        ParamTensor::Matrix(tensor) => tensor_host_values2(tensor),
    }
}

#[cfg(test)]
impl<B: Backend> OrganismGrads<B> {
    pub(crate) fn core_names(&self) -> Vec<&String> {
        self.core.keys().collect()
    }

    pub(crate) fn core_grad(&self, name: &str) -> Option<&ParamTensor<B>> {
        self.core.get(name)
    }

    pub(crate) fn row_grad(&self, row: u32) -> Option<&Vec<f32>> {
        self.rows.get(&row)
    }

    pub(crate) fn row_ids(&self) -> Vec<u32> {
        let mut rows: Vec<u32> = self.rows.keys().copied().collect();
        rows.sort_unstable();
        rows
    }
}
