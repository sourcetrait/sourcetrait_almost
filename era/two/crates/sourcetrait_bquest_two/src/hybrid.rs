//! Stateless Olmo-Hybrid forward on burn: the oracle triangle's
//! third stack (original python / candle lib / burn). No caches, no
//! decode path - every call is one full-sequence replay over all
//! positions, implemented from the pinned checkpoint semantics
//! (parent understood 09), deliberately not translated from the
//! candle implementation. Generic over the backend: CpuBack (ndarray
//! f32) is the reference grade; CudaBack (bf16 default float) is the
//! fast grade. The GDN gate/norm/recurrence region computes in f32
//! on EVERY grade (per-tensor casts; no-op at f32) - the reference
//! stacks' f32-state discipline, which the 1024-deep training
//! backward made mandatory rather than optional.
// The parity gates consume the None-adapter path; the trainer stage
// (cfg train) consumes the injected path plus the exposed structure,
// so non-train builds still need the allow.
#![allow(dead_code)]
use crate::*;

use burn::tensor::backend::Backend;

type FloatTensor<B, const D: usize> = burn::tensor::Tensor<B, D>;

pub(crate) struct HybridModel<B: Backend> {
    pub(crate) embed_rows: Vec<f32>,
    pub(crate) hidden: usize,
    pub(crate) layers: Vec<HybridBlock<B>>,
    pub(crate) final_norm: FloatTensor<B, 1>,
    pub(crate) lm_head_transposed: FloatTensor<B, 2>,
    pub(crate) attn_heads: usize,
    pub(crate) attn_head_dim: usize,
    pub(crate) gdn_heads: usize,
    pub(crate) gdn_key_dim: usize,
    pub(crate) gdn_value_dim: usize,
    pub(crate) eps: f64,
    pub(crate) device: B::Device,
}

pub(crate) enum HybridBlock<B: Backend> {
    Gdn(GdnBlock<B>),
    Attn(AttnBlock<B>),
}

pub(crate) struct GdnBlock<B: Backend> {
    pub(crate) input_norm: FloatTensor<B, 1>,
    pub(crate) post_attention_norm: FloatTensor<B, 1>,
    pub(crate) q_transposed: FloatTensor<B, 2>,
    pub(crate) k_transposed: FloatTensor<B, 2>,
    pub(crate) v_transposed: FloatTensor<B, 2>,
    pub(crate) g_transposed: FloatTensor<B, 2>,
    pub(crate) o_transposed: FloatTensor<B, 2>,
    pub(crate) a_transposed: FloatTensor<B, 2>,
    pub(crate) b_transposed: FloatTensor<B, 2>,
    pub(crate) q_conv_taps: Vec<FloatTensor<B, 2>>,
    pub(crate) k_conv_taps: Vec<FloatTensor<B, 2>>,
    pub(crate) v_conv_taps: Vec<FloatTensor<B, 2>>,
    pub(crate) a_log_row: FloatTensor<B, 2>,
    pub(crate) dt_bias_row: FloatTensor<B, 2>,
    pub(crate) o_norm: FloatTensor<B, 1>,
    pub(crate) gate_transposed: FloatTensor<B, 2>,
    pub(crate) up_transposed: FloatTensor<B, 2>,
    pub(crate) down_transposed: FloatTensor<B, 2>,
}

pub(crate) struct AttnBlock<B: Backend> {
    pub(crate) q_transposed: FloatTensor<B, 2>,
    pub(crate) k_transposed: FloatTensor<B, 2>,
    pub(crate) v_transposed: FloatTensor<B, 2>,
    pub(crate) o_transposed: FloatTensor<B, 2>,
    pub(crate) q_norm: FloatTensor<B, 1>,
    pub(crate) k_norm: FloatTensor<B, 1>,
    pub(crate) post_attention_norm: FloatTensor<B, 1>,
    pub(crate) post_feedforward_norm: FloatTensor<B, 1>,
    pub(crate) gate_transposed: FloatTensor<B, 2>,
    pub(crate) up_transposed: FloatTensor<B, 2>,
    pub(crate) down_transposed: FloatTensor<B, 2>,
}

/// The gated output RMSNorm's eps (the fla FusedRMSNormGated default;
/// deliberately NOT rms_norm_eps).
const O_NORM_EPS: f64 = 1e-5;
/// The qk L2 norm's eps.
const L2_EPS: f64 = 1e-6;

impl<B: Backend> HybridModel<B> {
    pub(crate) fn new(
        config: &HybridCheckpointConfig,
        mut weights: HybridWeights,
        device: B::Device,
    ) -> BquestResult<Self> {
        let (vocab, hidden, embed_rows) =
            weights.take_host_matrix("model.embed_tokens.weight")?;
        snafu::ensure_whatever!(
            vocab == config.vocab_size && hidden == config.hidden_size,
            "embedding shape ({vocab}, {hidden}) does not match the config"
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let block = if config.is_gdn_layer(index)? {
                HybridBlock::Gdn(GdnBlock::new(index, config, &mut weights, &device)?)
            } else {
                HybridBlock::Attn(AttnBlock::new(index, &mut weights, &device)?)
            };
            layers.push(block);
        }

        let final_norm = weights.take_vector::<B>("model.norm.weight", &device)?;
        let lm_head_transposed = if config.tie_word_embeddings {
            let data = burn::tensor::TensorData::new(embed_rows.clone(), [vocab, hidden]);
            FloatTensor::<B, 2>::from_data(data, &device).swap_dims(0, 1)
        } else {
            weights.take_linear_transposed::<B>("lm_head.weight", &device)?
        };

        Ok(Self {
            embed_rows,
            hidden,
            layers,
            final_norm,
            lm_head_transposed,
            attn_heads: config.num_attention_heads,
            attn_head_dim: config.hidden_size / config.num_attention_heads,
            gdn_heads: config.linear_num_key_heads,
            gdn_key_dim: config.linear_key_head_dim,
            gdn_value_dim: config.linear_value_head_dim,
            eps: config.rms_norm_eps,
            device,
        })
    }

    /// Host-side embedding gather for an id sequence (the table is
    /// frozen and host-resident).
    pub(crate) fn embed(&self, ids: &[u32]) -> FloatTensor<B, 2> {
        let n = ids.len();
        let mut gathered = Vec::with_capacity(n * self.hidden);
        for id in ids {
            let start = (*id as usize) * self.hidden;
            gathered.extend_from_slice(&self.embed_rows[start..start + self.hidden]);
        }
        FloatTensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(gathered, [n, self.hidden]),
            &self.device,
        )
    }

    /// Logits for every position as a flat row-major (n, vocab) f32
    /// buffer (row r predicts token r+1 - the C2 dump contract).
    pub(crate) fn forward_all(&self, ids: &[u32]) -> BquestResult<(usize, Vec<f32>)> {
        let n = ids.len();
        snafu::ensure_whatever!(n > 0, "empty id sequence");
        let mut x = self.embed(ids);

        let mask = causal_mask::<B>(n, &self.device);
        for block in &self.layers {
            x = match block {
                HybridBlock::Gdn(block) => block.forward(
                    x,
                    self.gdn_heads,
                    self.gdn_key_dim,
                    self.gdn_value_dim,
                    self.eps,
                    &self.device,
                    None,
                ),
                HybridBlock::Attn(block) => block.forward(
                    x,
                    &mask,
                    self.attn_heads,
                    self.attn_head_dim,
                    self.eps,
                    None,
                ),
            };
        }

        let x = rms_norm(x, &self.final_norm, self.eps);
        let logits = x.matmul(self.lm_head_transposed.clone());
        let data = logits.into_data().convert::<f32>();
        let values = match data.to_vec::<f32>() {
            Ok(values) => values,
            Err(error) => snafu::whatever!("logits extraction failed: {error:?}"),
        };
        Ok((n, values))
    }
}

impl<B: Backend> GdnBlock<B> {
    fn new(
        index: usize,
        config: &HybridCheckpointConfig,
        weights: &mut HybridWeights,
        device: &B::Device,
    ) -> BquestResult<Self> {
        let prefix = format!("model.layers.{index}");
        let kernel = config.linear_conv_kernel_dim;
        Ok(Self {
            input_norm: weights
                .take_vector::<B>(&format!("{prefix}.input_layernorm.weight"), device)?,
            post_attention_norm: weights.take_vector::<B>(
                &format!("{prefix}.post_attention_layernorm.weight"),
                device,
            )?,
            q_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.linear_attn.q_proj.weight"), device)?,
            k_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.linear_attn.k_proj.weight"), device)?,
            v_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.linear_attn.v_proj.weight"), device)?,
            g_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.linear_attn.g_proj.weight"), device)?,
            o_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.linear_attn.o_proj.weight"), device)?,
            a_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.linear_attn.a_proj.weight"), device)?,
            b_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.linear_attn.b_proj.weight"), device)?,
            q_conv_taps: weights.take_conv_taps::<B>(
                &format!("{prefix}.linear_attn.q_conv1d.weight"),
                kernel,
                device,
            )?,
            k_conv_taps: weights.take_conv_taps::<B>(
                &format!("{prefix}.linear_attn.k_conv1d.weight"),
                kernel,
                device,
            )?,
            v_conv_taps: weights.take_conv_taps::<B>(
                &format!("{prefix}.linear_attn.v_conv1d.weight"),
                kernel,
                device,
            )?,
            a_log_row: weights.take_row::<B>(&format!("{prefix}.linear_attn.A_log"), device)?,
            dt_bias_row: weights.take_row::<B>(&format!("{prefix}.linear_attn.dt_bias"), device)?,
            o_norm: weights.take_vector::<B>(&format!("{prefix}.linear_attn.o_norm.weight"), device)?,
            gate_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.mlp.gate_proj.weight"), device)?,
            up_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.mlp.up_proj.weight"), device)?,
            down_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.mlp.down_proj.weight"), device)?,
        })
    }

    /// Fully PRE-norm block: x + mixer(input_norm(x)), then
    /// h + mlp(post_attention_norm(h)). Adapters (when present) add
    /// their contributions on the direction-3 readout surface only -
    /// q/q_conv/g/o + MLP; the k/v/a/b state path never adapts - and
    /// the None path is op-identical to the frozen oracle.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        x: FloatTensor<B, 2>,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        eps: f64,
        device: &B::Device,
        adapters: Option<&GdnAdapters<B>>,
    ) -> FloatTensor<B, 2> {
        let n = x.dims()[0];
        let residual = x.clone();
        let normed = rms_norm(x, &self.input_norm, eps);

        // Projections, each through its causal depthwise conv + silu
        // (the gate projection takes no conv).
        let mut q_rows = normed.clone().matmul(self.q_transposed.clone());
        if let Some(adapters) = adapters {
            q_rows = q_rows + adapters.q.contribution(normed.clone());
        }
        let q = match adapters {
            Some(adapters) => causal_conv_silu(
                q_rows,
                &adapters.q_conv.effective_taps(&self.q_conv_taps),
                device,
            ),
            None => causal_conv_silu(q_rows, &self.q_conv_taps, device),
        };
        let k = causal_conv_silu(normed.clone().matmul(self.k_transposed.clone()), &self.k_conv_taps, device);
        let v = causal_conv_silu(normed.clone().matmul(self.v_transposed.clone()), &self.v_conv_taps, device);
        let mut gate_rows = normed.clone().matmul(self.g_transposed.clone());
        if let Some(adapters) = adapters {
            gate_rows = gate_rows + adapters.g.contribution(normed.clone());
        }

        // The f32 recurrence discipline (the reference stacks upcast
        // q/k/v/beta/g AND the state to f32 regardless of model
        // dtype; the paper's repeated-downcast warning is the
        // mechanism, and at bf16 the 1024-deep BACKWARD through the
        // recurrence NaNs - live-fired): the gate math, qk norms,
        // recurrence, and gated output norm compute in f32; the
        // mixer output casts back for o_proj. No-op on f32 grades.
        let compute = burn::tensor::FloatDType::F32;
        let q = q.cast(compute);
        let k = k.cast(compute);
        let v = v.cast(compute);
        let gate_rows = gate_rows.cast(compute);

        // Gating scalars: g = -exp(A_log) * softplus(a + dt_bias);
        // beta = 2 * sigmoid(b) (the negative-eigenvalue variant).
        let a_rows = normed.clone().matmul(self.a_transposed.clone()).cast(compute);
        let b_rows = normed.matmul(self.b_transposed.clone()).cast(compute);
        let decay = self
            .a_log_row
            .clone()
            .cast(compute)
            .exp()
            .expand([n, heads])
            .neg()
            .mul(softplus(
                a_rows + self.dt_bias_row.clone().cast(compute).expand([n, heads]),
            ));
        let beta = burn::tensor::activation::sigmoid(b_rows).mul_scalar(2.0);

        // Heads: q/k (n, heads, key_dim) L2-normed, q pre-scaled;
        // v (n, heads, value_dim).
        let q = l2_norm_last(q.reshape([n, heads, key_dim])).mul_scalar((key_dim as f64).powf(-0.5));
        let k = l2_norm_last(k.reshape([n, heads, key_dim]));
        let v = v.reshape([n, heads, value_dim]);

        // The gated-delta recurrence, sequential per token (the
        // correctness grade): decay the state BEFORE the delta
        // correction.
        let mut state =
            FloatTensor::<B, 3>::zeros([heads, key_dim, value_dim], device).cast(compute);
        let mut outputs: Vec<FloatTensor<B, 3>> = Vec::with_capacity(n);
        for t in 0..n {
            let q_t = q.clone().narrow(0, t, 1).reshape([heads, key_dim, 1]);
            let k_t = k.clone().narrow(0, t, 1).reshape([heads, key_dim, 1]);
            let v_t = v.clone().narrow(0, t, 1).reshape([heads, 1, value_dim]);
            let decay_t = decay
                .clone()
                .narrow(0, t, 1)
                .reshape([heads, 1, 1])
                .exp()
                .expand([heads, key_dim, value_dim]);
            let beta_t = beta
                .clone()
                .narrow(0, t, 1)
                .reshape([heads, 1, 1])
                .expand([heads, 1, value_dim]);

            state = state * decay_t;
            let kv_mem = (state.clone() * k_t.clone().expand([heads, key_dim, value_dim]))
                .sum_dim(1);
            let delta = (v_t - kv_mem) * beta_t;
            state = state
                + k_t.clone().expand([heads, key_dim, value_dim])
                    * delta.expand([heads, key_dim, value_dim]);
            let y_t = (state.clone() * q_t.expand([heads, key_dim, value_dim])).sum_dim(1);
            outputs.push(y_t);
        }
        let y = burn::tensor::Tensor::cat(outputs, 1).swap_dims(0, 1);

        // Gated output RMSNorm over the value-head dim (eps 1e-5),
        // norm THEN gate; gate = silu(g_proj(x)) - all in f32, then
        // one cast back to the projection dtype.
        let o_norm_compute = self.o_norm.clone().cast(compute);
        let y = rms_norm_last(y, &o_norm_compute, O_NORM_EPS);
        let gate = burn::tensor::activation::silu(gate_rows).reshape([n, heads, value_dim]);
        let gated_flat = (y * gate)
            .reshape([n, heads * value_dim])
            .cast(self.o_transposed.dtype());
        let mut mixed = gated_flat.clone().matmul(self.o_transposed.clone());
        if let Some(adapters) = adapters {
            mixed = mixed + adapters.o.contribution(gated_flat);
        }
        let x = residual + mixed;

        let residual = x.clone();
        let normed = rms_norm(x, &self.post_attention_norm, eps);
        let mut gate_lin = normed.clone().matmul(self.gate_transposed.clone());
        let mut up_lin = normed.clone().matmul(self.up_transposed.clone());
        if let Some(adapters) = adapters {
            gate_lin = gate_lin + adapters.gate.contribution(normed.clone());
            up_lin = up_lin + adapters.up.contribution(normed);
        }
        let activated = burn::tensor::activation::silu(gate_lin) * up_lin;
        let mut feedforward = activated.clone().matmul(self.down_transposed.clone());
        if let Some(adapters) = adapters {
            feedforward = feedforward + adapters.down.contribution(activated);
        }
        residual + feedforward
    }
}

impl<B: Backend> AttnBlock<B> {
    fn new(
        index: usize,
        weights: &mut HybridWeights,
        device: &B::Device,
    ) -> BquestResult<Self> {
        let prefix = format!("model.layers.{index}");
        Ok(Self {
            q_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.self_attn.q_proj.weight"), device)?,
            k_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.self_attn.k_proj.weight"), device)?,
            v_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.self_attn.v_proj.weight"), device)?,
            o_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.self_attn.o_proj.weight"), device)?,
            q_norm: weights.take_vector::<B>(&format!("{prefix}.self_attn.q_norm.weight"), device)?,
            k_norm: weights.take_vector::<B>(&format!("{prefix}.self_attn.k_norm.weight"), device)?,
            post_attention_norm: weights.take_vector::<B>(
                &format!("{prefix}.post_attention_layernorm.weight"),
                device,
            )?,
            post_feedforward_norm: weights.take_vector::<B>(
                &format!("{prefix}.post_feedforward_layernorm.weight"),
                device,
            )?,
            gate_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.mlp.gate_proj.weight"), device)?,
            up_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.mlp.up_proj.weight"), device)?,
            down_transposed: weights
                .take_linear_transposed::<B>(&format!("{prefix}.mlp.down_proj.weight"), device)?,
        })
    }

    /// Fully POST-norm block (raw hidden into attention, no input
    /// norm): x + norm(attn(x)), then h + norm(mlp(h)). NoPE: no
    /// rotation anywhere; positions exist only in the causal mask.
    /// Adapters (when present) ride q/o + MLP only (k/v never); the
    /// None path is op-identical to the frozen oracle.
    pub(crate) fn forward(
        &self,
        x: FloatTensor<B, 2>,
        mask: &FloatTensor<B, 2>,
        heads: usize,
        head_dim: usize,
        eps: f64,
        adapters: Option<&AttnAdapters<B>>,
    ) -> FloatTensor<B, 2> {
        let n = x.dims()[0];
        let hidden = heads * head_dim;
        let residual = x.clone();

        // Full-projection-width q/k RMSNorm before the head reshape.
        let mut q_rows = x.clone().matmul(self.q_transposed.clone());
        if let Some(adapters) = adapters {
            q_rows = q_rows + adapters.q.contribution(x.clone());
        }
        let q = rms_norm(q_rows, &self.q_norm, eps);
        let k = rms_norm(x.clone().matmul(self.k_transposed.clone()), &self.k_norm, eps);
        let v = x.matmul(self.v_transposed.clone());

        let q = q.reshape([n, heads, head_dim]).swap_dims(0, 1);
        let k = k.reshape([n, heads, head_dim]).swap_dims(0, 1);
        let v = v.reshape([n, heads, head_dim]).swap_dims(0, 1);

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale)
            + mask.clone().unsqueeze::<3>().expand([heads, n, n]);
        let attention = burn::tensor::activation::softmax(scores, 2);
        let context = attention.matmul(v).swap_dims(0, 1).reshape([n, hidden]);
        let mut projected = context.clone().matmul(self.o_transposed.clone());
        if let Some(adapters) = adapters {
            projected = projected + adapters.o.contribution(context);
        }
        let x = residual + rms_norm(projected, &self.post_attention_norm, eps);

        let residual = x.clone();
        let mut gate_lin = x.clone().matmul(self.gate_transposed.clone());
        let mut up_lin = x.clone().matmul(self.up_transposed.clone());
        if let Some(adapters) = adapters {
            gate_lin = gate_lin + adapters.gate.contribution(x.clone());
            up_lin = up_lin + adapters.up.contribution(x);
        }
        let activated = burn::tensor::activation::silu(gate_lin) * up_lin;
        let mut feedforward = activated.clone().matmul(self.down_transposed.clone());
        if let Some(adapters) = adapters {
            feedforward = feedforward + adapters.down.contribution(activated);
        }
        residual + rms_norm(feedforward, &self.post_feedforward_norm, eps)
    }
}

/// Additive 0/-inf causal mask (n, n).
pub(crate) fn causal_mask<B: Backend>(n: usize, device: &B::Device) -> FloatTensor<B, 2> {
    let mut values = vec![0f32; n * n];
    for query in 0..n {
        for key in (query + 1)..n {
            values[query * n + key] = f32::NEG_INFINITY;
        }
    }
    FloatTensor::<B, 2>::from_data(burn::tensor::TensorData::new(values, [n, n]), device)
}

/// HF RMSNorm over the last dim of a rank-2 tensor:
/// x * rsqrt(mean(x^2) + eps) * weight.
pub(crate) fn rms_norm<B: Backend>(
    x: FloatTensor<B, 2>,
    weight: &FloatTensor<B, 1>,
    eps: f64,
) -> FloatTensor<B, 2> {
    let [n, width] = x.dims();
    let mean_square = (x.clone() * x.clone()).mean_dim(1);
    let scale = (mean_square + eps).sqrt().recip().expand([n, width]);
    x * scale * weight.clone().unsqueeze::<2>().expand([n, width])
}

/// RMSNorm over the last dim of a rank-3 tensor (the gated output
/// norm's per-head form).
fn rms_norm_last<B: Backend>(
    x: FloatTensor<B, 3>,
    weight: &FloatTensor<B, 1>,
    eps: f64,
) -> FloatTensor<B, 3> {
    let [n, heads, width] = x.dims();
    let mean_square = (x.clone() * x.clone()).mean_dim(2);
    let scale = (mean_square + eps).sqrt().recip().expand([n, heads, width]);
    x * scale
        * weight
            .clone()
            .unsqueeze::<2>()
            .unsqueeze::<3>()
            .expand([n, heads, width])
}

/// SUM-based L2 norm over the last dim (x / sqrt(sum(x^2) + eps) -
/// deliberately not RMSNorm's mean).
fn l2_norm_last<B: Backend>(x: FloatTensor<B, 3>) -> FloatTensor<B, 3> {
    let [n, heads, width] = x.dims();
    let sum_square = (x.clone() * x.clone()).sum_dim(2);
    let scale = (sum_square + L2_EPS).sqrt().recip().expand([n, heads, width]);
    x * scale
}

/// softplus(x) = ln(1 + exp(x)), computed in the overflow-stable
/// form max(x, 0) + ln(1 + exp(-|x|)). The naive form's exp(x)
/// overflows to inf on large gating inputs: the FORWARD survives
/// (softplus saturates, decay = exp(-inf) reads zero) but the
/// BACKWARD computes inf/inf = NaN - the reference stacks all guard
/// this (the fla/vllm gating kernels' softplus guard). exp(-|x|)
/// never overflows and the gradient is sigmoid(x) everywhere.
fn softplus<B: Backend>(x: FloatTensor<B, 2>) -> FloatTensor<B, 2> {
    let positive_part = burn::tensor::activation::relu(x.clone());
    let negative_magnitude = x.abs().neg();
    positive_part + negative_magnitude.exp().add_scalar(1.0).log()
}

/// The causal depthwise conv (kernel k over time, per channel) + silu:
/// tap k-1 multiplies the current token, earlier taps reach back in
/// time, missing history is zero (the stateless left-pad form).
fn causal_conv_silu<B: Backend>(
    x: FloatTensor<B, 2>,
    taps: &[FloatTensor<B, 2>],
    device: &B::Device,
) -> FloatTensor<B, 2> {
    let [n, channels] = x.dims();
    let kernel = taps.len();
    let mut accumulated = FloatTensor::<B, 2>::zeros([n, channels], device);
    for (tap, weight_row) in taps.iter().enumerate() {
        let shift = kernel - 1 - tap;
        let weighted = if shift == 0 {
            x.clone() * weight_row.clone().expand([n, channels])
        } else if shift >= n {
            continue;
        } else {
            let kept = n - shift;
            let shifted = burn::tensor::Tensor::cat(
                vec![
                    FloatTensor::<B, 2>::zeros([shift, channels], device),
                    x.clone().narrow(0, 0, kept),
                ],
                0,
            );
            shifted * weight_row.clone().expand([n, channels])
        };
        accumulated = accumulated + weighted;
    }
    burn::tensor::activation::silu(accumulated)
}
