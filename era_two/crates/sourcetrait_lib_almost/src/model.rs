//! The stateless era-two model: one full-sequence forward over the 32
//! hybrid layers (24 pre-norm GDN + 8 post-norm NoPE attention),
//! returning all-position logits.
//!
//! D2 scope: no cache machinery, no cross-call state - the GDN
//! recurrence runs per token inside the forward. The recurrence and
//! its gates ride f32 regardless of model dtype (the pinned upstream
//! discipline); attention softmax computes in f32 and casts back.
use crate::*;

/// SwiGLU MLP, present on every layer: down(silu(gate(x)) * up(x)).
struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    fn new(
        hidden: usize,
        intermediate: usize,
        vb: candle_nn::VarBuilder,
    ) -> LibAlmostResult<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up_proj: candle_nn::linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let gated = self.gate_proj.forward(x)?.silu()?;
        Ok(self.down_proj.forward(&gated.mul(&self.up_proj.forward(x)?)?)?)
    }
}

/// A linear-attention (GDN) decoder layer - fully PRE-norm:
/// h = x + mixer(input_layernorm(x)), then
/// out = h + mlp(post_attention_layernorm(h)).
struct GdnLayer {
    input_layernorm: candle_core::Tensor,
    post_attention_layernorm: candle_core::Tensor,
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    a_proj: candle_nn::Linear,
    b_proj: candle_nn::Linear,
    g_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_conv: candle_core::Tensor,
    k_conv: candle_core::Tensor,
    v_conv: candle_core::Tensor,
    a_log: candle_core::Tensor,
    dt_bias: candle_core::Tensor,
    o_norm: candle_core::Tensor,
    mlp: Mlp,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    allow_neg_eigval: bool,
    rms_eps: f64,
}

impl GdnLayer {
    fn new(config: &OlmoHybridConfig, vb: candle_nn::VarBuilder) -> LibAlmostResult<Self> {
        let hidden = config.hidden_size;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let kernel = config.linear_conv_kernel_dim;
        let heads = config.linear_num_value_heads;
        let vb_attn = vb.pp("linear_attn");
        Ok(Self {
            input_layernorm: vb.pp("input_layernorm").get(hidden, "weight")?,
            post_attention_layernorm: vb.pp("post_attention_layernorm").get(hidden, "weight")?,
            q_proj: candle_nn::linear_no_bias(hidden, key_dim, vb_attn.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(hidden, key_dim, vb_attn.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(hidden, value_dim, vb_attn.pp("v_proj"))?,
            a_proj: candle_nn::linear_no_bias(hidden, heads, vb_attn.pp("a_proj"))?,
            b_proj: candle_nn::linear_no_bias(hidden, heads, vb_attn.pp("b_proj"))?,
            g_proj: candle_nn::linear_no_bias(hidden, value_dim, vb_attn.pp("g_proj"))?,
            o_proj: candle_nn::linear_no_bias(value_dim, hidden, vb_attn.pp("o_proj"))?,
            q_conv: vb_attn.pp("q_conv1d").get((key_dim, 1, kernel), "weight")?,
            k_conv: vb_attn.pp("k_conv1d").get((key_dim, 1, kernel), "weight")?,
            v_conv: vb_attn.pp("v_conv1d").get((value_dim, 1, kernel), "weight")?,
            a_log: vb_attn.get(heads, "A_log")?,
            dt_bias: vb_attn.get(heads, "dt_bias")?,
            o_norm: vb_attn.pp("o_norm").get(config.linear_value_head_dim, "weight")?,
            mlp: Mlp::new(hidden, config.intermediate_size, vb.pp("mlp"))?,
            num_heads: heads,
            head_k_dim: config.linear_key_head_dim,
            head_v_dim: config.linear_value_head_dim,
            allow_neg_eigval: config.linear_allow_neg_eigval,
            rms_eps: config.rms_norm_eps,
        })
    }

    fn forward(&self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let mixed = self.mixer(&norms::rms_norm(x, &self.input_layernorm, self.rms_eps)?)?;
        let h = (x + mixed)?;
        let mlp_in = norms::rms_norm(&h, &self.post_attention_layernorm, self.rms_eps)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        Ok((h + mlp_out)?)
    }

    /// The GDN mixer over a whole sequence, sequential form: conv'd
    /// projections -> l2-normed heads -> per-token gated-delta
    /// recurrence -> gated output norm -> o_proj.
    fn mixer(&self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let (seq_len, _) = x.dims2()?;
        let dtype = x.dtype();
        let q = gdn::causal_conv_silu(&self.q_proj.forward(x)?, &self.q_conv)?;
        let k = gdn::causal_conv_silu(&self.k_proj.forward(x)?, &self.k_conv)?;
        let v = gdn::causal_conv_silu(&self.v_proj.forward(x)?, &self.v_conv)?;

        // The recurrence's f32 discipline: q/k/v and both gates upcast,
        // the state lives f32, the output casts back to the model dtype.
        let q = q
            .reshape((seq_len, self.num_heads, self.head_k_dim))?
            .to_dtype(candle_core::DType::F32)?;
        let k = k
            .reshape((seq_len, self.num_heads, self.head_k_dim))?
            .to_dtype(candle_core::DType::F32)?;
        let v = v
            .reshape((seq_len, self.num_heads, self.head_v_dim))?
            .to_dtype(candle_core::DType::F32)?;
        let q = gdn::l2_norm(&q)?;
        let k = gdn::l2_norm(&k)?;
        let q = (q * (self.head_k_dim as f64).powf(-0.5))?;

        let (g, beta) = gdn::gdn_gates(
            &self.a_proj.forward(x)?,
            &self.b_proj.forward(x)?,
            &self.a_log,
            &self.dt_bias,
            self.allow_neg_eigval,
        )?;
        let decay = g.exp()?;

        let mut state = candle_core::Tensor::zeros(
            (self.num_heads, self.head_k_dim, self.head_v_dim),
            candle_core::DType::F32,
            x.device(),
        )?;
        let mut rows = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let q_t = q.narrow(0, position, 1)?.squeeze(0)?;
            let k_t = k.narrow(0, position, 1)?.squeeze(0)?;
            let v_t = v.narrow(0, position, 1)?.squeeze(0)?;
            let decay_t = decay.narrow(0, position, 1)?.squeeze(0)?;
            let beta_t = beta.narrow(0, position, 1)?.squeeze(0)?;
            let (next_state, y_t) = gdn::recurrent_step(&state, &q_t, &k_t, &v_t, &decay_t, &beta_t)?;
            state = next_state;
            rows.push(y_t);
        }
        let y = candle_core::Tensor::stack(&rows, 0)?.to_dtype(dtype)?;

        let gate = self
            .g_proj
            .forward(x)?
            .reshape((seq_len, self.num_heads, self.head_v_dim))?;
        let y = norms::rms_norm_gated(&y, &gate, &self.o_norm, gdn::O_NORM_EPS)?;
        Ok(self
            .o_proj
            .forward(&y.reshape((seq_len, self.num_heads * self.head_v_dim))?)?)
    }
}

/// A full-attention decoder layer - fully POST-norm, olmo2/3 style
/// (raw hidden enters attention, no input norm):
/// h = x + post_attention_layernorm(attn(x)), then
/// out = h + post_feedforward_layernorm(mlp(h)).
struct AttnLayer {
    post_attention_layernorm: candle_core::Tensor,
    post_feedforward_layernorm: candle_core::Tensor,
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: candle_core::Tensor,
    k_norm: candle_core::Tensor,
    mlp: Mlp,
    num_heads: usize,
    head_dim: usize,
    rms_eps: f64,
}

impl AttnLayer {
    fn new(config: &OlmoHybridConfig, vb: candle_nn::VarBuilder) -> LibAlmostResult<Self> {
        let hidden = config.hidden_size;
        let width = config.num_attention_heads * config.head_dim();
        let vb_attn = vb.pp("self_attn");
        Ok(Self {
            post_attention_layernorm: vb.pp("post_attention_layernorm").get(hidden, "weight")?,
            post_feedforward_layernorm: vb
                .pp("post_feedforward_layernorm")
                .get(hidden, "weight")?,
            q_proj: candle_nn::linear_no_bias(hidden, width, vb_attn.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(hidden, width, vb_attn.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(hidden, width, vb_attn.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(width, hidden, vb_attn.pp("o_proj"))?,
            q_norm: vb_attn.pp("q_norm").get(width, "weight")?,
            k_norm: vb_attn.pp("k_norm").get(width, "weight")?,
            mlp: Mlp::new(hidden, config.intermediate_size, vb.pp("mlp"))?,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            rms_eps: config.rms_norm_eps,
        })
    }

    fn forward(
        &self,
        x: &candle_core::Tensor,
        mask: &candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let attn_out = norms::rms_norm(
            &self.attend(x, mask)?,
            &self.post_attention_layernorm,
            self.rms_eps,
        )?;
        let h = (x + attn_out)?;
        let mlp_out = norms::rms_norm(
            &self.mlp.forward(&h)?,
            &self.post_feedforward_layernorm,
            self.rms_eps,
        )?;
        Ok((h + mlp_out)?)
    }

    /// Eager MHA, NoPE: full-projection-width q/k RMSNorm before the
    /// head reshape, no rotation anywhere, softmax in f32 cast back.
    fn attend(
        &self,
        x: &candle_core::Tensor,
        mask: &candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let (seq_len, _) = x.dims2()?;
        let dtype = x.dtype();
        let q = norms::rms_norm(&self.q_proj.forward(x)?, &self.q_norm, self.rms_eps)?;
        let k = norms::rms_norm(&self.k_proj.forward(x)?, &self.k_norm, self.rms_eps)?;
        let v = self.v_proj.forward(x)?;
        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = k
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = v
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let scores = (q.matmul(&k.transpose(1, 2)?)? * (self.head_dim as f64).powf(-0.5))?;
        let scores = scores.broadcast_add(&mask.unsqueeze(0)?)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.to_dtype(candle_core::DType::F32)?)?
            .to_dtype(dtype)?;
        let out = probs
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq_len, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }
}

enum Layer {
    Gdn(GdnLayer),
    Attn(AttnLayer),
}

/// The era-two hybrid model in its stateless form: forward_all over
/// one unbatched id sequence.
pub struct OlmoHybrid {
    embed_tokens: candle_core::Tensor,
    layers: Vec<Layer>,
    norm: candle_core::Tensor,
    lm_head: candle_nn::Linear,
    rms_eps: f64,
}

impl OlmoHybrid {
    pub fn new(config: &OlmoHybridConfig, vb: candle_nn::VarBuilder) -> LibAlmostResult<Self> {
        config.validate()?;
        let vb_model = vb.pp("model");
        let embed_tokens = vb_model
            .pp("embed_tokens")
            .get((config.vocab_size, config.hidden_size), "weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let vb_layer = vb_model.pp(format!("layers.{layer_idx}"));
            layers.push(match config.layer_kind(layer_idx) {
                LayerKind::LinearAttention => Layer::Gdn(GdnLayer::new(config, vb_layer)?),
                LayerKind::FullAttention => Layer::Attn(AttnLayer::new(config, vb_layer)?),
            });
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: vb_model.pp("norm").get(config.hidden_size, "weight")?,
            lm_head: candle_nn::linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?,
            rms_eps: config.rms_norm_eps,
        })
    }

    /// All-position logits for one unbatched id sequence: [T] u32 in,
    /// [T, vocab] out in the model dtype (row r predicts token r+1).
    pub fn forward_all(
        &self,
        input_ids: &candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let seq_len = input_ids.dim(0)?;
        let mut hidden = self.embed_tokens.index_select(input_ids, 0)?;
        let mask = causal_mask(seq_len, hidden.dtype(), hidden.device())?;
        for layer in &self.layers {
            hidden = match layer {
                Layer::Gdn(layer) => layer.forward(&hidden)?,
                Layer::Attn(layer) => layer.forward(&hidden, &mask)?,
            };
        }
        let hidden = norms::rms_norm(&hidden, &self.norm, self.rms_eps)?;
        Ok(self.lm_head.forward(&hidden)?)
    }
}

/// Additive causal mask [T, T]: 0 on and below the diagonal, -inf
/// above (exp saturates to exactly 0 either way upstream renders it).
fn causal_mask(
    seq_len: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> LibAlmostResult<candle_core::Tensor> {
    let mut values = vec![0f32; seq_len * seq_len];
    for (row, chunk) in values.chunks_exact_mut(seq_len).enumerate() {
        for value in chunk.iter_mut().skip(row + 1) {
            *value = f32::NEG_INFINITY;
        }
    }
    Ok(
        candle_core::Tensor::from_vec(values, (seq_len, seq_len), device)?
            .to_dtype(dtype)?,
    )
}
