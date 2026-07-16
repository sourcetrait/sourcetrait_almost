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
/// Carries the D3 cross-chunk cache: the f32 recurrent state plus the
/// three raw pre-conv tails (kernel-1 rows each, stored f32).
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
    /// The three depthwise conv weights concatenated along channels
    /// ([2*key + value, 1, kernel]) - one batched conv pass per
    /// forward instead of three; per-channel math is bit-identical.
    conv_weight: candle_core::Tensor,
    a_log: candle_core::Tensor,
    dt_bias: candle_core::Tensor,
    o_norm: candle_core::Tensor,
    mlp: Mlp,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_width: usize,
    value_width: usize,
    allow_neg_eigval: bool,
    rms_eps: f64,
    state: candle_core::Tensor,
    conv_tail: candle_core::Tensor,
}

impl GdnLayer {
    fn new(config: &OlmoHybridConfig, vb: candle_nn::VarBuilder) -> LibAlmostResult<Self> {
        let hidden = config.hidden_size;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let kernel = config.linear_conv_kernel_dim;
        let heads = config.linear_num_value_heads;
        let device = vb.device().clone();
        let f32_zeros = |shape: (usize, usize)| {
            candle_core::Tensor::zeros(shape, candle_core::DType::F32, &device)
        };
        let vb_attn = vb.pp("linear_attn");
        Ok(Self {
            state: candle_core::Tensor::zeros(
                (heads, config.linear_key_head_dim, config.linear_value_head_dim),
                candle_core::DType::F32,
                &device,
            )?,
            conv_tail: f32_zeros((kernel - 1, key_dim + key_dim + value_dim))?,
            input_layernorm: vb.pp("input_layernorm").get(hidden, "weight")?,
            post_attention_layernorm: vb.pp("post_attention_layernorm").get(hidden, "weight")?,
            q_proj: candle_nn::linear_no_bias(hidden, key_dim, vb_attn.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(hidden, key_dim, vb_attn.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(hidden, value_dim, vb_attn.pp("v_proj"))?,
            a_proj: candle_nn::linear_no_bias(hidden, heads, vb_attn.pp("a_proj"))?,
            b_proj: candle_nn::linear_no_bias(hidden, heads, vb_attn.pp("b_proj"))?,
            g_proj: candle_nn::linear_no_bias(hidden, value_dim, vb_attn.pp("g_proj"))?,
            o_proj: candle_nn::linear_no_bias(value_dim, hidden, vb_attn.pp("o_proj"))?,
            conv_weight: candle_core::Tensor::cat(
                &[
                    &vb_attn.pp("q_conv1d").get((key_dim, 1, kernel), "weight")?,
                    &vb_attn.pp("k_conv1d").get((key_dim, 1, kernel), "weight")?,
                    &vb_attn.pp("v_conv1d").get((value_dim, 1, kernel), "weight")?,
                ],
                0,
            )?,
            a_log: vb_attn.get(heads, "A_log")?,
            dt_bias: vb_attn.get(heads, "dt_bias")?,
            o_norm: vb_attn.pp("o_norm").get(config.linear_value_head_dim, "weight")?,
            mlp: Mlp::new(hidden, config.intermediate_size, vb.pp("mlp"))?,
            num_heads: heads,
            head_k_dim: config.linear_key_head_dim,
            head_v_dim: config.linear_value_head_dim,
            key_width: key_dim,
            value_width: value_dim,
            allow_neg_eigval: config.linear_allow_neg_eigval,
            rms_eps: config.rms_norm_eps,
        })
    }

    /// The batched conv input: q/k/v projections concatenated along
    /// channels (packed - cat on a non-zero dim is a transposed view).
    fn projected(&self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        Ok(candle_core::Tensor::cat(
            &[
                &self.q_proj.forward(x)?,
                &self.k_proj.forward(x)?,
                &self.v_proj.forward(x)?,
            ],
            1,
        )?
        .contiguous()?)
    }

    /// Split a batched conv output back into contiguous q/k/v spans.
    fn split_conv(
        &self,
        conv_out: &candle_core::Tensor,
    ) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)> {
        Ok((
            conv_out.narrow(1, 0, self.key_width)?.contiguous()?,
            conv_out.narrow(1, self.key_width, self.key_width)?.contiguous()?,
            conv_out
                .narrow(1, 2 * self.key_width, self.value_width)?
                .contiguous()?,
        ))
    }

    fn forward(&self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let mixed = self.mixer_stateless(&norms::rms_norm(x, &self.input_layernorm, self.rms_eps)?)?;
        self.close_halves(x, mixed)
    }

    fn forward_chunk(&mut self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let mixed = self.mixer_carried(&norms::rms_norm(x, &self.input_layernorm, self.rms_eps)?)?;
        self.close_halves(x, mixed)
    }

    /// The pre-norm block's residual adds shared by both paths.
    fn close_halves(
        &self,
        x: &candle_core::Tensor,
        mixed: candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let h = (x + mixed)?;
        let mlp_in = norms::rms_norm(&h, &self.post_attention_layernorm, self.rms_eps)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        Ok((h + mlp_out)?)
    }

    /// Post-conv prep shared by both mixer paths: head reshape, the
    /// recurrence's f32 upcast discipline, l2 norms, the q scale, and
    /// the gating scalars. Returns (q, k, v, g, beta) rule-ready.
    #[allow(clippy::type_complexity)]
    fn heads_and_gates(
        &self,
        q: candle_core::Tensor,
        k: candle_core::Tensor,
        v: candle_core::Tensor,
        x: &candle_core::Tensor,
        seq_len: usize,
    ) -> LibAlmostResult<(
        candle_core::Tensor,
        candle_core::Tensor,
        candle_core::Tensor,
        candle_core::Tensor,
        candle_core::Tensor,
    )> {
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
        Ok((q, k, v, g, beta))
    }

    /// The gated output norm + o_proj tail shared by both mixer paths.
    fn finish_mixer(
        &self,
        y: candle_core::Tensor,
        x: &candle_core::Tensor,
        seq_len: usize,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let y = y.to_dtype(x.dtype())?;
        let gate = self
            .g_proj
            .forward(x)?
            .reshape((seq_len, self.num_heads, self.head_v_dim))?;
        let y = norms::rms_norm_gated(&y, &gate, &self.o_norm, gdn::O_NORM_EPS)?;
        Ok(self
            .o_proj
            .forward(&y.reshape((seq_len, self.num_heads * self.head_v_dim))?)?)
    }

    /// The GDN mixer, STATELESS sequential form (the D2 parity
    /// instrument): zero-seeded per-token recurrence, no state kept.
    fn mixer_stateless(&self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let (seq_len, _) = x.dims2()?;
        let conv_out = gdn::causal_conv_silu(&self.projected(x)?, &self.conv_weight)?;
        let (q, k, v) = self.split_conv(&conv_out)?;
        let (q, k, v, g, beta) = self.heads_and_gates(q, k, v, x, seq_len)?;
        let decay = g.exp()?;
        let mut state = self.state.zeros_like()?;
        let mut rows = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let q_t = q.narrow(0, position, 1)?.squeeze(0)?;
            let k_t = k.narrow(0, position, 1)?.squeeze(0)?;
            let v_t = v.narrow(0, position, 1)?.squeeze(0)?;
            let decay_t = decay.narrow(0, position, 1)?.squeeze(0)?;
            let beta_t = beta.narrow(0, position, 1)?.squeeze(0)?;
            let (next_state, y_t) =
                gdn::recurrent_step(&state, &q_t, &k_t, &v_t, &decay_t, &beta_t)?;
            state = next_state;
            rows.push(y_t);
        }
        let y = candle_core::Tensor::stack(&rows, 0)?;
        self.finish_mixer(y, x, seq_len)
    }

    /// The GDN mixer, CARRIED form (the D3 engine path): conv tails
    /// prepend the chunk, the persisted state seeds the rule, and both
    /// advance. Decode (seq 1) rides the recurrent step; larger chunks
    /// ride the chunked rule - the upstream's own path split.
    fn mixer_carried(&mut self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let (seq_len, _) = x.dims2()?;
        let (conv_out, tail) =
            gdn::conv_with_tail(&self.projected(x)?, &self.conv_weight, &self.conv_tail)?;
        self.conv_tail = tail;
        let (q, k, v) = self.split_conv(&conv_out)?;
        let (q, k, v, g, beta) = self.heads_and_gates(q, k, v, x, seq_len)?;
        let y = if seq_len == 1 {
            let (next_state, y_t) = gdn::recurrent_step(
                &self.state,
                &q.squeeze(0)?,
                &k.squeeze(0)?,
                &v.squeeze(0)?,
                &g.exp()?.squeeze(0)?,
                &beta.squeeze(0)?,
            )?;
            self.state = next_state;
            y_t.unsqueeze(0)?
        } else {
            let (out, next_state) = gdn::chunk_rule(&q, &k, &v, &g, &beta, &self.state)?;
            self.state = next_state;
            out
        };
        self.finish_mixer(y, x, seq_len)
    }

    fn clear_cache(&mut self) -> LibAlmostResult<()> {
        self.state = self.state.zeros_like()?;
        self.conv_tail = self.conv_tail.zeros_like()?;
        Ok(())
    }
}

/// A full-attention decoder layer - fully POST-norm, olmo2/3 style
/// (raw hidden enters attention, no input norm):
/// h = x + post_attention_layernorm(attn(x)), then
/// out = h + post_feedforward_layernorm(mlp(h)).
/// The reserve grain for the carried KV buffers: growth happens in
/// KV_RESERVE_STEP-token steps (one copy per grow, amortized), reads
/// narrow to the live length, and clear keeps the capacity - the
/// cat-per-append copy2d tax (12.6% of 32K prefill GPU time) dies.
const KV_RESERVE_STEP: usize = 1024;

/// Carried attention KV: capacity-reserved [heads, capacity,
/// head_dim] buffers plus the live length.
struct AttnKv {
    k: candle_core::Tensor,
    v: candle_core::Tensor,
    len: usize,
}

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
    kv: Option<AttnKv>,
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
            kv: None,
        })
    }

    fn forward(
        &self,
        x: &candle_core::Tensor,
        mask: &candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let (q, k, v) = self.qkv(x)?;
        let attended = self.attend(&q, &k, &v, Some(mask))?;
        self.close_halves(x, attended)
    }

    fn forward_chunk(
        &mut self,
        x: &candle_core::Tensor,
        mask: Option<&candle_core::Tensor>,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let (q, k_new, v_new) = self.qkv(x)?;
        let added = k_new.dim(1)?;
        self.reserve(added, &k_new)?;
        let kv = self.kv.as_mut().expect("reserved");
        kv.k.slice_set(&k_new, 1, kv.len)?;
        kv.v.slice_set(&v_new, 1, kv.len)?;
        kv.len += added;
        let len = kv.len;
        let k_all = kv.k.narrow(1, 0, len)?;
        let v_all = kv.v.narrow(1, 0, len)?;
        let attended = self.attend(&q, &k_all, &v_all, mask)?;
        self.close_halves(x, attended)
    }

    /// Ensure the KV buffers hold `added` more rows: allocate at the
    /// next KV_RESERVE_STEP multiple and copy the live prefix once.
    fn reserve(
        &mut self,
        added: usize,
        template: &candle_core::Tensor,
    ) -> LibAlmostResult<()> {
        let needed = self.kv.as_ref().map_or(added, |kv| kv.len + added);
        if let Some(kv) = &self.kv
            && needed <= kv.k.dim(1)?
        {
            return Ok(());
        }
        let (heads, _, head_dim) = template.dims3()?;
        let capacity = needed.div_ceil(KV_RESERVE_STEP) * KV_RESERVE_STEP;
        let new_k = candle_core::Tensor::zeros(
            (heads, capacity, head_dim),
            template.dtype(),
            template.device(),
        )?;
        let new_v = new_k.zeros_like()?;
        let len = self.kv.as_ref().map_or(0, |kv| kv.len);
        if let Some(kv) = &self.kv
            && kv.len > 0
        {
            new_k.slice_set(&kv.k.narrow(1, 0, kv.len)?.contiguous()?, 1, 0)?;
            new_v.slice_set(&kv.v.narrow(1, 0, kv.len)?.contiguous()?, 1, 0)?;
        }
        self.kv = Some(AttnKv { k: new_k, v: new_v, len });
        Ok(())
    }

    /// The post-norm block's residual adds shared by both paths.
    fn close_halves(
        &self,
        x: &candle_core::Tensor,
        attended: candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let attn_out = norms::rms_norm(&attended, &self.post_attention_layernorm, self.rms_eps)?;
        let h = (x + attn_out)?;
        let mlp_out = norms::rms_norm(
            &self.mlp.forward(&h)?,
            &self.post_feedforward_layernorm,
            self.rms_eps,
        )?;
        Ok((h + mlp_out)?)
    }

    /// Projections with the full-projection-width q/k RMSNorm BEFORE
    /// the head reshape; [heads, t, head_dim] matmul-ready.
    fn qkv(
        &self,
        x: &candle_core::Tensor,
    ) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)> {
        let (seq_len, _) = x.dims2()?;
        let q = norms::rms_norm(&self.q_proj.forward(x)?, &self.q_norm, self.rms_eps)?;
        let k = norms::rms_norm(&self.k_proj.forward(x)?, &self.k_norm, self.rms_eps)?;
        let v = self.v_proj.forward(x)?;
        let heads = |t: candle_core::Tensor| -> LibAlmostResult<candle_core::Tensor> {
            Ok(t.reshape((seq_len, self.num_heads, self.head_dim))?
                .transpose(0, 1)?
                .contiguous()?)
        };
        Ok((heads(q)?, heads(k)?, heads(v)?))
    }

    /// Eager MHA, NoPE: q [heads, t, d] against the given k/v span
    /// (its own chunk stateless, the whole cache carried); softmax in
    /// f32 cast back. A single-row query (decode) runs mask-free.
    /// With the flash-attn feature compiled, an eligible prefill
    /// (cuda, bf16/f16, q_len > 1) dispatches to the flash kernel
    /// instead - decode stays eager (candle-flash-attn 0.11 has no
    /// split-kv kernel; the era-one measured gate), and cpu/f32 legs
    /// never dispatch, so they stay bitwise-identical either way.
    fn attend(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
        mask: Option<&candle_core::Tensor>,
    ) -> LibAlmostResult<candle_core::Tensor> {
        #[cfg(feature = "flash-attn")]
        if q.dim(1)? > 1
            && q.device().is_cuda()
            && matches!(
                q.dtype(),
                candle_core::DType::BF16 | candle_core::DType::F16
            )
        {
            return self.attend_flash(q, k, v);
        }
        let seq_len = q.dim(1)?;
        let dtype = q.dtype();
        let mut scores = (q.matmul(&k.transpose(1, 2)?)? * (self.head_dim as f64).powf(-0.5))?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(&mask.unsqueeze(0)?)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores.to_dtype(candle_core::DType::F32)?)?
            .to_dtype(dtype)?;
        let out = probs
            .matmul(v)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq_len, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }

    /// The flash prefill: q the tail block of the k/v span, causal
    /// with the kernel's bottom-right alignment (absolute-position
    /// causality over the carried prefix - the era-one R1 finding).
    /// Layout: flash takes [batch, seq, heads, head_dim].
    #[cfg(feature = "flash-attn")]
    fn attend_flash(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let seq_len = q.dim(1)?;
        let block = |t: &candle_core::Tensor| -> LibAlmostResult<candle_core::Tensor> {
            Ok(t.transpose(0, 1)?.unsqueeze(0)?.contiguous()?)
        };
        let out = r::flash::flash_attn(
            &block(q)?,
            &block(k)?,
            &block(v)?,
            (self.head_dim as f32).powf(-0.5),
            true,
        )?;
        let out = out
            .squeeze(0)?
            .reshape((seq_len, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }

    /// Length resets; reserved capacity stays (the era-one semantic).
    fn clear_cache(&mut self) {
        if let Some(kv) = &mut self.kv {
            kv.len = 0;
        }
    }
}

enum Layer {
    Gdn(GdnLayer),
    Attn(AttnLayer),
}

/// The era-two hybrid model: forward_all (stateless whole-sequence,
/// the parity instrument) plus the carried forward_chunk path
/// (chunked prefill + decode over the layers' internal caches).
pub struct OlmoHybrid {
    embed_tokens: candle_core::Tensor,
    layers: Vec<Layer>,
    norm: candle_core::Tensor,
    lm_head: candle_nn::Linear,
    rms_eps: f64,
    context_len: usize,
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
            context_len: 0,
        })
    }

    /// All-position logits for one unbatched id sequence: [T] u32 in,
    /// [T, vocab] out in the model dtype (row r predicts token r+1).
    /// STATELESS - never touches the carried caches.
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

    /// One CARRIED forward over the next chunk of the context: [t] u32
    /// in, [t, vocab] all-position logits out; every layer's cache
    /// (GDN state + conv tails, attention KV) advances by t. Decode is
    /// t == 1 (mask-free); prefill chunks any t. Chunk boundaries are
    /// logit-exact to f32 rounding against the stateless path.
    pub fn forward_chunk(
        &mut self,
        input_ids: &candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let seq_len = input_ids.dim(0)?;
        let mut hidden = self.embed_tokens.index_select(input_ids, 0)?;
        let mask = if seq_len > 1 {
            Some(offset_causal_mask(
                seq_len,
                self.context_len,
                hidden.dtype(),
                hidden.device(),
            )?)
        } else {
            None
        };
        for layer in &mut self.layers {
            hidden = match layer {
                Layer::Gdn(layer) => layer.forward_chunk(&hidden)?,
                Layer::Attn(layer) => layer.forward_chunk(&hidden, mask.as_ref())?,
            };
        }
        self.context_len += seq_len;
        let hidden = norms::rms_norm(&hidden, &self.norm, self.rms_eps)?;
        Ok(self.lm_head.forward(&hidden)?)
    }

    /// Reset every carried cache (GDN states, conv tails, attention
    /// KV) and the context length; the next forward_chunk starts a
    /// fresh context.
    pub fn clear_cache(&mut self) -> LibAlmostResult<()> {
        for layer in &mut self.layers {
            match layer {
                Layer::Gdn(layer) => layer.clear_cache()?,
                Layer::Attn(layer) => layer.clear_cache(),
            }
        }
        self.context_len = 0;
        Ok(())
    }

    /// Tokens currently held in the carried caches.
    pub fn context_len(&self) -> usize {
        self.context_len
    }

    /// The device the model's tensors live on.
    pub fn device(&self) -> &candle_core::Device {
        self.embed_tokens.device()
    }
}

/// Additive causal mask [T, T]: 0 on and below the diagonal, -inf
/// above (exp saturates to exactly 0 either way upstream renders it).
fn causal_mask(
    seq_len: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> LibAlmostResult<candle_core::Tensor> {
    offset_causal_mask(seq_len, 0, dtype, device)
}

/// Additive causal mask for a chunk at an offset: [t, past + t], row
/// r sees every column through past + r, -inf beyond.
fn offset_causal_mask(
    seq_len: usize,
    past: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> LibAlmostResult<candle_core::Tensor> {
    let width = past + seq_len;
    let mut values = vec![0f32; seq_len * width];
    for (row, chunk) in values.chunks_exact_mut(width).enumerate() {
        for value in chunk.iter_mut().skip(past + row + 1) {
            *value = f32::NEG_INFINITY;
        }
    }
    Ok(
        candle_core::Tensor::from_vec(values, (seq_len, width), device)?
            .to_dtype(dtype)?,
    )
}
