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
/// gate/up ride one row-fused projection ([2 * intermediate, hidden],
/// row order gate | up); down stays separate (its input is the
/// product, not x).
struct Mlp {
    gate_up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    intermediate: usize,
}

impl Mlp {
    fn new(
        hidden: usize,
        intermediate: usize,
        vb: candle_nn::VarBuilder,
    ) -> LibAlmostResult<Self> {
        let gate_up = candle_core::Tensor::cat(
            &[
                &vb.pp("gate_proj").get((intermediate, hidden), "weight")?,
                &vb.pp("up_proj").get((intermediate, hidden), "weight")?,
            ],
            0,
        )?;
        Ok(Self {
            gate_up_proj: candle_nn::Linear::new(gate_up, None),
            down_proj: candle_nn::linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
            intermediate,
        })
    }

    fn forward(&self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gated = gate_up.narrow(1, 0, self.intermediate)?.silu()?;
        let up = gate_up.narrow(1, self.intermediate, self.intermediate)?;
        Ok(self.down_proj.forward(&gated.mul(&up)?)?)
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
    /// q/k/v/g projections row-fused at load ([2*key + 2*value,
    /// hidden]; row order q | k | v | g, so the first 2*key + value
    /// columns of the output are exactly the batched conv's input).
    /// a_proj/b_proj deliberately stay separate: folding them into
    /// the fused gemv changes the gating scalars' accumulation order,
    /// and that drift compounds through the recurrent state.
    qkvg_proj: candle_nn::Linear,
    a_proj: candle_nn::Linear,
    b_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    /// The three depthwise conv weights concatenated along channels
    /// ([2*key + value, 1, kernel]) - one batched conv pass per
    /// forward instead of three; per-channel math is bit-identical.
    conv_weight: candle_core::Tensor,
    /// The same weights prestored row-major ([kernel, channels]) for
    /// the fused decode tap's packed layout; Some only when fusion is
    /// armed.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    conv_weight_rows: Option<candle_core::Tensor>,
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
    /// GdnChainFusion armed: the decode step rides the fused kernel
    /// on cuda (the cpu path is always the classic chain).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fused_gdn: bool,
    state: candle_core::Tensor,
    conv_tail: candle_core::Tensor,
}

impl GdnLayer {
    fn new(
        config: &OlmoHybridConfig,
        fused_gdn: bool,
        vb: candle_nn::VarBuilder,
    ) -> LibAlmostResult<Self> {
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
        let conv_weight = candle_core::Tensor::cat(
            &[
                &vb_attn.pp("q_conv1d").get((key_dim, 1, kernel), "weight")?,
                &vb_attn.pp("k_conv1d").get((key_dim, 1, kernel), "weight")?,
                &vb_attn.pp("v_conv1d").get((value_dim, 1, kernel), "weight")?,
            ],
            0,
        )?;
        let conv_weight_rows = if fused_gdn {
            Some(conv_weight.squeeze(1)?.transpose(0, 1)?.contiguous()?)
        } else {
            None
        };
        Ok(Self {
            state: candle_core::Tensor::zeros(
                (heads, config.linear_key_head_dim, config.linear_value_head_dim),
                candle_core::DType::F32,
                &device,
            )?,
            conv_tail: f32_zeros((kernel - 1, key_dim + key_dim + value_dim))?,
            input_layernorm: vb.pp("input_layernorm").get(hidden, "weight")?,
            post_attention_layernorm: vb.pp("post_attention_layernorm").get(hidden, "weight")?,
            qkvg_proj: candle_nn::Linear::new(
                candle_core::Tensor::cat(
                    &[
                        &vb_attn.pp("q_proj").get((key_dim, hidden), "weight")?,
                        &vb_attn.pp("k_proj").get((key_dim, hidden), "weight")?,
                        &vb_attn.pp("v_proj").get((value_dim, hidden), "weight")?,
                        &vb_attn.pp("g_proj").get((value_dim, hidden), "weight")?,
                    ],
                    0,
                )?,
                None,
            ),
            a_proj: candle_nn::linear_no_bias(hidden, heads, vb_attn.pp("a_proj"))?,
            b_proj: candle_nn::linear_no_bias(hidden, heads, vb_attn.pp("b_proj"))?,
            o_proj: candle_nn::linear_no_bias(value_dim, hidden, vb_attn.pp("o_proj"))?,
            conv_weight,
            conv_weight_rows,
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
            fused_gdn,
        })
    }

    /// One fused gemv, split by the load-time row order: the packed
    /// conv input (q|k|v - the batched conv weight's channel order)
    /// and the raw gate span (consumed by finish_mixer after the
    /// recurrence).
    fn project_fused(
        &self,
        x: &candle_core::Tensor,
    ) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
        let qkvg = self.qkvg_proj.forward(x)?;
        let conv_width = 2 * self.key_width + self.value_width;
        Ok((
            qkvg.narrow(1, 0, conv_width)?.contiguous()?,
            qkvg.narrow(1, conv_width, self.value_width)?.contiguous()?,
        ))
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
        self.close_halves(x, mixed, false)
    }

    fn forward_chunk(&mut self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let fused = self.fused_gdn;
        let mixed = self.mixer_carried(&norms::rms_norm_auto(
            x,
            &self.input_layernorm,
            self.rms_eps,
            fused,
        )?)?;
        self.close_halves(x, mixed, fused)
    }

    /// The pre-norm block's residual adds shared by both paths;
    /// `fused` arms the decode norm kernel (carried callers only).
    fn close_halves(
        &self,
        x: &candle_core::Tensor,
        mixed: candle_core::Tensor,
        fused: bool,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let h = (x + mixed)?;
        let mlp_in =
            norms::rms_norm_auto(&h, &self.post_attention_layernorm, self.rms_eps, fused)?;
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

    /// The gated output norm + o_proj tail shared by both mixer paths;
    /// gate is the fused projection's raw g span. A carried decode row
    /// with fusion armed rides the single-launch gated kernel over the
    /// RAW f32 y (the classic path's y -> bf16 round-trip before the
    /// variance disappears; envelope-class).
    fn finish_mixer(
        &self,
        y: candle_core::Tensor,
        gate: candle_core::Tensor,
        seq_len: usize,
        fused: bool,
    ) -> LibAlmostResult<candle_core::Tensor> {
        #[cfg(feature = "cuda")]
        if fused
            && seq_len == 1
            && y.device().is_cuda()
            && y.dtype() == candle_core::DType::F32
            && gate.dtype() == candle_core::DType::BF16
        {
            let y_rows = y.reshape((self.num_heads, self.head_v_dim))?;
            let gate_rows = gate
                .reshape((self.num_heads, self.head_v_dim))?
                .to_dtype(candle_core::DType::F32)?;
            let packed =
                candle_core::Tensor::cat(&[&y_rows, &gate_rows], 1)?.contiguous()?;
            let out = candle_core::Tensor::zeros(
                (self.num_heads, self.head_v_dim),
                candle_core::DType::BF16,
                y.device(),
            )?;
            out.inplace_op3(
                &packed,
                &self.o_norm,
                &fused::RmsNormGatedFused {
                    eps: gdn::O_NORM_EPS as f32,
                },
            )?;
            return Ok(self
                .o_proj
                .forward(&out.reshape((1, self.num_heads * self.head_v_dim))?)?);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = fused;
        let y = y.to_dtype(gate.dtype())?;
        let gate = gate.reshape((seq_len, self.num_heads, self.head_v_dim))?;
        let y = norms::rms_norm_gated(&y, &gate, &self.o_norm, gdn::O_NORM_EPS)?;
        Ok(self
            .o_proj
            .forward(&y.reshape((seq_len, self.num_heads * self.head_v_dim))?)?)
    }

    /// The GDN mixer, STATELESS sequential form (the D2 parity
    /// instrument): zero-seeded per-token recurrence, no state kept.
    fn mixer_stateless(&self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let (seq_len, _) = x.dims2()?;
        let (conv_in, gate) = self.project_fused(x)?;
        let conv_out = gdn::causal_conv_silu(&conv_in, &self.conv_weight)?;
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
        self.finish_mixer(y, gate, seq_len, false)
    }

    /// The GDN mixer, CARRIED form (the D3 engine path): conv tails
    /// prepend the chunk, the persisted state seeds the rule, and both
    /// advance. Decode (seq 1) rides the recurrent step; larger chunks
    /// ride the chunked rule - the upstream's own path split.
    /// The carried caches (state, conv tail) update IN PLACE via
    /// slice_set - never rebind - so their device addresses stay
    /// stable for the lifetime of the layer. Captured decode graphs
    /// bake those addresses; a rebind anywhere (this path, clear)
    /// would leave every cached graph reading dead memory.
    fn mixer_carried(&mut self, x: &candle_core::Tensor) -> LibAlmostResult<candle_core::Tensor> {
        let (seq_len, _) = x.dims2()?;
        let (conv_in, gate) = self.project_fused(x)?;
        let conv_out = self.conv_carried(&conv_in, seq_len)?;
        let (q, k, v) = self.split_conv(&conv_out)?;
        let (q, k, v, g, beta) = self.heads_and_gates(q, k, v, x, seq_len)?;
        let y = if seq_len == 1 {
            self.carried_decode_step(&q, &k, &v, &g, &beta)?
        } else {
            let (out, next_state) = gdn::chunk_rule(&q, &k, &v, &g, &beta, &self.state)?;
            self.state.slice_set(&next_state, 0, 0)?;
            out
        };
        self.finish_mixer(y, gate, seq_len, self.fused_gdn)
    }

    /// One carried decode step (t == 1): the fused kernel when armed
    /// on cuda (the state updates in place INSIDE the launch), the
    /// classic candle chain otherwise. Both return the [1, heads, dv]
    /// f32 readout.
    fn carried_decode_step(
        &mut self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
        g: &candle_core::Tensor,
        beta: &candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        #[cfg(feature = "cuda")]
        if self.fused_gdn && q.device().is_cuda() {
            let decay = g.exp()?.reshape((self.num_heads, 1))?;
            let beta_column = beta.reshape((self.num_heads, 1))?;
            // cat on a non-zero dim returns a transposed view (the
            // era-one contiguity lesson); the kernel wants packed rows.
            let packed = candle_core::Tensor::cat(
                &[
                    &q.squeeze(0)?,
                    &k.squeeze(0)?,
                    &v.squeeze(0)?,
                    &decay,
                    &beta_column,
                ],
                1,
            )?
            .contiguous()?;
            let y = candle_core::Tensor::zeros(
                (self.num_heads, self.head_v_dim),
                candle_core::DType::F32,
                q.device(),
            )?;
            self.state
                .inplace_op3(&packed, &y, &fused::GdnFusedStep)?;
            return Ok(y.unsqueeze(0)?);
        }
        let (next_state, y_t) = gdn::recurrent_step(
            &self.state,
            &q.squeeze(0)?,
            &k.squeeze(0)?,
            &v.squeeze(0)?,
            &g.exp()?.squeeze(0)?,
            &beta.squeeze(0)?,
        )?;
        self.state.slice_set(&next_state, 0, 0)?;
        Ok(y_t.unsqueeze(0)?)
    }

    /// The carried conv over one chunk: the fused single-launch tap
    /// on a cuda decode row (the tail rotates in place INSIDE the
    /// kernel - no slice_set), the classic chain otherwise.
    fn conv_carried(
        &mut self,
        conv_in: &candle_core::Tensor,
        seq_len: usize,
    ) -> LibAlmostResult<candle_core::Tensor> {
        #[cfg(feature = "cuda")]
        if seq_len == 1
            && self.fused_gdn
            && conv_in.device().is_cuda()
            && conv_in.dtype() == candle_core::DType::BF16
            && let Some(rows) = &self.conv_weight_rows
        {
            let packed = candle_core::Tensor::cat(&[conv_in, rows], 0)?;
            let out = candle_core::Tensor::zeros(
                (1, 2 * self.key_width + self.value_width),
                candle_core::DType::BF16,
                conv_in.device(),
            )?;
            out.inplace_op3(&self.conv_tail, &packed, &fused::ConvStepFused)?;
            return Ok(out);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = seq_len;
        let (conv_out, tail) =
            gdn::conv_with_tail(conv_in, &self.conv_weight, &self.conv_tail)?;
        self.conv_tail.slice_set(&tail, 0, 0)?;
        Ok(conv_out)
    }

    /// Zero the carried caches in place (address-stable; see
    /// mixer_carried).
    fn clear_cache(&mut self) -> LibAlmostResult<()> {
        self.state.slice_set(&self.state.zeros_like()?, 0, 0)?;
        self.conv_tail
            .slice_set(&self.conv_tail.zeros_like()?, 0, 0)?;
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
    /// q/k/v projections row-fused at load ([3 * width, hidden]; row
    /// order q | k | v); o stays separate (its input is the attention
    /// output, not x).
    qkv_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: candle_core::Tensor,
    k_norm: candle_core::Tensor,
    mlp: Mlp,
    num_heads: usize,
    head_dim: usize,
    rms_eps: f64,
    /// Drives the flash prefill dispatch; only read under the
    /// flash-attn feature (dead by construction on cuda-only builds).
    #[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
    use_flash_attn: bool,
    /// GdnChainFusion armed: decode norms ride the fused kernel on
    /// cuda (the stateless path always runs the classic chain).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fused_gdn: bool,
    kv: Option<AttnKv>,
    /// The KV buffers are baked into a captured decode graph: their
    /// eventual replacement must PARK them (mem::forget), never free
    /// them - ever-captured buffers refuse cuMemFreeAsync (the
    /// captured-buffer free law).
    buffers_captured: bool,
    /// Latched by a reserve that parked captured buffers; the model
    /// collects it and retires capture for its lifetime.
    parked_graphs: bool,
    /// A3 eviction armed (settings.eviction present at construction).
    evict: bool,
    /// Last-pass scores beside the KV: [heads, capacity, 1] f32,
    /// reserved/parked in lockstep with the KV buffers (captured
    /// graphs bake its address too).
    scores: Option<candle_core::Tensor>,
    /// Persistent (heads, 1, 1) f32 zero row - the self-slot score
    /// reset's source, address-stable for captured graphs.
    score_zero: Option<candle_core::Tensor>,
    /// A2 observation accumulator; Some only while a profile battery
    /// is armed (interior mutability: attention runs under &self).
    #[cfg(feature = "attn-profile")]
    profile: std::cell::RefCell<Option<profile::ProfileAccum>>,
}

impl AttnLayer {
    fn new(
        config: &OlmoHybridConfig,
        use_flash_attn: bool,
        evict: bool,
        fused_gdn: bool,
        vb: candle_nn::VarBuilder,
    ) -> LibAlmostResult<Self> {
        let hidden = config.hidden_size;
        let width = config.num_attention_heads * config.head_dim();
        let vb_attn = vb.pp("self_attn");
        Ok(Self {
            post_attention_layernorm: vb.pp("post_attention_layernorm").get(hidden, "weight")?,
            post_feedforward_layernorm: vb
                .pp("post_feedforward_layernorm")
                .get(hidden, "weight")?,
            qkv_proj: candle_nn::Linear::new(
                candle_core::Tensor::cat(
                    &[
                        &vb_attn.pp("q_proj").get((width, hidden), "weight")?,
                        &vb_attn.pp("k_proj").get((width, hidden), "weight")?,
                        &vb_attn.pp("v_proj").get((width, hidden), "weight")?,
                    ],
                    0,
                )?,
                None,
            ),
            o_proj: candle_nn::linear_no_bias(width, hidden, vb_attn.pp("o_proj"))?,
            q_norm: vb_attn.pp("q_norm").get(width, "weight")?,
            k_norm: vb_attn.pp("k_norm").get(width, "weight")?,
            mlp: Mlp::new(hidden, config.intermediate_size, vb.pp("mlp"))?,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            rms_eps: config.rms_norm_eps,
            use_flash_attn,
            fused_gdn,
            kv: None,
            buffers_captured: false,
            parked_graphs: false,
            evict,
            scores: None,
            score_zero: None,
            #[cfg(feature = "attn-profile")]
            profile: std::cell::RefCell::new(None),
        })
    }

    fn kv_len(&self) -> usize {
        self.kv.as_ref().map_or(0, |kv| kv.len)
    }

    fn kv_capacity(&self) -> usize {
        self.kv
            .as_ref()
            .and_then(|kv| kv.k.dims().get(1).copied())
            .unwrap_or(0)
    }

    /// Host bookkeeping for a staged decode step (the device write
    /// already happened in-graph).
    fn advance_len(&mut self) {
        if let Some(kv) = &mut self.kv {
            kv.len += 1;
        }
    }

    /// Read-and-clear the park latch.
    fn take_parked(&mut self) -> bool {
        std::mem::take(&mut self.parked_graphs)
    }

    fn forward(
        &self,
        x: &candle_core::Tensor,
        mask: &candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let (q, k, v) = self.qkv(x, false)?;
        let attended = self.attend(&q, &k, &v, Some(mask))?;
        self.close_halves(x, attended, false)
    }

    fn forward_chunk(
        &mut self,
        x: &candle_core::Tensor,
        mask: Option<&candle_core::Tensor>,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let (q, k_new, v_new) = self.qkv(x, self.fused_gdn)?;
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
        if self.evict {
            self.score_pass(&q, added)?;
        }
        self.close_halves(x, attended, self.fused_gdn)
    }

    /// The armed last-pass scoring after a carried forward: prefill
    /// chunks re-score the whole store from their tail queries (by the
    /// final chunk that is the question window); a decode row
    /// overwrites with its own attention masses and zeroes its own
    /// slot (recent protection covers the fresh token - a large
    /// self-attention score must not outlive the recent window).
    fn score_pass(&mut self, q: &candle_core::Tensor, added: usize) -> LibAlmostResult<()> {
        let Some(kv) = &self.kv else {
            return Ok(());
        };
        let len = kv.len;
        let scale = (self.head_dim as f64).powf(-0.5);
        let tail = if added > 1 {
            evict::SCORE_TAIL.min(added)
        } else {
            1
        };
        let q_tail = q.narrow(1, added - tail, tail)?;
        let scores_new =
            evict::last_pass_scores(&q_tail, &kv.k.narrow(1, 0, len)?, scale)?;
        let scores = self
            .scores
            .as_ref()
            .expect("armed layers carry score buffers");
        scores.slice_set(&scores_new, 1, 0)?;
        if added == 1 {
            let zero = self
                .score_zero
                .as_ref()
                .expect("armed layers carry the zero row");
            scores.slice_set(zero, 1, len - 1)?;
        }
        Ok(())
    }

    /// Compact the store to the per-head keep-sets (order-preserving;
    /// all sets one length). `shrink_to` re-homes the survivors into
    /// fresh capacity-sized buffers (the post-prefill residency win) -
    /// only legal while nothing captured the buffers; otherwise (and
    /// at overflow epochs) the gather packs IN PLACE, keeping every
    /// baked address alive.
    fn compact(
        &mut self,
        keep: &[Vec<u32>],
        shrink_to: Option<usize>,
    ) -> LibAlmostResult<()> {
        let Some(kv) = &self.kv else {
            return Ok(());
        };
        let len = kv.len;
        let cap = keep.first().map_or(0, Vec::len);
        let gathered_k = evict::gather_rows(&kv.k, len, keep)?;
        let gathered_v = evict::gather_rows(&kv.v, len, keep)?;
        let scores = self
            .scores
            .as_ref()
            .expect("armed layers carry score buffers");
        let gathered_scores = evict::gather_rows(scores, len, keep)?;
        match shrink_to {
            Some(capacity) if !self.buffers_captured => {
                let new_k = candle_core::Tensor::zeros(
                    (self.num_heads, capacity, self.head_dim),
                    gathered_k.dtype(),
                    gathered_k.device(),
                )?;
                let new_v = new_k.zeros_like()?;
                let new_scores = candle_core::Tensor::zeros(
                    (self.num_heads, capacity, 1),
                    candle_core::DType::F32,
                    gathered_k.device(),
                )?;
                new_k.slice_set(&gathered_k, 1, 0)?;
                new_v.slice_set(&gathered_v, 1, 0)?;
                new_scores.slice_set(&gathered_scores, 1, 0)?;
                self.kv = Some(AttnKv { k: new_k, v: new_v, len: cap });
                self.scores = Some(new_scores);
            }
            _ => {
                let kv = self.kv.as_mut().expect("checked above");
                kv.k.slice_set(&gathered_k, 1, 0)?;
                kv.v.slice_set(&gathered_v, 1, 0)?;
                scores.slice_set(&gathered_scores, 1, 0)?;
                kv.len = cap;
            }
        }
        Ok(())
    }

    /// Ensure the KV buffers hold `added` more rows: allocate at the
    /// next KV_RESERVE_STEP multiple and copy the live prefix once.
    fn reserve(
        &mut self,
        added: usize,
        template: &candle_core::Tensor,
    ) -> LibAlmostResult<()> {
        let needed = self.kv.as_ref().map_or(added, |kv| kv.len + added);
        let capacity = needed.div_ceil(KV_RESERVE_STEP) * KV_RESERVE_STEP;
        self.reserve_capacity(capacity, template.dtype(), template.device())
    }

    /// Ensure the KV buffers hold at least `capacity` rows (the graph
    /// arm's pre-grow; also the append path's grow core). Replaced
    /// buffers that a captured graph baked are PARKED, never freed -
    /// the armed score buffer rides the same lifecycle (captured
    /// graphs bake its address too).
    fn reserve_capacity(
        &mut self,
        capacity: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> LibAlmostResult<()> {
        if self.kv_capacity() >= capacity.max(1) && (!self.evict || self.scores.is_some()) {
            return Ok(());
        }
        let (heads, head_dim) = (self.num_heads, self.head_dim);
        let capacity = capacity.max(self.kv_capacity()).max(1);
        let new_k = candle_core::Tensor::zeros((heads, capacity, head_dim), dtype, device)?;
        let new_v = new_k.zeros_like()?;
        let new_scores = if self.evict {
            Some(candle_core::Tensor::zeros(
                (heads, capacity, 1),
                candle_core::DType::F32,
                device,
            )?)
        } else {
            None
        };
        let len = self.kv.as_ref().map_or(0, |kv| kv.len);
        if let Some(kv) = &self.kv
            && kv.len > 0
        {
            new_k.slice_set(&kv.k.narrow(1, 0, kv.len)?.contiguous()?, 1, 0)?;
            new_v.slice_set(&kv.v.narrow(1, 0, kv.len)?.contiguous()?, 1, 0)?;
            if let (Some(new_scores), Some(old_scores)) = (&new_scores, &self.scores) {
                new_scores.slice_set(&old_scores.narrow(1, 0, kv.len)?.contiguous()?, 1, 0)?;
            }
        }
        let old_scores = self.scores.take();
        if let Some(old) = self.kv.take()
            && self.buffers_captured
        {
            std::mem::forget(old.k);
            std::mem::forget(old.v);
            if let Some(old_scores) = old_scores {
                std::mem::forget(old_scores);
            }
            self.buffers_captured = false;
            self.parked_graphs = true;
        }
        self.kv = Some(AttnKv { k: new_k, v: new_v, len });
        self.scores = new_scores;
        if self.evict && self.score_zero.is_none() {
            self.score_zero = Some(candle_core::Tensor::zeros(
                (heads, 1, 1),
                candle_core::DType::F32,
                device,
            )?);
        }
        Ok(())
    }

    /// The post-norm block's residual adds shared by both paths;
    /// `fused` arms the decode norm kernel (carried callers only).
    fn close_halves(
        &self,
        x: &candle_core::Tensor,
        attended: candle_core::Tensor,
        fused: bool,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let attn_out = norms::rms_norm_auto(
            &attended,
            &self.post_attention_layernorm,
            self.rms_eps,
            fused,
        )?;
        let h = (x + attn_out)?;
        let mlp_out = norms::rms_norm_auto(
            &self.mlp.forward(&h)?,
            &self.post_feedforward_layernorm,
            self.rms_eps,
            fused,
        )?;
        Ok((h + mlp_out)?)
    }

    /// One fused projection gemv split into q/k/v, with the
    /// full-projection-width q/k RMSNorm BEFORE the head reshape;
    /// [heads, t, head_dim] matmul-ready. `fused` arms the decode
    /// norm kernel (carried callers only).
    fn qkv(
        &self,
        x: &candle_core::Tensor,
        fused: bool,
    ) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)> {
        let (seq_len, _) = x.dims2()?;
        let width = self.num_heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x)?;
        let q =
            norms::rms_norm_auto(&qkv.narrow(1, 0, width)?, &self.q_norm, self.rms_eps, fused)?;
        let k = norms::rms_norm_auto(
            &qkv.narrow(1, width, width)?,
            &self.k_norm,
            self.rms_eps,
            fused,
        )?;
        let v = qkv.narrow(1, 2 * width, width)?.contiguous()?;
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
        // An armed profile forces the sliced eager path regardless of
        // flash settings - the observation needs materialized weights.
        #[cfg(feature = "attn-profile")]
        if self.profile.borrow().is_some() {
            return self.attend_profiled(q, k, v, mask);
        }
        #[cfg(feature = "flash-attn")]
        if self.use_flash_attn
            && q.dim(1)? > 1
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

    /// The profile-armed attention: eager math in PROFILE_ROW_SLICE
    /// row slices (value-identical - softmax and the weighted sum are
    /// row-independent) so the f32 softmax transient stays bounded at
    /// 32K, folding each slice's masses into the armed accumulator.
    #[cfg(feature = "attn-profile")]
    fn attend_profiled(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
        mask: Option<&candle_core::Tensor>,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let seq_len = q.dim(1)?;
        let width = k.dim(1)?;
        let past = width - seq_len;
        let dtype = q.dtype();
        let scale = (self.head_dim as f64).powf(-0.5);
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let mut outs = Vec::new();
        let mut row = 0usize;
        while row < seq_len {
            let rows = profile::PROFILE_ROW_SLICE.min(seq_len - row);
            let mut scores = (q.narrow(1, row, rows)?.matmul(&k_t)? * scale)?;
            if let Some(mask) = mask {
                scores = scores.broadcast_add(&mask.narrow(0, row, rows)?.unsqueeze(0)?)?;
            }
            let probs =
                candle_nn::ops::softmax_last_dim(&scores.to_dtype(candle_core::DType::F32)?)?;
            if let Some(accum) = self.profile.borrow_mut().as_mut() {
                accum.accumulate(&probs, past + row, seq_len == 1)?;
            }
            outs.push(probs.to_dtype(dtype)?.matmul(v)?);
            row += rows;
        }
        let out = candle_core::Tensor::cat(&outs, 1)?
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
        // q packs (chunk-sized, cheap); k/v stay VIEWS - the flash
        // kernels take strided rows (last dim contiguous), and a
        // contiguous() here would copy the WHOLE carried span per
        // chunk per layer (~2 GiB-class transients late in a 32K
        // prefill, and the top copy share on the prefill path).
        let block = |t: &candle_core::Tensor| -> LibAlmostResult<candle_core::Tensor> {
            Ok(t.transpose(0, 1)?.unsqueeze(0)?.contiguous()?)
        };
        let span = |t: &candle_core::Tensor| -> LibAlmostResult<candle_core::Tensor> {
            Ok(t.transpose(0, 1)?.unsqueeze(0)?)
        };
        let out = r::flash::flash_attn(
            &block(q)?,
            &span(k)?,
            &span(v)?,
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

    /// The staged decode step (batch-1, t=1): the same math as the
    /// classic mask-free decode, over a static bucket width - the new
    /// k/v scatter to a device-resident slot, the pad mask hides the
    /// bucket's tail (exactly zero post-softmax mass), and every
    /// per-step value rides a staged buffer.
    #[cfg(feature = "cuda")]
    fn forward_graph(
        &self,
        x: &candle_core::Tensor,
        stage: &graph::DecodeStage,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let (q, k_new, v_new) = self.qkv(x, self.fused_gdn)?;
        let Some(kv) = &self.kv else {
            snafu::whatever!("graph decode reached unallocated KV buffers");
        };
        let write = graph::SlotWrite { dtype: k_new.dtype() };
        kv.k.inplace_op3(&k_new, &stage.kv_slot, &write)?;
        kv.v.inplace_op3(&v_new, &stage.kv_slot, &write)?;
        let keys = kv.k.narrow(1, 0, stage.bucket)?;
        let values = kv.v.narrow(1, 0, stage.bucket)?;
        let dtype = q.dtype();
        let scores = (q.matmul(&keys.transpose(1, 2)?)? * (self.head_dim as f64).powf(-0.5))?;
        let scores = scores.broadcast_add(&stage.kv_mask)?;
        let probs_f32 =
            candle_nn::ops::softmax_last_dim(&scores.to_dtype(candle_core::DType::F32)?)?;
        if self.evict {
            // The in-graph last-pass write: the row's masses over the
            // static bucket (pad columns write ~0 to slots beyond the
            // live length - overwritten when those slots fill), then
            // the staged self-slot zero (recent protection covers the
            // fresh token). Bucket-static shapes; the score buffer +
            // zero row are address-stable, so capture bakes them
            // safely.
            let score_buffer = self
                .scores
                .as_ref()
                .expect("armed layers carry score buffers");
            score_buffer.slice_set(&probs_f32.transpose(1, 2)?.contiguous()?, 1, 0)?;
            let zero = self
                .score_zero
                .as_ref()
                .expect("armed layers carry the zero row");
            score_buffer.inplace_op3(
                zero,
                &stage.kv_slot,
                &graph::SlotWrite { dtype: candle_core::DType::F32 },
            )?;
        }
        let probs = probs_f32.to_dtype(dtype)?;
        let out = probs
            .matmul(&values)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((1, self.num_heads * self.head_dim))?;
        let attended = self.o_proj.forward(&out)?;
        self.close_halves(x, attended, self.fused_gdn)
    }
}

enum Layer {
    Gdn(GdnLayer),
    Attn(AttnLayer),
}

/// The margin a graph arm pre-reserves past the live length when the
/// sample budget exceeds it (the 32768 default budget would
/// otherwise pre-grow gigabytes a typical decode never touches).
const RESERVE_DECODE_MARGIN: usize = 256;

/// The era-two hybrid model: forward_all (stateless whole-sequence,
/// the parity instrument) plus the carried forward_chunk path
/// (chunked prefill + decode over the layers' internal caches), and,
/// under settings.graph on a cuda build, the staged decode path that
/// CUDA graphs capture and replay.
pub struct OlmoHybrid {
    embed_tokens: candle_core::Tensor,
    layers: Vec<Layer>,
    norm: candle_core::Tensor,
    lm_head: candle_nn::Linear,
    rms_eps: f64,
    context_len: usize,
    settings: LibSettings,
    #[cfg(feature = "cuda")]
    graph_stage: Option<graph::DecodeStage>,
}

impl OlmoHybrid {
    pub fn new(
        config: &OlmoHybridConfig,
        settings: LibSettings,
        vb: candle_nn::VarBuilder,
    ) -> LibAlmostResult<Self> {
        config.validate()?;
        if settings.graph {
            #[cfg(not(feature = "cuda"))]
            snafu::whatever!("decode graphs need a cuda build (--features cuda)");
            #[cfg(feature = "cuda")]
            snafu::ensure_whatever!(
                vb.device().is_cuda(),
                "decode graphs need a cuda device"
            );
        }
        let vb_model = vb.pp("model");
        let embed_tokens = vb_model
            .pp("embed_tokens")
            .get((config.vocab_size, config.hidden_size), "weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let vb_layer = vb_model.pp(format!("layers.{layer_idx}"));
            layers.push(match config.layer_kind(layer_idx) {
                LayerKind::LinearAttention => {
                    Layer::Gdn(GdnLayer::new(config, settings.fused_gdn, vb_layer)?)
                }
                LayerKind::FullAttention => Layer::Attn(AttnLayer::new(
                    config,
                    settings.use_flash_attn,
                    settings.eviction.is_some(),
                    settings.fused_gdn,
                    vb_layer,
                )?),
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
            settings,
            #[cfg(feature = "cuda")]
            graph_stage: None,
        })
    }

    /// The settings this model was built with.
    pub fn settings(&self) -> &LibSettings {
        &self.settings
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
        self.contain_parked_grows();
        let hidden =
            norms::rms_norm_auto(&hidden, &self.norm, self.rms_eps, self.settings.fused_gdn)?;
        Ok(self.lm_head.forward(&hidden)?)
    }

    /// Collect the layers' park latches: a grow replaced
    /// ever-captured KV buffers, so every cached graph is stale and
    /// capture retires for this Model (one parked buffer set per
    /// lifetime stays the bound).
    #[cfg(feature = "cuda")]
    fn contain_parked_grows(&mut self) {
        let mut parked = false;
        for layer in &mut self.layers {
            if let Layer::Attn(attn) = layer
                && attn.take_parked()
            {
                parked = true;
            }
        }
        if parked {
            self.flush_captured_graphs();
            if let Some(stage) = &mut self.graph_stage {
                stage.armed = false;
                stage.capture_enabled = false;
                stage.capture_permitted = false;
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn contain_parked_grows(&mut self) {}

    /// Reset every carried cache (GDN states, conv tails, attention
    /// KV) and the context length; the next forward_chunk starts a
    /// fresh context. A staged graph stage disarms (an epoch) - its
    /// buffers and cached graphs survive for the next arm.
    pub fn clear_cache(&mut self) -> LibAlmostResult<()> {
        for layer in &mut self.layers {
            match layer {
                Layer::Gdn(layer) => layer.clear_cache()?,
                Layer::Attn(layer) => layer.clear_cache(),
            }
        }
        self.context_len = 0;
        #[cfg(feature = "cuda")]
        if let Some(stage) = &mut self.graph_stage {
            stage.armed = false;
        }
        Ok(())
    }

    /// Known-length KV pre-reserve (generate calls this before its
    /// prefill loop): one allocation at prompt + decode margin,
    /// KV_RESERVE_STEP-rounded, instead of ~one regrow per reserve
    /// step of prefill - each regrow holds old+new buffers during
    /// its copy and frees an odd-sized block into the raw cudaMalloc
    /// heap (the fragmentation that inflates the 32K peak). A decode
    /// outrunning the margin regrows coarsely as before.
    pub fn reserve_for_generation(&mut self, prompt_tokens: usize) -> LibAlmostResult<()> {
        let device = self.device().clone();
        let dtype = self.embed_tokens.dtype();
        let capacity = (prompt_tokens + RESERVE_DECODE_MARGIN)
            .div_ceil(KV_RESERVE_STEP)
            * KV_RESERVE_STEP;
        for layer in &mut self.layers {
            if let Layer::Attn(attn) = layer {
                attn.reserve_capacity(capacity, dtype, &device)?;
            }
        }
        self.contain_parked_grows();
        Ok(())
    }

    /// Tokens currently held in the carried caches.
    pub fn context_len(&self) -> usize {
        self.context_len
    }

    /// A3 keep-sets from the live last-pass scores, one per attention
    /// layer (host-side ranking; per-head sets, all decode_cap long).
    fn evict_keep_sets(
        &self,
        evict: &EvictionSettings,
    ) -> LibAlmostResult<Vec<Vec<Vec<u32>>>> {
        let mut sets = Vec::new();
        for layer in &self.layers {
            if let Layer::Attn(attn) = layer {
                let Some(kv) = &attn.kv else {
                    snafu::whatever!("eviction reached unallocated KV buffers");
                };
                let scores = attn
                    .scores
                    .as_ref()
                    .expect("armed layers carry score buffers")
                    .narrow(1, 0, kv.len)?
                    .squeeze(2)?
                    .to_vec2::<f32>()?;
                sets.push(
                    scores
                        .iter()
                        .map(|row| {
                            evict::keep_indices(row, evict.decode_cap, evict.recent, evict.sink)
                        })
                        .collect(),
                );
            }
        }
        Ok(sets)
    }

    /// A3: the one-shot post-prefill compaction (generate calls this
    /// between prefill and graph arming): rank by the FINAL scoring
    /// pass (the question window), compact every attention layer to
    /// the cap, and - buffers never yet captured - SHRINK them to the
    /// decode-bounded capacity, cashing the KV-residency win. GDN
    /// layers are untouched (constant state).
    pub(crate) fn evict_post_prefill(
        &mut self,
        evict: &EvictionSettings,
    ) -> LibAlmostResult<()> {
        if self.context_len <= evict.decode_cap {
            return Ok(());
        }
        let keep_sets = self.evict_keep_sets(evict)?;
        // Overflow epochs bound decode length at cap + slack, so this
        // capacity never regrows classic; a graph arm may pre-grow it
        // once more (pre-capture, a cap-sized copy - trivial).
        let capacity = (evict.decode_cap + evict::OVERFLOW_SLACK + RESERVE_DECODE_MARGIN)
            .div_ceil(KV_RESERVE_STEP)
            * KV_RESERVE_STEP;
        let mut sets = keep_sets.into_iter();
        for layer in &mut self.layers {
            if let Layer::Attn(attn) = layer {
                let keep = sets.next().expect("one keep-set per attention layer");
                attn.compact(&keep, Some(capacity))?;
            }
        }
        self.context_len = evict.decode_cap;
        Ok(())
    }

    /// A decode-overflow re-compaction epoch (uncaptured host work
    /// between steps): pack the store back to the cap IN PLACE (every
    /// baked address stays alive) and re-point an armed graph stage's
    /// mask at the shrunk length (stale columns re-hide; the bucket
    /// is unchanged, so the same captured graph replays on).
    pub(crate) fn evict_overflow(&mut self, evict: &EvictionSettings) -> LibAlmostResult<()> {
        let keep_sets = self.evict_keep_sets(evict)?;
        let mut sets = keep_sets.into_iter();
        for layer in &mut self.layers {
            if let Layer::Attn(attn) = layer {
                let keep = sets.next().expect("one keep-set per attention layer");
                attn.compact(&keep, None)?;
            }
        }
        self.context_len = evict.decode_cap;
        #[cfg(feature = "cuda")]
        if let Some(stage) = &mut self.graph_stage
            && stage.armed
        {
            stage.rearm(self.context_len)?;
        }
        Ok(())
    }

    /// Arm the A2 observation pass on every attention layer; `window`
    /// is the recency width the middle-mass metric excludes. Armed
    /// attention runs the sliced eager path (flash never dispatches).
    #[cfg(feature = "attn-profile")]
    pub fn arm_attn_profile(&mut self, window: usize) {
        for layer in &mut self.layers {
            if let Layer::Attn(attn) = layer {
                attn.profile
                    .replace(Some(profile::ProfileAccum::new(attn.num_heads, window)));
            }
        }
    }

    /// Collect + disarm the observation pass; reports carry absolute
    /// layer indices.
    #[cfg(feature = "attn-profile")]
    pub fn take_attn_profile(&mut self) -> Vec<profile::LayerProfile> {
        let mut reports = Vec::new();
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            if let Layer::Attn(attn) = layer
                && let Some(accum) = attn.profile.replace(None)
            {
                reports.push(accum.report(layer_index));
            }
        }
        reports
    }

    /// The device the model's tensors live on.
    pub fn device(&self) -> &candle_core::Device {
        self.embed_tokens.device()
    }

    /// The uniform attention KV length graph staging derives from
    /// (uniform by construction - every layer sees every token;
    /// asserted at arming).
    #[cfg(feature = "cuda")]
    fn graph_kv_state(&self) -> LibAlmostResult<(usize, usize)> {
        let mut kv_len: Option<usize> = None;
        let mut capacity = 0usize;
        for layer in &self.layers {
            if let Layer::Attn(attn) = layer {
                let len = attn.kv_len();
                if let Some(existing) = kv_len {
                    snafu::ensure_whatever!(
                        existing == len,
                        "attention KV lengths disagree ({existing} vs {len})"
                    );
                } else {
                    kv_len = Some(len);
                }
                capacity = attn.kv_capacity();
            }
        }
        let Some(kv_len) = kv_len else {
            snafu::whatever!("graph staging found no attention layers");
        };
        Ok((kv_len, capacity))
    }

    /// E4: arm the staged graph-mode decode path (generate() calls
    /// this after prefill). Pre-grows every attention layer's KV once
    /// to the run's bucket ceiling - so later bucket crossings never
    /// reallocate under a captured graph - then builds or re-arms the
    /// staged buffers from the live cache state.
    #[cfg(feature = "cuda")]
    pub(crate) fn arm_graph_decode(&mut self, expected_total: usize) -> LibAlmostResult<()> {
        snafu::ensure_whatever!(
            self.device().is_cuda(),
            "graph-mode decode requires a cuda device"
        );
        let device = self.device().clone();
        let dtype = self.embed_tokens.dtype();
        let (kv_len, _) = self.graph_kv_state()?;
        snafu::ensure_whatever!(
            kv_len == self.context_len,
            "attention KV length {kv_len} does not match the context length {}",
            self.context_len
        );
        let grain = self.settings.graph_bucket_grain;
        // The run ceiling: the sample budget's contribution clamps to
        // the reserve margin; a decode that outruns the armed
        // capacity falls to the classic path (the capacity epoch).
        let ceiling = expected_total
            .min(kv_len + 1 + RESERVE_DECODE_MARGIN)
            .max(kv_len + 1);
        let capacity = graph::bucket_for(ceiling, grain);
        for layer in &mut self.layers {
            if let Layer::Attn(attn) = layer {
                attn.reserve_capacity(capacity, dtype, &device)?;
            }
        }
        self.contain_parked_grows();
        let (_, kv_capacity) = self.graph_kv_state()?;
        let vocab = self.embed_tokens.dim(0)?;
        match &mut self.graph_stage {
            Some(stage) => {
                if stage.kv_capacity != kv_capacity {
                    stage.graphs.clear();
                    graph::trim_graph_memory();
                    stage.kv_capacity = kv_capacity;
                }
                stage.rearm(kv_len)?;
            }
            None => {
                let mut stage =
                    graph::DecodeStage::new(kv_len, grain, vocab, dtype, &device)?;
                stage.kv_capacity = kv_capacity;
                self.graph_stage = Some(stage);
            }
        }
        Ok(())
    }

    /// Non-cuda builds carry no graph path; arming is a hard error
    /// (settings.graph is already rejected at Model::new).
    #[cfg(not(feature = "cuda"))]
    pub(crate) fn arm_graph_decode(&mut self, _expected_total: usize) -> LibAlmostResult<()> {
        snafu::whatever!("this build carries no cuda support (decode graphs need --features cuda)")
    }

    /// Whether the staged graph-mode decode path is armed for the
    /// current cache state.
    #[cfg(feature = "cuda")]
    pub(crate) fn graph_armed(&self) -> bool {
        self.graph_stage.as_ref().is_some_and(|stage| stage.armed)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn graph_armed(&self) -> bool {
        false
    }

    /// Gate-side switch between capture+replay and uncaptured staged
    /// stepping (the gates' reference legs).
    #[cfg(feature = "cuda")]
    pub(crate) fn set_graph_capture(&mut self, enabled: bool) -> LibAlmostResult<()> {
        let Some(stage) = &mut self.graph_stage else {
            snafu::whatever!("no graph stage to configure (arm first)");
        };
        stage.capture_enabled = enabled;
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn set_graph_capture(&mut self, _enabled: bool) -> LibAlmostResult<()> {
        snafu::whatever!("this build carries no cuda support (decode graphs need --features cuda)")
    }

    /// E4 capacity epoch: when the next staged append would exceed
    /// the armed KV capacity, disarm and retire capture for this
    /// Model - the remaining steps take the classic path, whose
    /// append growth applies the park containment.
    #[cfg(feature = "cuda")]
    pub(crate) fn graph_disarm_when_full(&mut self) -> LibAlmostResult<()> {
        if !self.graph_armed() {
            return Ok(());
        }
        let (kv_len, capacity) = self.graph_kv_state()?;
        // The BUCKET the next step needs is the binding width (it
        // rounds up by the grain, so it can overshoot a capacity the
        // grain does not divide before the raw length would).
        let next_bucket = graph::bucket_for(kv_len + 1, self.settings.graph_bucket_grain);
        if next_bucket > capacity {
            self.flush_captured_graphs();
            if let Some(stage) = &mut self.graph_stage {
                stage.armed = false;
                stage.capture_enabled = false;
                stage.capture_permitted = false;
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn graph_disarm_when_full(&mut self) -> LibAlmostResult<()> {
        Ok(())
    }

    /// Drop every captured decode graph (the staged buffers and masks
    /// stay) and trim the driver's cached graph pools.
    #[cfg(feature = "cuda")]
    pub(crate) fn flush_captured_graphs(&mut self) {
        if let Some(stage) = &mut self.graph_stage {
            stage.graphs.clear();
            graph::trim_graph_memory();
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn flush_captured_graphs(&mut self) {}

    /// One staged decode step: consume `token`, return the persistent
    /// (1, vocab) f32 logits buffer (valid until the next step
    /// overwrites it). Host bookkeeping (KV lengths, context length)
    /// advances in lockstep with the device writes.
    #[cfg(feature = "cuda")]
    pub(crate) fn graph_decode_step(
        &mut self,
        token: u32,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let kv_len = self.context_len;
        let mut stage = match self.graph_stage.take() {
            Some(stage) => stage,
            None => snafu::whatever!("graph decode step without an armed stage"),
        };
        // Run the step with the stage held out, then ALWAYS put it
        // back - an error must not destroy the persistent staging.
        let step_result = self.graph_step_with(&mut stage, token, kv_len);
        let out = stage.logits_out.clone();
        self.graph_stage = Some(stage);
        step_result?;
        for layer in &mut self.layers {
            if let Layer::Attn(attn) = layer {
                attn.advance_len();
            }
        }
        self.context_len += 1;
        Ok(out)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn graph_decode_step(
        &mut self,
        _token: u32,
    ) -> LibAlmostResult<candle_core::Tensor> {
        snafu::whatever!("this build carries no cuda support (decode graphs need --features cuda)")
    }

    /// One staged step against a held-out stage: stage the dynamics,
    /// then either replay the bucket's captured graph (capturing it
    /// first if uncached) or run the sequence as ordinary ops (the
    /// uncaptured reference mode).
    #[cfg(feature = "cuda")]
    fn graph_step_with(
        &mut self,
        stage: &mut graph::DecodeStage,
        token: u32,
        kv_len: usize,
    ) -> LibAlmostResult<()> {
        snafu::ensure_whatever!(
            stage.armed,
            "graph decode step on a disarmed stage (cleared mid-run; re-arm first)"
        );
        stage.ensure_bucket(kv_len)?;
        stage.stage_step(token, kv_len)?;
        if !stage.capture_enabled {
            return self.graph_forward_sequence(stage);
        }
        let key = stage.bucket;
        match stage.graphs.get(key) {
            Some(graph) => {
                if let Err(error) = graph.launch() {
                    snafu::whatever!("decode graph launch failed: {error}");
                }
            }
            None => {
                // First step at this bucket: the capture's WARMUP run
                // performs this step's real work (and populates
                // candle's param cache); the recording that follows
                // only records - launching here would execute the
                // step twice.
                let captured = self.capture_decode_graph(stage)?;
                stage.graphs.insert(key, captured);
                // The KV buffers are baked into a captured graph now:
                // their eventual replacement must PARK them.
                for layer in self.layers.iter_mut() {
                    if let Layer::Attn(attn) = layer {
                        attn.buffers_captured = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// Capture the current bucket's decode sequence into an
    /// instantiated CUDA graph on candle's created stream. The pinned
    /// recipe: hold candle's param-cache guard, WARMUP-run the
    /// sequence uncaptured (populating the content-keyed dims/strides
    /// cache, whose miss during active capture is a designed hard
    /// error, and performing this step's real work), then record; the
    /// recording performs no work and its intermediates drop as
    /// in-graph free nodes. THREAD_LOCAL capture fails loudly on
    /// capture-illegal calls from this thread; AUTO_FREE_ON_LAUNCH is
    /// the relaunch semantic for in-graph memory nodes, and the exec
    /// is pre-uploaded explicitly (UPLOAD is WithParams-only).
    #[cfg(feature = "cuda")]
    fn capture_decode_graph(
        &mut self,
        stage: &graph::DecodeStage,
    ) -> LibAlmostResult<cudarc::driver::CudaGraph> {
        let cuda_device = match self.device() {
            candle_core::Device::Cuda(cuda_device) => cuda_device.clone(),
            _ => snafu::whatever!("graph capture requires a cuda device"),
        };
        let stream = cuda_device.cuda_stream();
        let _htod_cache = cuda_device.enable_cuda_graph_htod_cache();
        // Warmup: the real step, param cache populated.
        self.graph_forward_sequence(stage)?;
        if let Err(error) = stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        ) {
            snafu::whatever!("begin_capture failed: {error}");
        }
        let run_result = self.graph_forward_sequence(stage);
        let end_result = stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        run_result?;
        let graph = match end_result {
            Ok(Some(graph)) => graph,
            Ok(None) => snafu::whatever!("end_capture returned no graph"),
            Err(error) => snafu::whatever!("end_capture failed: {error}"),
        };
        if let Err(error) = graph.upload() {
            snafu::whatever!("graph upload failed: {error}");
        }
        Ok(graph)
    }

    /// The captured region: embed -> layers -> norm -> lm_head -> f32
    /// cast -> the persistent logits_out write. GDN layers run their
    /// classic t=1 carried step (fixed shapes, in-place state - the
    /// captured form by construction); attention layers run the
    /// staged bucket form. Every per-step value rides a staged
    /// buffer; no host constant is baked.
    #[cfg(feature = "cuda")]
    fn graph_forward_sequence(
        &mut self,
        stage: &graph::DecodeStage,
    ) -> LibAlmostResult<()> {
        let mut hidden = self.embed_tokens.index_select(&stage.ids, 0)?;
        for layer in &mut self.layers {
            hidden = match layer {
                Layer::Gdn(layer) => layer.forward_chunk(&hidden)?,
                Layer::Attn(layer) => layer.forward_graph(&hidden, stage)?,
            };
        }
        let hidden =
            norms::rms_norm_auto(&hidden, &self.norm, self.rms_eps, self.settings.fused_gdn)?;
        let logits = self
            .lm_head
            .forward(&hidden)?
            .to_dtype(candle_core::DType::F32)?;
        stage.logits_out.slice_set(&logits, 0, 0)?;
        Ok(())
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
