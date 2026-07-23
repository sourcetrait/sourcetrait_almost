//! The linear-attention (GDN) decoder layer - fully PRE-norm.
use crate::*;

/// A linear-attention (GDN) decoder layer - fully PRE-norm:
/// h = x + mixer(input_layernorm(x)), then
/// out = h + mlp(post_attention_layernorm(h)).
/// Carries the StateCarry cross-chunk cache: the f32 recurrent state
/// plus the three raw pre-conv tails (kernel-1 rows each, stored f32).
pub(crate) struct GdnLayer {
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
    /// PrepFusion static rows ([kernel, conv_width + 2 * heads] bf16):
    /// the conv-weight tap rows with A_log | dt_bias riding the row-0
    /// pad columns - the prep kernel's per-layer constant operand;
    /// Some only when fused_prefill is armed.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    prefill_static: Option<candle_core::Tensor>,
    /// FusedHeadPrep static ([1, 2 * heads] bf16, A_log | dt_bias) -
    /// the tail of the decode step's dyn row; Some only when
    /// fused_gdn is armed (independent of fused_prefill by design).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    decode_static: Option<candle_core::Tensor>,
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
    /// PrefillDispatch armed: multi-token chunks ride the fused
    /// prefill kernels on cuda (the cpu path and the stateless
    /// parity form always run the classic chain).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fused_prefill: bool,
    /// PrefillScratchReuse: the pool shared by every GDN layer (the
    /// model installs one handle post-construction when
    /// fused_prefill is armed; layers run sequentially, so one set
    /// serves all 24).
    #[cfg(feature = "cuda")]
    prefill_scratch: Option<fused_prefill::SharedPrefillScratch>,
    state: candle_core::Tensor,
    conv_tail: candle_core::Tensor,
    /// SpeculationPort shadow buffers (state + conv tail), built at
    /// the first spec mark; plain device copies - speculation rides
    /// the classic path only, so no capture/address constraints bind.
    spec_shadow: Option<(candle_core::Tensor, candle_core::Tensor)>,
    /// KernelAccept capture arm, set by shadow_save (the spec mark):
    /// the next multi-token carried chunk stashes its rule inputs.
    spec_capture: bool,
    /// The captured rule-input rows (conv_in [span, 2*key + value],
    /// a_rows / b_rows [span, heads], model dtype) - what a partial
    /// accept re-advances the recurrence from without a replay
    /// forward.
    spec_rows: Option<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)>,
}

impl GdnLayer {
    pub(crate) fn new(
        config: &OlmoHybridConfig,
        fused_gdn: bool,
        fused_prefill: bool,
        vb: candle_nn::VarBuilder,
    ) -> LibQuestResult<Self> {
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
        let a_log = vb_attn.get(heads, "A_log")?;
        let dt_bias = vb_attn.get(heads, "dt_bias")?;
        let prefill_static = if fused_prefill {
            let weight_rows = conv_weight.squeeze(1)?.transpose(0, 1)?.contiguous()?;
            let head_pad = candle_core::Tensor::cat(
                &[&a_log.reshape((1, heads))?, &dt_bias.reshape((1, heads))?],
                1,
            )?;
            let zero_pad = candle_core::Tensor::zeros(
                (kernel - 1, 2 * heads),
                weight_rows.dtype(),
                &device,
            )?;
            let pad_column = candle_core::Tensor::cat(&[&head_pad, &zero_pad], 0)?;
            Some(candle_core::Tensor::cat(&[&weight_rows, &pad_column], 1)?.contiguous()?)
        } else {
            None
        };
        let decode_static = if fused_gdn {
            Some(
                candle_core::Tensor::cat(
                    &[&a_log.reshape((1, heads))?, &dt_bias.reshape((1, heads))?],
                    1,
                )?
                .contiguous()?,
            )
        } else {
            None
        };
        Ok(Self {
            #[cfg(feature = "cuda")]
            prefill_scratch: None,
            state: candle_core::Tensor::zeros(
                (heads, config.linear_key_head_dim, config.linear_value_head_dim),
                candle_core::DType::F32,
                &device,
            )?,
            conv_tail: f32_zeros((kernel - 1, key_dim + key_dim + value_dim))?,
            spec_shadow: None,
            spec_capture: false,
            spec_rows: None,
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
            a_log,
            dt_bias,
            prefill_static,
            decode_static,
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
            fused_prefill,
        })
    }

    /// PrefillScratchReuse: install the model's shared pool handle
    /// (post-construction; only when fused_prefill is armed).
    #[cfg(feature = "cuda")]
    pub(crate) fn install_prefill_scratch(
        &mut self,
        shared: fused_prefill::SharedPrefillScratch,
    ) {
        self.prefill_scratch = Some(shared);
    }

    /// One fused gemv, split by the load-time row order: the packed
    /// conv input (q|k|v - the batched conv weight's channel order)
    /// and the raw gate span (consumed by finish_mixer after the
    /// recurrence).
    fn project_fused(
        &self,
        x: &candle_core::Tensor,
    ) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor)> {
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
    ) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)> {
        Ok((
            conv_out.narrow(1, 0, self.key_width)?.contiguous()?,
            conv_out.narrow(1, self.key_width, self.key_width)?.contiguous()?,
            conv_out
                .narrow(1, 2 * self.key_width, self.value_width)?
                .contiguous()?,
        ))
    }

    pub(crate) fn forward(&self, x: &candle_core::Tensor) -> LibQuestResult<candle_core::Tensor> {
        let mixed = self.mixer_stateless(&norms::rms_norm(x, &self.input_layernorm, self.rms_eps)?)?;
        self.close_halves(x, mixed, false)
    }

    pub(crate) fn forward_chunk(&mut self, x: &candle_core::Tensor) -> LibQuestResult<candle_core::Tensor> {
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
    ) -> LibQuestResult<candle_core::Tensor> {
        let h = (x + mixed)?;
        let mlp_in =
            norms::rms_norm_auto(&h, &self.post_attention_layernorm, self.rms_eps, fused)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        Ok((h + mlp_out)?)
    }

    /// Post-conv prep shared by both mixer paths: head reshape, the
    /// recurrence's f32 upcast discipline, l2 norms, the q scale, and
    /// the gating scalars from the hoisted a/b gemv rows (callers
    /// project them - the KernelAccept capture stashes the same rows).
    /// Returns (q, k, v, g, beta) rule-ready.
    #[allow(clippy::type_complexity)]
    fn heads_and_gates(
        &self,
        q: candle_core::Tensor,
        k: candle_core::Tensor,
        v: candle_core::Tensor,
        a_rows: &candle_core::Tensor,
        b_rows: &candle_core::Tensor,
        seq_len: usize,
    ) -> LibQuestResult<(
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
            a_rows,
            b_rows,
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
    ) -> LibQuestResult<candle_core::Tensor> {
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

    /// The GDN mixer, STATELESS sequential form (the StatelessCore
    /// parity instrument): zero-seeded per-token recurrence, no state
    /// kept.
    fn mixer_stateless(&self, x: &candle_core::Tensor) -> LibQuestResult<candle_core::Tensor> {
        let (seq_len, _) = x.dims2()?;
        let (conv_in, gate) = self.project_fused(x)?;
        let a_rows = self.a_proj.forward(x)?;
        let b_rows = self.b_proj.forward(x)?;
        let conv_out = gdn::causal_conv_silu(&conv_in, &self.conv_weight)?;
        let (q, k, v) = self.split_conv(&conv_out)?;
        let (q, k, v, g, beta) = self.heads_and_gates(q, k, v, &a_rows, &b_rows, seq_len)?;
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

    /// The GDN mixer, CARRIED form (the StateCarry engine path): conv
    /// tails prepend the chunk, the persisted state seeds the rule, and
    /// both advance. Decode (seq 1) rides the recurrent step; larger
    /// chunks ride the chunked rule - the upstream's own path split.
    /// The carried caches (state, conv tail) update IN PLACE via
    /// slice_set - never rebind - so their device addresses stay
    /// stable for the lifetime of the layer. Captured decode graphs
    /// bake those addresses; a rebind anywhere (this path, clear)
    /// would leave every cached graph reading dead memory.
    fn mixer_carried(&mut self, x: &candle_core::Tensor) -> LibQuestResult<candle_core::Tensor> {
        let (seq_len, _) = x.dims2()?;
        #[cfg(feature = "cuda")]
        if seq_len == 1
            && self.fused_gdn
            && x.device().is_cuda()
            && x.dtype() == candle_core::DType::BF16
            && self.decode_static.is_some()
        {
            return self.fused_decode_step(x);
        }
        #[cfg(feature = "cuda")]
        if seq_len > 1
            && self.fused_prefill
            && x.device().is_cuda()
            && x.dtype() == candle_core::DType::BF16
            && self.prefill_static.is_some()
        {
            return self.mixer_carried_fused_chunk(x, seq_len);
        }
        let (conv_in, gate) = self.project_fused(x)?;
        let a_rows = self.a_proj.forward(x)?;
        let b_rows = self.b_proj.forward(x)?;
        if seq_len > 1 && self.spec_capture {
            self.spec_rows = Some((conv_in.clone(), a_rows.clone(), b_rows.clone()));
        }
        let conv_out = self.conv_carried(&conv_in, seq_len)?;
        let (q, k, v) = self.split_conv(&conv_out)?;
        let (q, k, v, g, beta) = self.heads_and_gates(q, k, v, &a_rows, &b_rows, seq_len)?;
        let y = if seq_len == 1 {
            self.carried_decode_step(&q, &k, &v, &g, &beta)?
        } else {
            self.carried_chunk(&q, &k, &v, &g, &beta)?
        };
        self.finish_mixer(y, gate, seq_len, self.fused_gdn)
    }

    /// The PrepFusion chunk branch: one prep launch (the carried
    /// conv, l2 norms, and gates written bundle-direct) plus the rule
    /// kernels and the two cublas gemms replace the classic prep
    /// chain. The a/b gemvs stay cublas (the accumulation-order
    /// rule); the carried caches still update IN PLACE via slice_set
    /// (address-stable). The cpu path, the stateless parity form, and
    /// decode never come here.
    #[cfg(feature = "cuda")]
    fn mixer_carried_fused_chunk(
        &mut self,
        x: &candle_core::Tensor,
        seq_len: usize,
    ) -> LibQuestResult<candle_core::Tensor> {
        let qkvg = self.qkvg_proj.forward(x)?;
        let conv_width = 2 * self.key_width + self.value_width;
        let conv_in = qkvg.narrow(1, 0, conv_width)?;
        let gate = qkvg.narrow(1, conv_width, self.value_width)?.contiguous()?;
        let a_rows = self.a_proj.forward(x)?;
        let b_rows = self.b_proj.forward(x)?;
        if self.spec_capture {
            self.spec_rows = Some((conv_in.clone(), a_rows.clone(), b_rows.clone()));
        }
        let static_rows = self
            .prefill_static
            .as_ref()
            .expect("checked by the dispatch");
        let (y, next_state, new_tail) = fused_prefill::prep_chunk_rule(
            fused_prefill::PrepInputs {
                conv_in: &conv_in,
                a_rows: &a_rows,
                b_rows: &b_rows,
                static_rows,
                conv_tail: &self.conv_tail,
                state: &self.state,
            },
            (self.head_k_dim as f64).powf(-0.5),
            self.allow_neg_eigval,
            self.prefill_scratch.as_ref(),
        )?;
        self.state.slice_set(&next_state, 0, 0)?;
        self.conv_tail.slice_set(&new_tail, 0, 0)?;
        self.finish_mixer(y, gate, seq_len, self.fused_gdn)
    }

    /// The FusedHeadPrep decode step (t == 1, cuda bf16, fused_gdn):
    /// conv tap -> ONE head-prep launch (l2 norms + q scale + gates
    /// written straight into the packed operand) -> GdnFusedStep.
    /// The a/b gemvs stay cublas (the accumulation-order rule); the
    /// dyn row is their outputs catted with the packed static. The
    /// classic paths (cpu, non-bf16, fused off) never come here.
    #[cfg(feature = "cuda")]
    fn fused_decode_step(
        &mut self,
        x: &candle_core::Tensor,
    ) -> LibQuestResult<candle_core::Tensor> {
        let (conv_in, gate) = self.project_fused(x)?;
        let conv_out = self.conv_carried(&conv_in, 1)?;
        let a_row = self.a_proj.forward(x)?;
        let b_row = self.b_proj.forward(x)?;
        let statics = self
            .decode_static
            .as_ref()
            .expect("checked by the dispatch");
        let dyn_row =
            candle_core::Tensor::cat(&[&a_row, &b_row, statics], 1)?.contiguous()?;
        let packed = candle_core::Tensor::zeros(
            (self.num_heads, 2 * self.head_k_dim + self.head_v_dim + 2),
            candle_core::DType::F32,
            x.device(),
        )?;
        packed.inplace_op3(&conv_out, &dyn_row, &fused::HeadPrepFused {
            q_scale: (self.head_k_dim as f64).powf(-0.5) as f32,
            beta_scale: if self.allow_neg_eigval { 2.0 } else { 1.0 },
        })?;
        let y = candle_core::Tensor::zeros(
            (self.num_heads, self.head_v_dim),
            candle_core::DType::F32,
            x.device(),
        )?;
        self.state.inplace_op3(&packed, &y, &fused::GdnFusedStep)?;
        self.finish_mixer(y.unsqueeze(0)?, gate, 1, true)
    }

    /// One carried multi-token chunk on the CLASSIC chain (the
    /// PrefillDispatch kernel family dispatches earlier, in
    /// mixer_carried; cpu and non-bf16 grades stay here). Advances
    /// the persisted state IN PLACE (address-stable).
    fn carried_chunk(
        &mut self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
        g: &candle_core::Tensor,
        beta: &candle_core::Tensor,
    ) -> LibQuestResult<candle_core::Tensor> {
        let (out, next_state) = gdn::chunk_rule(q, k, v, g, beta, &self.state)?;
        self.state.slice_set(&next_state, 0, 0)?;
        Ok(out)
    }

    /// One carried decode step (t == 1) on the CLASSIC chain; the
    /// armed cuda bf16 path rides fused_decode_step (FusedHeadPrep)
    /// and never comes here. Returns the [1, heads, dv] f32 readout.
    fn carried_decode_step(
        &mut self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
        g: &candle_core::Tensor,
        beta: &candle_core::Tensor,
    ) -> LibQuestResult<candle_core::Tensor> {
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
    ) -> LibQuestResult<candle_core::Tensor> {
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

    /// The carried caches as snapshot handles (cheap clones sharing
    /// storage; the writer serializes them immediately): (state f32
    /// [heads, dk, dv], conv tail f32 [kernel-1, channels]).
    pub(crate) fn cache_snapshot(&self) -> (candle_core::Tensor, candle_core::Tensor) {
        (self.state.clone(), self.conv_tail.clone())
    }

    /// Overwrite the carried caches from restored tensors, in place
    /// (address-stable; see mixer_carried). Shapes and the f32
    /// discipline are validated here - the layer owns its geometry.
    pub(crate) fn restore_cache(
        &mut self,
        state: &candle_core::Tensor,
        conv_tail: &candle_core::Tensor,
    ) -> LibQuestResult<()> {
        snafu::ensure_whatever!(
            state.dims() == self.state.dims()
                && state.dtype() == candle_core::DType::F32,
            "gdn state restore wants f32 {:?}, got {:?} {:?}",
            self.state.dims(),
            state.dtype(),
            state.dims()
        );
        snafu::ensure_whatever!(
            conv_tail.dims() == self.conv_tail.dims()
                && conv_tail.dtype() == candle_core::DType::F32,
            "conv tail restore wants f32 {:?}, got {:?} {:?}",
            self.conv_tail.dims(),
            conv_tail.dtype(),
            conv_tail.dims()
        );
        self.state.slice_set(state, 0, 0)?;
        self.conv_tail.slice_set(conv_tail, 0, 0)?;
        Ok(())
    }

    /// Zero the carried caches in place (address-stable; see
    /// mixer_carried).
    pub(crate) fn clear_cache(&mut self) -> LibQuestResult<()> {
        self.state.slice_set(&self.state.zeros_like()?, 0, 0)?;
        self.conv_tail
            .slice_set(&self.conv_tail.zeros_like()?, 0, 0)?;
        Ok(())
    }

    /// SpeculationPort: snapshot the carried caches into the shadow
    /// buffers (lazily allocated fresh storage - slice_set refuses
    /// shared storage), drop any stale captured rows, and arm the
    /// KernelAccept capture for the coming verify chunk.
    pub(crate) fn shadow_save(&mut self) -> LibQuestResult<()> {
        self.spec_rows = None;
        self.spec_capture = true;
        if self.spec_shadow.is_none() {
            self.spec_shadow =
                Some((self.state.zeros_like()?, self.conv_tail.zeros_like()?));
        }
        let (state_shadow, tail_shadow) =
            self.spec_shadow.as_ref().expect("just ensured");
        state_shadow.slice_set(&self.state, 0, 0)?;
        tail_shadow.slice_set(&self.conv_tail, 0, 0)?;
        Ok(())
    }

    /// SpeculationPort: restore the carried caches from the shadow
    /// buffers, in place (address-stable like every cache write).
    pub(crate) fn shadow_restore(&mut self) -> LibQuestResult<()> {
        let Some((state_shadow, tail_shadow)) = &self.spec_shadow else {
            snafu::whatever!("shadow restore without a saved shadow");
        };
        self.state.slice_set(state_shadow, 0, 0)?;
        self.conv_tail.slice_set(tail_shadow, 0, 0)?;
        Ok(())
    }

    /// KernelAccept: drop the captured rule inputs and disarm the
    /// capture (a full accept or a full rewind ends the round).
    pub(crate) fn spec_release(&mut self) {
        self.spec_rows = None;
        self.spec_capture = false;
    }

    /// KernelAccept: re-advance the carried caches over the first
    /// `accepted` captured rule-input rows from the SHADOW-RESTORED
    /// state and tail (the caller runs shadow_restore first) - pure
    /// recurrence math over cached post-projection operands, no
    /// projections and no weight traffic. Rides the verify chunk's
    /// own grade dispatch (the fused prefill pipeline on armed cuda
    /// bf16, the classic chain otherwise); y/gate/finish are skipped
    /// entirely - only the caches matter.
    pub(crate) fn spec_readvance(&mut self, accepted: usize) -> LibQuestResult<()> {
        let Some((conv_in, a_rows, b_rows)) = self.spec_rows.take() else {
            snafu::whatever!("spec re-advance without captured rule inputs");
        };
        self.spec_capture = false;
        let span = conv_in.dim(0)?;
        snafu::ensure_whatever!(
            (1..=span).contains(&accepted),
            "spec re-advance wants 1..={span} accepted rows, got {accepted}"
        );
        let conv_in = conv_in.narrow(0, 0, accepted)?;
        let a_rows = a_rows.narrow(0, 0, accepted)?;
        let b_rows = b_rows.narrow(0, 0, accepted)?;
        #[cfg(feature = "cuda")]
        if self.fused_prefill
            && conv_in.device().is_cuda()
            && conv_in.dtype() == candle_core::DType::BF16
            && let Some(static_rows) = &self.prefill_static
        {
            let (_, next_state, new_tail) = fused_prefill::prep_chunk_rule(
                fused_prefill::PrepInputs {
                    conv_in: &conv_in,
                    a_rows: &a_rows,
                    b_rows: &b_rows,
                    static_rows,
                    conv_tail: &self.conv_tail,
                    state: &self.state,
                },
                (self.head_k_dim as f64).powf(-0.5),
                self.allow_neg_eigval,
                // A readvance span (<= 16 rows) is never a full
                // span; keep it off the pool for clarity.
                None,
            )?;
            self.state.slice_set(&next_state, 0, 0)?;
            self.conv_tail.slice_set(&new_tail, 0, 0)?;
            return Ok(());
        }
        let (conv_out, new_tail) =
            gdn::conv_with_tail(&conv_in, &self.conv_weight, &self.conv_tail)?;
        let (q, k, v) = self.split_conv(&conv_out)?;
        let (q, k, v, g, beta) = self.heads_and_gates(q, k, v, &a_rows, &b_rows, accepted)?;
        let (_, next_state) = gdn::chunk_rule(&q, &k, &v, &g, &beta, &self.state)?;
        self.state.slice_set(&next_state, 0, 0)?;
        self.conv_tail.slice_set(&new_tail, 0, 0)?;
        Ok(())
    }
}
