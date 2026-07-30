//! The full-attention decoder layer, fully POST-norm, with its KV store.
use crate::*;

/// The reserve grain for the carried KV buffers.
pub(crate) const KV_RESERVE_STEP: usize = 1024;

/// Carried attention KV: capacity-reserved buffers plus the live length.
pub(crate) struct AttnKv {
    pub(crate) k: candle_core::Tensor,
    pub(crate) v: candle_core::Tensor,
    pub(crate) len: usize,
}

/// A full-attention decoder layer, post-norm on both halves.
pub(crate) struct AttnLayer {
    post_attention_layernorm: candle_core::Tensor,
    post_feedforward_layernorm: candle_core::Tensor,
    /// q/k/v row-fused at load; o stays separate, taking a different input.
    qkv_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: candle_core::Tensor,
    k_norm: candle_core::Tensor,
    mlp: Mlp,
    pub(crate) num_heads: usize,
    head_dim: usize,
    rms_eps: f64,
    /// Drives the flash prefill dispatch, under that feature alone.
    #[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
    use_flash_attn: bool,
    /// Fusion armed: decode norms ride the fused kernel on cuda.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fused_gdn: bool,
    pub(crate) kv: Option<AttnKv>,
    /// A captured graph baked these buffers, so replacing them PARKS them.
    pub(crate) buffers_captured: bool,
    /// Latched by a reserve that parked buffers; the model collects it.
    parked_graphs: bool,
    /// Eviction armed at construction.
    evict: bool,
    /// The armed observation window, or the default when unarmed.
    score_tail: usize,
    /// Query rows per scoring matmul, or the default when unarmed.
    score_slice: usize,
    /// Last-pass scores, reserved and parked in lockstep with the KV.
    pub(crate) scores: Option<candle_core::Tensor>,
    /// A persistent zero row: the self-slot reset's stable source.
    score_zero: Option<candle_core::Tensor>,
    /// The density accumulator, Some only while a profile is armed.
    #[cfg(feature = "attn-profile")]
    pub(crate) profile: std::cell::RefCell<Option<profile::ProfileAccum>>,
}

impl AttnLayer {
    pub(crate) fn new(
        config: &OlmoHybridConfig,
        use_flash_attn: bool,
        eviction: Option<&EvictionSettings>,
        fused_gdn: bool,
        vb: candle_nn::VarBuilder,
    ) -> LibQuestResult<Self> {
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
            evict: eviction.is_some(),
            score_tail: eviction.map_or(evict::SCORE_TAIL, |settings| settings.score_tail),
            score_slice: eviction.map_or(evict::SCORE_SLICE, |settings| settings.score_slice),
            scores: None,
            score_zero: None,
            #[cfg(feature = "attn-profile")]
            profile: std::cell::RefCell::new(None),
        })
    }

    /// Only the GraphDecode staging reads it (cuda builds).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn kv_len(&self) -> usize {
        self.kv.as_ref().map_or(0, |kv| kv.len)
    }

    pub(crate) fn kv_capacity(&self) -> usize {
        self.kv
            .as_ref()
            .and_then(|kv| kv.k.dims().get(1).copied())
            .unwrap_or(0)
    }

    /// Host bookkeeping for a staged step; the device write is done.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn advance_len(&mut self) {
        if let Some(kv) = &mut self.kv {
            kv.len += 1;
        }
    }

    /// Read-and-clear the park latch.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn take_parked(&mut self) -> bool {
        std::mem::take(&mut self.parked_graphs)
    }

    pub(crate) fn forward(
        &self,
        x: &candle_core::Tensor,
        mask: &candle_core::Tensor,
    ) -> LibQuestResult<candle_core::Tensor> {
        let (q, k, v) = self.qkv(x, false)?;
        let attended = self.attend(&q, &k, &v, Some(mask))?;
        self.close_halves(x, attended, false)
    }

    pub(crate) fn forward_chunk(
        &mut self,
        x: &candle_core::Tensor,
        mask: Option<&candle_core::Tensor>,
    ) -> LibQuestResult<candle_core::Tensor> {
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

    /// The armed last-pass scoring after a carried forward.
    fn score_pass(&mut self, q: &candle_core::Tensor, added: usize) -> LibQuestResult<()> {
        let Some(kv) = &self.kv else {
            return Ok(());
        };
        let len = kv.len;
        let scale = (self.head_dim as f64).powf(-0.5);
        let tail = if added > 1 {
            self.score_tail.min(added)
        } else {
            1
        };
        let q_tail = q.narrow(1, added - tail, tail)?;
        let scores_new =
            evict::last_pass_scores(&q_tail, &kv.k.narrow(1, 0, len)?, scale, self.score_slice)?;
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

    /// Compact the store to the per-head keep-sets, optionally shrinking.
    pub(crate) fn compact(
        &mut self,
        keep: &[Vec<u32>],
        shrink_to: Option<usize>,
    ) -> LibQuestResult<()> {
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

    /// Ensure the KV buffers hold `added` more rows, growing by grain.
    fn reserve(
        &mut self,
        added: usize,
        template: &candle_core::Tensor,
    ) -> LibQuestResult<()> {
        let needed = self.kv.as_ref().map_or(added, |kv| kv.len + added);
        if self.kv_capacity() >= needed.max(1) && (!self.evict || self.scores.is_some()) {
            return Ok(());
        }
        let capacity = needed.div_ceil(KV_RESERVE_STEP) * KV_RESERVE_STEP;
        self.reserve_capacity(capacity, template.dtype(), template.device())
    }

    /// Ensure at least `capacity` rows; captured buffers park, never free.
    pub(crate) fn reserve_capacity(
        &mut self,
        capacity: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> LibQuestResult<()> {
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

    /// The post-norm block's residual adds, shared by both paths.
    fn close_halves(
        &self,
        x: &candle_core::Tensor,
        attended: candle_core::Tensor,
        fused: bool,
    ) -> LibQuestResult<candle_core::Tensor> {
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

    /// One fused gemv split to q/k/v, normed before the head reshape.
    fn qkv(
        &self,
        x: &candle_core::Tensor,
        fused: bool,
    ) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)> {
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
        let heads = |t: candle_core::Tensor| -> LibQuestResult<candle_core::Tensor> {
            Ok(t.reshape((seq_len, self.num_heads, self.head_dim))?
                .transpose(0, 1)?
                .contiguous()?)
        };
        Ok((heads(q)?, heads(k)?, heads(v)?))
    }

    /// Eager MHA with no positional encoding; decode runs mask-free.
    fn attend(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
        mask: Option<&candle_core::Tensor>,
    ) -> LibQuestResult<candle_core::Tensor> {
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
        snafu::ensure_whatever!(
            seq_len == 1 || mask.is_some(),
            "eager multi-token attention without a mask (the flash dispatch is the only \
             mask-free prefill; a FlashMaskSkip leak must never attend full-visibility)"
        );
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

    /// The profile-armed attention: the same eager math, row-sliced.
    #[cfg(feature = "attn-profile")]
    fn attend_profiled(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
        mask: Option<&candle_core::Tensor>,
    ) -> LibQuestResult<candle_core::Tensor> {
        let seq_len = q.dim(1)?;
        snafu::ensure_whatever!(
            seq_len == 1 || mask.is_some(),
            "profiled multi-token attention without a mask (the profile arm keeps the \
             mask build on - a skip reaching here is a FlashMaskSkip leak)"
        );
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

    /// The flash prefill, causal by the kernel's bottom-right alignment.
    #[cfg(feature = "flash-attn")]
    fn attend_flash(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> LibQuestResult<candle_core::Tensor> {
        let seq_len = q.dim(1)?;
        let block = |t: &candle_core::Tensor| -> LibQuestResult<candle_core::Tensor> {
            Ok(t.transpose(0, 1)?.unsqueeze(0)?.contiguous()?)
        };
        let span = |t: &candle_core::Tensor| -> LibQuestResult<candle_core::Tensor> {
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

    /// Restore the carried KV from snapshot rows, park-aware.
    pub(crate) fn restore_kv(
        &mut self,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> LibQuestResult<usize> {
        let (heads, len, head_dim) = k.dims3()?;
        snafu::ensure_whatever!(
            heads == self.num_heads && head_dim == self.head_dim,
            "kv restore wants [{}, len, {}], got {:?}",
            self.num_heads,
            self.head_dim,
            k.dims()
        );
        snafu::ensure_whatever!(
            v.dims() == k.dims() && v.dtype() == k.dtype(),
            "kv restore k/v disagree: {:?} {:?} vs {:?} {:?}",
            k.dims(),
            k.dtype(),
            v.dims(),
            v.dtype()
        );
        let capacity = len.div_ceil(KV_RESERVE_STEP) * KV_RESERVE_STEP;
        self.reserve_capacity(capacity, k.dtype(), k.device())?;
        let kv = self.kv.as_mut().expect("reserved");
        kv.k.slice_set(k, 1, 0)?;
        kv.v.slice_set(v, 1, 0)?;
        kv.len = len;
        Ok(len)
    }

    /// Length resets; the reserved capacity stays.
    pub(crate) fn clear_cache(&mut self) {
        if let Some(kv) = &mut self.kv {
            kv.len = 0;
        }
    }

    /// The staged decode step: the classic math over a static bucket.
    #[cfg(feature = "cuda")]
    pub(crate) fn forward_graph(
        &self,
        x: &candle_core::Tensor,
        stage: &graph::DecodeStage,
    ) -> LibQuestResult<candle_core::Tensor> {
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
