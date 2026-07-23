use crate::*;

/// Olmo 3 decoder: Olmo 2 plus sliding-window attention on 3/4 of the
/// layers and YaRN RoPE on the full-attention layers only. Deliberately
/// generic baseline: every layer keeps its full KV history (no trimming);
/// the window is enforced by mask on multi-token forwards and by slicing
/// the cached view on single-token decode.
#[derive(Debug, Clone)]
pub(crate) struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: candle_nn::RmsNorm,
    lm_head: candle_nn::Linear,
    sliding_window: usize,
    max_position_embeddings: usize,
    device: candle_core::Device,
    dtype: candle_core::DType,
}

impl Model {
    pub(crate) fn new(cfg: &Olmo3Config, vb: candle_nn::VarBuilder) -> CompatResult<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let full_rope = std::sync::Arc::new(RopeTables::for_full_layers(cfg, vb.dtype(), vb_m.device())?);
        let sliding_rope = std::sync::Arc::new(RopeTables::vanilla(
            cfg.head_dim(),
            cfg.rope_theta,
            cfg.max_position_embeddings,
            vb.dtype(),
            vb_m.device(),
        )?);
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_type = cfg.layer_type(layer_idx);
            let rotary = match layer_type {
                LayerType::Full => full_rope.clone(),
                LayerType::Sliding => sliding_rope.clone(),
            };
            layers.push(DecoderLayer::new(rotary, layer_type, cfg, vb_l.pp(layer_idx))?);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb_m.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Logits for the LAST position only, shape (batch, 1, vocab).
    pub(crate) fn forward(&mut self, input_ids: &candle_core::Tensor, seqlen_offset: usize) -> CompatResult<candle_core::Tensor> {
        self.forward_inner(input_ids, seqlen_offset, false)
    }

    /// Logits for EVERY input position, shape (batch, seq, vocab); the
    /// dump/verify path (costs seq * vocab activation memory).
    pub(crate) fn forward_all(&mut self, input_ids: &candle_core::Tensor, seqlen_offset: usize) -> CompatResult<candle_core::Tensor> {
        self.forward_inner(input_ids, seqlen_offset, true)
    }

    fn forward_inner(
        &mut self,
        input_ids: &candle_core::Tensor,
        seqlen_offset: usize,
        all_positions: bool,
    ) -> CompatResult<candle_core::Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        snafu::ensure_whatever!(
            seqlen_offset + seq_len <= self.max_position_embeddings,
            "context length {} exceeds max_position_embeddings {}",
            seqlen_offset + seq_len,
            self.max_position_embeddings
        );
        let (full_mask, sliding_mask) = if seq_len <= 1 {
            (None, None)
        } else {
            (
                Some(self.mask(b_size, seqlen_offset, seq_len, None)?),
                Some(self.mask(b_size, seqlen_offset, seq_len, Some(self.sliding_window))?),
            )
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            let mask = match layer.layer_type {
                LayerType::Full => full_mask.as_ref(),
                LayerType::Sliding => sliding_mask.as_ref(),
            };
            xs = layer.forward(&xs, mask, seqlen_offset)?;
        }
        let xs = if all_positions {
            xs
        } else {
            xs.narrow(1, seq_len - 1, 1)?
        };
        Ok(xs.apply(&self.norm)?.apply(&self.lm_head)?)
    }

    /// Unused by the one-shot bin; kept as olmo2-parity API for reuse.
    #[allow(dead_code)]
    pub(crate) fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    fn mask(
        &self,
        b_size: usize,
        offset: usize,
        q_len: usize,
        window: Option<usize>,
    ) -> CompatResult<candle_core::Tensor> {
        let total = offset + q_len;
        let values = banded_mask_values(offset, q_len, window);
        let mask = candle_core::Tensor::from_vec(values, (q_len, total), &self.device)?;
        Ok(mask.expand((b_size, 1, q_len, total))?.to_dtype(self.dtype)?)
    }
}

/// Row-major 0.0 / -inf visibility band. Columns are absolute kv positions
/// 0..offset+q_len; row i is the query at absolute position offset+i. A
/// window w additionally hides keys more than w-1 positions back.
pub(crate) fn banded_mask_values(offset: usize, q_len: usize, window: Option<usize>) -> Vec<f32> {
    let total = offset + q_len;
    let mut values = Vec::with_capacity(q_len * total);
    for i in 0..q_len {
        let pos = offset + i;
        for j in 0..total {
            let causal = j <= pos;
            let within_window = window.is_none_or(|w| j + w > pos);
            values.push(if causal && within_window { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    values
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    post_attention_layernorm: candle_nn::RmsNorm,
    post_feedforward_layernorm: candle_nn::RmsNorm,
    layer_type: LayerType,
}

impl DecoderLayer {
    fn new(
        rotary: std::sync::Arc<RopeTables>,
        layer_type: LayerType,
        cfg: &Olmo3Config,
        vb: candle_nn::VarBuilder,
    ) -> CompatResult<Self> {
        let sliding_window = match layer_type {
            LayerType::Sliding => Some(cfg.sliding_window),
            LayerType::Full => None,
        };
        let self_attn = Attention::new(rotary, sliding_window, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let post_attention_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let post_feedforward_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_feedforward_layernorm"))?;
        Ok(Self {
            self_attn,
            mlp,
            post_attention_layernorm,
            post_feedforward_layernorm,
            layer_type,
        })
    }

    /// Post-norm block: norm(sublayer(x)) + x, on both halves.
    fn forward(
        &mut self,
        xs: &candle_core::Tensor,
        attention_mask: Option<&candle_core::Tensor>,
        seqlen_offset: usize,
    ) -> CompatResult<candle_core::Tensor> {
        let residual = xs;
        let xs = self.self_attn.forward(xs, attention_mask, seqlen_offset)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let mlp_out = self.mlp.forward(&xs)?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        Ok((residual + mlp_out)?)
    }

    #[allow(dead_code)]
    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    rotary: std::sync::Arc<RopeTables>,
    sliding_window: Option<usize>,
    kv_cache: Option<(candle_core::Tensor, candle_core::Tensor)>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Attention {
    fn new(
        rotary: std::sync::Arc<RopeTables>,
        sliding_window: Option<usize>,
        cfg: &Olmo3Config,
        vb: candle_nn::VarBuilder,
    ) -> CompatResult<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_kv_heads();
        let head_dim = cfg.head_dim();
        let bias = cfg.attention_bias;
        let q_proj = candle_nn::linear_b(cfg.hidden_size, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_b(cfg.hidden_size, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_b(cfg.hidden_size, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_b(num_heads * head_dim, cfg.hidden_size, bias, vb.pp("o_proj"))?;
        let q_norm = candle_nn::rms_norm(num_heads * head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(num_kv_heads * head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            sliding_window,
            kv_cache: None,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(
        &mut self,
        xs: &candle_core::Tensor,
        attention_mask: Option<&candle_core::Tensor>,
        seqlen_offset: usize,
    ) -> CompatResult<candle_core::Tensor> {
        let (b_size, q_len, _) = xs.dims3()?;
        let dtype = xs.dtype();

        // q/k RMSNorm over the full projection width, before head reshape.
        let query_states = self.q_norm.forward(&self.q_proj.forward(xs)?)?;
        let key_states = self.k_norm.forward(&self.k_proj.forward(xs)?)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_size, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_size, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_size, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = self.rotary.apply(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => (
                candle_core::Tensor::cat(&[prev_k, &key_states], 2)?,
                candle_core::Tensor::cat(&[prev_v, &value_states], 2)?,
            ),
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // Single-token decode on a sliding layer: the visible set is exactly
        // the last `window` cached positions, so slice instead of masking.
        let kv_len = key_states.dim(2)?;
        let (key_states, value_states) = match self.sliding_window {
            Some(window) if q_len == 1 && kv_len > window => (
                key_states.narrow(2, kv_len - window, window)?,
                value_states.narrow(2, kv_len - window, window)?,
            ),
            _ => (key_states, value_states),
        };

        let key_states = r::candle::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = r::candle::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let mut attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }
        // HF computes softmax in f32 and casts back; match it for parity.
        let attn_weights = if dtype == candle_core::DType::F32 {
            candle_nn::ops::softmax_last_dim(&attn_weights)?
        } else {
            candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(candle_core::DType::F32)?)?.to_dtype(dtype)?
        };
        let attn_output = attn_weights.matmul(&value_states)?;
        Ok(attn_output
            .transpose(1, 2)?
            .reshape((b_size, q_len, self.hidden_size))?
            .apply(&self.o_proj)?)
    }

    #[allow(dead_code)]
    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Olmo3Config, vb: candle_nn::VarBuilder) -> CompatResult<Self> {
        let gate_proj = candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    /// SwiGLU: down(silu(gate(x)) * up(x)).
    fn forward(&self, xs: &candle_core::Tensor) -> CompatResult<candle_core::Tensor> {
        let gated = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let up = xs.apply(&self.up_proj)?;
        Ok((gated * up)?.apply(&self.down_proj)?)
    }
}
