use crate::*;

/// Runtime knobs the model is constructed with; a container so the API can
/// grow without signature churn. use_flash_attn is wired by D6 (slice 2)
/// and inert in the eager core.
#[derive(Debug, Clone, Copy, Default)]
pub struct Settings {
    pub use_flash_attn: bool,
}

/// Why visibility is limited (or is not), so flash dispatch (D6) can pick
/// the fused path: None = single-position decode with nothing to hide
/// (D4); Causal = plain causality - including an in-window sliding
/// prefill, which degenerates to it; Window = the sliding band. The
/// tensor is present only when an eager consumer will read it - a
/// flash-active forward carries tensor-less descriptors (the kernels take
/// the semantics as parameters), skipping the per-chunk mask build
/// entirely.
#[derive(Debug, Clone)]
pub(crate) enum AttnMask {
    None,
    Causal(Option<candle_core::Tensor>),
    Window(Option<candle_core::Tensor>),
}

impl AttnMask {
    /// The eager path's read: kinds that limit visibility MUST carry a
    /// tensor there; reaching a descriptor without one is a dispatch bug,
    /// never a silent full-visibility pass.
    fn eager_tensor(&self) -> AlmostResult<Option<&candle_core::Tensor>> {
        match self {
            AttnMask::None => Ok(None),
            AttnMask::Causal(Some(mask)) | AttnMask::Window(Some(mask)) => Ok(Some(mask)),
            AttnMask::Causal(None) | AttnMask::Window(None) => {
                snafu::whatever!("eager attention reached a tensor-less mask descriptor")
            }
        }
    }
}

/// Row-major 0.0 / -inf visibility band. Columns are absolute kv positions
/// kv_start..offset+q_len; row i is the query at absolute position
/// offset+i. A window w additionally hides keys more than w-1 positions
/// back. kv_start > 0 models a trimmed sliding cache (D3), whose first
/// retained key sits at absolute position offset - min(offset, w-1).
pub(crate) fn banded_mask_values(
    offset: usize,
    q_len: usize,
    kv_start: usize,
    window: Option<usize>,
) -> Vec<f32> {
    let total = offset + q_len - kv_start;
    let mut values = Vec::with_capacity(q_len * total);
    for i in 0..q_len {
        let pos = offset + i;
        for j in 0..total {
            let key = kv_start + j;
            let causal = key <= pos;
            let within_window = window.is_none_or(|w| key + w > pos);
            values.push(if causal && within_window { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    values
}

/// D3 retention bounds: the (start, length) narrow a sliding cache keeps
/// after growing to kv_len entries; None when everything already fits. A
/// decoding query needs at most window-1 past keys, so window-1 is the cap.
pub(crate) fn sliding_trim_bounds(kv_len: usize, window: usize) -> Option<(usize, usize)> {
    let cap = window - 1;
    if kv_len > cap {
        Some((kv_len - cap, cap))
    } else {
        None
    }
}

/// The almost Olmo 3 decoder (D1): HF-shaped single-target module, MHA
/// only by policy (the sole checkpoint is MHA; kv-group machinery is
/// deliberately absent). Sliding layers persist at most window-1 cached
/// entries (D3); decode builds no masks on any layer (D4).
#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: candle_nn::RmsNorm,
    lm_head: candle_nn::Linear,
    sliding_window: usize,
    max_position_embeddings: usize,
    device: candle_core::Device,
    dtype: candle_core::DType,
    settings: Settings,
}

impl Model {
    pub fn new(cfg: &Olmo3Config, settings: Settings, vb: candle_nn::VarBuilder) -> AlmostResult<Self> {
        snafu::ensure_whatever!(
            cfg.num_kv_heads() == cfg.num_attention_heads,
            "almost supports MHA only (sole-checkpoint policy): num_key_value_heads {} != num_attention_heads {}",
            cfg.num_kv_heads(),
            cfg.num_attention_heads
        );
        if settings.use_flash_attn {
            snafu::ensure_whatever!(
                cfg!(feature = "flash-attn"),
                "this build carries no flash-attn support (rebuild with --features flash-attn)"
            );
            snafu::ensure_whatever!(
                vb.device().is_cuda(),
                "flash attention requires a cuda device"
            );
            snafu::ensure_whatever!(
                matches!(vb.dtype(), candle_core::DType::BF16 | candle_core::DType::F16),
                "flash attention requires bf16 or f16, got {:?}",
                vb.dtype()
            );
        }
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
            layers.push(DecoderLayer::new(rotary, layer_type, cfg, settings.use_flash_attn, vb_l.pp(layer_idx))?);
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
            settings,
        })
    }

    /// Logits for the LAST position only, shape (batch, 1, vocab).
    pub fn forward(&mut self, input_ids: &candle_core::Tensor, seqlen_offset: usize) -> AlmostResult<candle_core::Tensor> {
        self.forward_inner(input_ids, seqlen_offset, false)
    }

    /// Logits for EVERY input position, shape (batch, seq, vocab); the
    /// dump/verify path (costs seq * vocab activation memory).
    pub fn forward_all(&mut self, input_ids: &candle_core::Tensor, seqlen_offset: usize) -> AlmostResult<candle_core::Tensor> {
        self.forward_inner(input_ids, seqlen_offset, true)
    }

    fn forward_inner(
        &mut self,
        input_ids: &candle_core::Tensor,
        seqlen_offset: usize,
        all_positions: bool,
    ) -> AlmostResult<candle_core::Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        snafu::ensure_whatever!(
            seqlen_offset + seq_len <= self.max_position_embeddings,
            "context length {} exceeds max_position_embeddings {}",
            seqlen_offset + seq_len,
            self.max_position_embeddings
        );
        let flash_active = self.settings.use_flash_attn && seq_len > 1;
        let (full_mask, sliding_mask) = self.prefill_masks(b_size, seqlen_offset, seq_len, flash_active)?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            let mask = match layer.layer_type {
                LayerType::Full => &full_mask,
                LayerType::Sliding => &sliding_mask,
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

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    /// Largest sliding-layer cache length currently held; the T4 invariant
    /// pins it at window-1 or less after any forward.
    pub(crate) fn max_sliding_cache_len(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.layer_type == LayerType::Sliding)
            .map(|layer| layer.self_attn.cached_len())
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn device(&self) -> &candle_core::Device {
        &self.device
    }

    pub(crate) fn flash_enabled(&self) -> bool {
        self.settings.use_flash_attn
    }

    /// D4: decode (seq_len <= 1) builds no masks on any layer; masks exist
    /// only on prefill forwards, once per type. The sliding mask reuses the
    /// causal tensor when the whole span sits inside the window (D5). A
    /// flash-active forward gets tensor-less descriptors - the kernels
    /// carry the semantics, so no mask values are built or uploaded.
    fn prefill_masks(
        &self,
        b_size: usize,
        offset: usize,
        seq_len: usize,
        flash_active: bool,
    ) -> AlmostResult<(AttnMask, AttnMask)> {
        if seq_len <= 1 {
            return Ok((AttnMask::None, AttnMask::None));
        }
        let in_window = offset + seq_len <= self.sliding_window;
        if flash_active {
            // Sliding layers stay on the WINDOWED kernel even in-window
            // (identical visibility): mixing kernel families across chunk
            // boundaries lets per-layer bf16 accumulation differences
            // compound through the depth - measured 1.3e-2 nmse on the
            // long battery's chunked-vs-single when chunk A ran causal
            // against a windowed single reference.
            return Ok((AttnMask::Causal(None), AttnMask::Window(None)));
        }
        let causal = self.mask_tensor(b_size, offset, seq_len, 0, None)?;
        let sliding = if in_window {
            AttnMask::Causal(Some(causal.clone()))
        } else {
            let kv_start = offset - offset.min(self.sliding_window - 1);
            AttnMask::Window(Some(
                self.mask_tensor(b_size, offset, seq_len, kv_start, Some(self.sliding_window))?,
            ))
        };
        Ok((AttnMask::Causal(Some(causal)), sliding))
    }

    fn mask_tensor(
        &self,
        b_size: usize,
        offset: usize,
        q_len: usize,
        kv_start: usize,
        window: Option<usize>,
    ) -> AlmostResult<candle_core::Tensor> {
        let total = offset + q_len - kv_start;
        let values = banded_mask_values(offset, q_len, kv_start, window);
        let mask = candle_core::Tensor::from_vec(values, (q_len, total), &self.device)?;
        Ok(mask.expand((b_size, 1, q_len, total))?.to_dtype(self.dtype)?)
    }
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
        use_flash_attn: bool,
        vb: candle_nn::VarBuilder,
    ) -> AlmostResult<Self> {
        let sliding_window = match layer_type {
            LayerType::Sliding => Some(cfg.sliding_window),
            LayerType::Full => None,
        };
        let self_attn = Attention::new(rotary, sliding_window, cfg, use_flash_attn, vb.pp("self_attn"))?;
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
        attention_mask: &AttnMask,
        seqlen_offset: usize,
    ) -> AlmostResult<candle_core::Tensor> {
        let residual = xs;
        let xs = self.self_attn.forward(xs, attention_mask, seqlen_offset)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let mlp_out = self.mlp.forward(&xs)?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        Ok((residual + mlp_out)?)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Full-attention caches reserve capacity in coarse steps so context
/// growth costs a handful of allocations, not one realloc-plus-copy per
/// prefill chunk - the raw-cudaMalloc fragmentation that pattern causes is
/// what OOMs an otherwise-fitting 32K run on Tier A.
const FULL_CACHE_STEP: usize = 8192;

/// D6 flash dispatch: (b, seq, heads, head_dim) operands, softmax computed
/// in-kernel (f32 accumulators), causal masking bottom-right aligned so a
/// q block that is the tail of kv gets absolute-position causality (the
/// cached/chunked-prefill shape; R1-probed on candle 0.11).
#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    softmax_scale: f32,
    causal: bool,
) -> AlmostResult<candle_core::Tensor> {
    Ok(r::flash::flash_attn(q, k, v, softmax_scale, causal)?)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(
    _q: &candle_core::Tensor,
    _k: &candle_core::Tensor,
    _v: &candle_core::Tensor,
    _softmax_scale: f32,
    _causal: bool,
) -> AlmostResult<candle_core::Tensor> {
    snafu::whatever!("this build carries no flash-attn support (rebuild with --features flash-attn)")
}

/// The sliding band as a fused kernel: local attention with
/// window_size_left = window-1 and window_size_right = 0, bottom-right
/// aligned like the causal case, so a trimmed-cache q block still gets
/// absolute [p-w+1, p] visibility (probed).
#[cfg(feature = "flash-attn")]
fn flash_attn_windowed(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    softmax_scale: f32,
    window: usize,
) -> AlmostResult<candle_core::Tensor> {
    Ok(r::flash::flash_attn_windowed(
        q,
        k,
        v,
        softmax_scale,
        Some(window - 1),
        Some(0),
    )?)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn_windowed(
    _q: &candle_core::Tensor,
    _k: &candle_core::Tensor,
    _v: &candle_core::Tensor,
    _softmax_scale: f32,
    _window: usize,
) -> AlmostResult<candle_core::Tensor> {
    snafu::whatever!("this build carries no flash-attn support (rebuild with --features flash-attn)")
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
    use_flash_attn: bool,
    kv_cache: Option<(candle_core::Tensor, candle_core::Tensor)>,
    full_key_buffer: Option<candle_core::Tensor>,
    full_value_buffer: Option<candle_core::Tensor>,
    full_cache_len: usize,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Attention {
    fn new(
        rotary: std::sync::Arc<RopeTables>,
        sliding_window: Option<usize>,
        cfg: &Olmo3Config,
        use_flash_attn: bool,
        vb: candle_nn::VarBuilder,
    ) -> AlmostResult<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();
        let bias = cfg.attention_bias;
        let q_proj = candle_nn::linear_b(cfg.hidden_size, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_b(cfg.hidden_size, num_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_b(cfg.hidden_size, num_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_b(num_heads * head_dim, cfg.hidden_size, bias, vb.pp("o_proj"))?;
        let q_norm = candle_nn::rms_norm(num_heads * head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(num_heads * head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            sliding_window,
            use_flash_attn,
            kv_cache: None,
            full_key_buffer: None,
            full_value_buffer: None,
            full_cache_len: 0,
            num_heads,
            head_dim,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Append rotated k/v into the capacity-stepped full-layer buffers and
    /// return the live (b, h, len, dim) views. Append-only cat semantics;
    /// only the allocation granularity is coarse (FULL_CACHE_STEP).
    fn full_append(
        &mut self,
        key_states: &candle_core::Tensor,
        value_states: &candle_core::Tensor,
    ) -> AlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
        // slice_set requires contiguous sources; k is post-rope contiguous
        // already, v arrives as a transposed view.
        let key_states = key_states.contiguous()?;
        let value_states = value_states.contiguous()?;
        let (b_size, heads, seq_len, head_dim) = key_states.dims4()?;
        let needed = self.full_cache_len + seq_len;
        let capacity = match &self.full_key_buffer {
            Some(buffer) => buffer.dims()[2],
            None => 0,
        };
        if needed > capacity {
            let new_capacity = needed.div_ceil(FULL_CACHE_STEP) * FULL_CACHE_STEP;
            let shape = (b_size, heads, new_capacity, head_dim);
            let new_key = candle_core::Tensor::zeros(shape, key_states.dtype(), key_states.device())?;
            let new_value = candle_core::Tensor::zeros(shape, key_states.dtype(), key_states.device())?;
            if self.full_cache_len > 0
                && let (Some(old_key), Some(old_value)) = (&self.full_key_buffer, &self.full_value_buffer)
            {
                new_key.slice_set(&old_key.narrow(2, 0, self.full_cache_len)?.contiguous()?, 2, 0)?;
                new_value.slice_set(&old_value.narrow(2, 0, self.full_cache_len)?.contiguous()?, 2, 0)?;
            }
            self.full_key_buffer = Some(new_key);
            self.full_value_buffer = Some(new_value);
        }
        let (Some(key_buffer), Some(value_buffer)) = (&self.full_key_buffer, &self.full_value_buffer) else {
            snafu::whatever!("full-layer cache buffers absent after reserve");
        };
        key_buffer.slice_set(&key_states, 2, self.full_cache_len)?;
        value_buffer.slice_set(&value_states, 2, self.full_cache_len)?;
        self.full_cache_len = needed;
        Ok((
            key_buffer.narrow(2, 0, needed)?,
            value_buffer.narrow(2, 0, needed)?,
        ))
    }

    fn forward(
        &mut self,
        xs: &candle_core::Tensor,
        attention_mask: &AttnMask,
        seqlen_offset: usize,
    ) -> AlmostResult<candle_core::Tensor> {
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
            .reshape((b_size, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_size, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = self.rotary.apply(&query_states, &key_states, seqlen_offset)?;

        // Attention always runs on the pre-trim concatenation; the mask
        // (prefill) or the cache bound itself (decode) handles visibility.
        let (key_states, value_states) = match self.sliding_window {
            // D3: sliding layers cat then persist only the last window-1
            // entries; a decoding query at position p needs exactly
            // [p-w+1, p-1] plus itself, so nothing visible is ever dropped.
            Some(window) => {
                let (key_states, value_states) = match &self.kv_cache {
                    None => (key_states, value_states),
                    Some((prev_k, prev_v)) => (
                        candle_core::Tensor::cat(&[prev_k, &key_states], 2)?,
                        candle_core::Tensor::cat(&[prev_v, &value_states], 2)?,
                    ),
                };
                let retained = match sliding_trim_bounds(key_states.dim(2)?, window) {
                    Some((start, length)) => (
                        key_states.narrow(2, start, length)?,
                        value_states.narrow(2, start, length)?,
                    ),
                    None => (key_states.clone(), value_states.clone()),
                };
                self.kv_cache = Some(retained);
                // The cuda matmul wants matrix-packed operands; a cat
                // output already is one (no-op), the first-forward
                // transposed passthrough is not.
                (key_states.contiguous()?, value_states.contiguous()?)
            }
            // Full layers append into the capacity-stepped buffers.
            None => self.full_append(&key_states, &value_states)?,
        };

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        // D5 drives dispatch: Causal prefill blocks go flash, Window
        // bands go windowed flash; decode (q_len 1) stays eager -
        // candle-flash-attn 0.11 has no split-kv decode kernel, so a lone
        // q block underfills the SMs and loses to the eager gemv on long
        // kv (measured: -11% at 2K to -19% at 32K, winning only under
        // ~few-hundred kv).
        let attn_output = if self.use_flash_attn && q_len > 1 {
            let q = query_states.transpose(1, 2)?;
            let k = key_states.transpose(1, 2)?;
            let v = value_states.transpose(1, 2)?;
            let fused = match attention_mask {
                AttnMask::Window(_) => match self.sliding_window {
                    Some(window) => flash_attn_windowed(&q, &k, &v, scale as f32, window)?,
                    None => snafu::whatever!("window descriptor on a full-attention layer"),
                },
                AttnMask::Causal(_) => flash_attn(&q, &k, &v, scale as f32, true)?,
                AttnMask::None => flash_attn(&q, &k, &v, scale as f32, false)?,
            };
            fused.transpose(1, 2)?
        } else {
            let mut attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
            if let Some(mask) = attention_mask.eager_tensor()? {
                attn_weights = attn_weights.broadcast_add(mask)?;
            }
            // HF computes softmax in f32 and casts back; match it for parity.
            let attn_weights = if dtype == candle_core::DType::F32 {
                candle_nn::ops::softmax_last_dim(&attn_weights)?
            } else {
                candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(candle_core::DType::F32)?)?.to_dtype(dtype)?
            };
            attn_weights.matmul(&value_states)?
        };
        Ok(attn_output
            .transpose(1, 2)?
            .reshape((b_size, q_len, self.hidden_size))?
            .apply(&self.o_proj)?)
    }

    fn cached_len(&self) -> usize {
        match self.sliding_window {
            Some(_) => match &self.kv_cache {
                Some((key_cache, _)) => key_cache.dims()[2],
                None => 0,
            },
            None => self.full_cache_len,
        }
    }

    /// Sliding caches drop; full-layer buffers keep their reserved
    /// capacity (only the length resets), so repeated generations do not
    /// re-pay the allocation.
    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        self.full_cache_len = 0;
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
    fn new(cfg: &Olmo3Config, vb: candle_nn::VarBuilder) -> AlmostResult<Self> {
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
    fn forward(&self, xs: &candle_core::Tensor) -> AlmostResult<candle_core::Tensor> {
        let gated = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let up = xs.apply(&self.up_proj)?;
        Ok((gated * up)?.apply(&self.down_proj)?)
    }
}
