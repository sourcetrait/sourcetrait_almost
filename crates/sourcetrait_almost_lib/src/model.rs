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

/// D3 retention bounds: the (start, length) of the tail a sliding cache
/// persists after a prefill reaches total_len positions - the last
/// window-1 entries (a following query needs at most that many past
/// keys).
pub(crate) fn sliding_trim_bounds(total_len: usize, window: usize) -> (usize, usize) {
    let cap = window - 1;
    if total_len > cap {
        (total_len - cap, cap)
    } else {
        (0, total_len)
    }
}

/// `count` logical entries of a sliding ring starting at logical index
/// `logical_start`, as a linear tensor: a plain narrow when the span is
/// contiguous in the buffer, a two-narrow cat when it wraps across the
/// physical end.
fn ring_linear_span(
    ring: &candle_core::Tensor,
    head: usize,
    logical_start: usize,
    count: usize,
) -> AlmostResult<candle_core::Tensor> {
    let window = ring.dims()[2];
    let start = (head + logical_start) % window;
    if start + count <= window {
        Ok(ring.narrow(2, start, count)?)
    } else {
        let first = window - start;
        Ok(candle_core::Tensor::cat(
            &[&ring.narrow(2, start, first)?, &ring.narrow(2, 0, count - first)?],
            2,
        )?)
    }
}

/// The last `need` logical entries of a sliding ring (see
/// ring_linear_span).
fn ring_linear_tail(
    ring: &candle_core::Tensor,
    head: usize,
    len: usize,
    need: usize,
) -> AlmostResult<candle_core::Tensor> {
    ring_linear_span(ring, head, len - need, need)
}

/// Per-layer cache checkpoint for speculative verification (E5a): the
/// scalar lengths plus the oldest `span` logical ring entries - the only
/// content the verification forward's linear ring rewrite can destroy
/// that a rollback might need.
struct AttentionMark {
    full_len: usize,
    saved_key: Option<candle_core::Tensor>,
    saved_value: Option<candle_core::Tensor>,
}

/// Model-level cache checkpoint: taken before a speculative verification
/// forward of `span` tokens at `offset`, rolled back to the accepted
/// prefix afterwards.
pub(crate) struct CacheMark {
    marks: Vec<AttentionMark>,
    offset: usize,
    span: usize,
}

/// The almost Olmo 3 decoder (D1): HF-shaped single-target module, MHA
/// only by policy (the sole checkpoint is MHA; kv-group machinery is
/// deliberately absent). Sliding layers hold a fixed window-sized ring
/// (D3/E3; at most w-1 entries persist between prefill chunks, w after
/// a decode step - the attended set is [p-w+1, p] either way); decode
/// builds no masks on any layer (D4).
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

    /// Pre-reserve full-layer cache capacity for a known upcoming context
    /// length, at the fine grain (FULL_CACHE_RESERVE_GRAIN): kills the
    /// up-to-a-step tail overshoot of append-time growth without
    /// re-introducing per-chunk reallocation. Growth past the reserve
    /// keeps the coarse FULL_CACHE_STEP behavior.
    pub(crate) fn reserve_full_caches(&mut self, total_len: usize) -> AlmostResult<()> {
        let device = self.device.clone();
        for layer in self.layers.iter_mut() {
            layer.self_attn.reserve_full(total_len, self.dtype, &device)?;
        }
        Ok(())
    }

    /// Largest sliding-layer ring occupancy currently held; the T4
    /// invariant pins it at window-1 between prefill chunks and window
    /// after a decode step (the current token's slot included).
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

    /// Checkpoint the caches before a speculative verification forward
    /// of `span` tokens at `offset` (E5a). Cost: span ring slots per
    /// sliding layer.
    pub(crate) fn cache_mark(&self, offset: usize, span: usize) -> AlmostResult<CacheMark> {
        snafu::ensure_whatever!(
            span >= 2,
            "a verification span of {span} would not take the prefill path the rollback undoes"
        );
        let mut marks = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            marks.push(layer.self_attn.cache_mark(offset, span)?);
        }
        Ok(CacheMark { marks, offset, span })
    }

    /// E2: persist the current cache state (the standing prefix) as one
    /// safetensors file. Returns the snapshotted context length.
    pub(crate) fn snapshot_caches(&self, path: &Path) -> AlmostResult<usize> {
        let mut context_len: Option<usize> = None;
        let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let Some((key, value)) = layer.self_attn.export_cache()? else {
                continue;
            };
            if layer.layer_type == LayerType::Full {
                let len = key.dims()[2];
                if let Some(existing) = context_len {
                    snafu::ensure_whatever!(
                        existing == len,
                        "full-layer cache lengths disagree ({existing} vs {len})"
                    );
                } else {
                    context_len = Some(len);
                }
            }
            tensors.insert(format!("layer{layer_idx}.key"), key);
            tensors.insert(format!("layer{layer_idx}.value"), value);
        }
        let Some(context_len) = context_len else {
            snafu::whatever!("nothing to snapshot (empty caches)");
        };
        let cpu = candle_core::Device::Cpu;
        tensors.insert(
            String::from("meta"),
            candle_core::Tensor::new(&[1u32, context_len as u32], &cpu)?,
        );
        candle_core::safetensors::save(&tensors, path)?;
        Ok(context_len)
    }

    /// E2: load a snapshot back into the caches, replacing their state.
    /// Returns the restored context length (the offset to continue at).
    pub(crate) fn restore_caches(&mut self, path: &Path) -> AlmostResult<usize> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        let Some(meta) = tensors.get("meta") else {
            snafu::whatever!("snapshot carries no meta tensor");
        };
        let meta: Vec<u32> = meta.to_device(&candle_core::Device::Cpu)?.to_vec1()?;
        snafu::ensure_whatever!(
            meta.len() == 2 && meta[0] == 1,
            "unsupported snapshot meta {meta:?}"
        );
        let context_len = meta[1] as usize;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let key = tensors.get(&format!("layer{layer_idx}.key"));
            let value = tensors.get(&format!("layer{layer_idx}.value"));
            let (Some(key), Some(value)) = (key, value) else {
                snafu::whatever!("snapshot is missing layer {layer_idx}");
            };
            snafu::ensure_whatever!(
                key.dtype() == self.dtype,
                "snapshot dtype {:?} does not match the model dtype {:?}",
                key.dtype(),
                self.dtype
            );
            layer.self_attn.import_cache(key, value)?;
        }
        Ok(context_len)
    }

    /// Rewind the caches to `accepted` consumed tokens out of the
    /// marked verification span.
    pub(crate) fn cache_rollback(&mut self, mark: &CacheMark, accepted: usize) -> AlmostResult<()> {
        snafu::ensure_whatever!(
            accepted >= 1 && accepted <= mark.span,
            "rollback to {} tokens outside the marked span of {}",
            accepted,
            mark.span
        );
        for (layer, layer_mark) in self.layers.iter_mut().zip(mark.marks.iter()) {
            layer
                .self_attn
                .cache_rollback(layer_mark, mark.offset, accepted, mark.span)?;
        }
        Ok(())
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

/// Fine grain for a known-length reserve (generate's prompt prefill,
/// snapshot restore): tapers the tail overshoot that FULL_CACHE_STEP
/// rounding leaves (up to a full step - ~1 GiB across the 8 full layers
/// at bf16) while append-time growth stays coarse.
const FULL_CACHE_RESERVE_GRAIN: usize = 1024;

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
    /// E3 sliding-layer ring: a fixed (b, h, window, d) buffer per side.
    /// Decode writes the new slot in place and attends the whole ring
    /// (rope carries absolute positions, so attention is order-free);
    /// prefill leaves the ring linear (head 0). ring_head is the oldest
    /// slot once rotation starts; ring_len counts valid slots.
    ring_key: Option<candle_core::Tensor>,
    ring_value: Option<candle_core::Tensor>,
    ring_head: usize,
    ring_len: usize,
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
            ring_key: None,
            ring_value: None,
            ring_head: 0,
            ring_len: 0,
            full_key_buffer: None,
            full_value_buffer: None,
            full_cache_len: 0,
            num_heads,
            head_dim,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Grow the full-layer buffers to `capacity` slots, preserving content;
    /// no-op when they already hold enough. Callers pick the rounding:
    /// append growth is coarse (FULL_CACHE_STEP), known-length reserves are
    /// fine (FULL_CACHE_RESERVE_GRAIN).
    fn full_grow(
        &mut self,
        b_size: usize,
        capacity: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> AlmostResult<()> {
        let current = match &self.full_key_buffer {
            Some(buffer) => buffer.dims()[2],
            None => 0,
        };
        if capacity <= current {
            return Ok(());
        }
        let shape = (b_size, self.num_heads, capacity, self.head_dim);
        let new_key = candle_core::Tensor::zeros(shape, dtype, device)?;
        let new_value = candle_core::Tensor::zeros(shape, dtype, device)?;
        if self.full_cache_len > 0
            && let (Some(old_key), Some(old_value)) = (&self.full_key_buffer, &self.full_value_buffer)
        {
            new_key.slice_set(&old_key.narrow(2, 0, self.full_cache_len)?.contiguous()?, 2, 0)?;
            new_value.slice_set(&old_value.narrow(2, 0, self.full_cache_len)?.contiguous()?, 2, 0)?;
        }
        self.full_key_buffer = Some(new_key);
        self.full_value_buffer = Some(new_value);
        Ok(())
    }

    /// Known-length capacity reserve at the fine grain; no-op on sliding
    /// layers. Batch-1 shape by design (the generate() surface is batch-1).
    fn reserve_full(
        &mut self,
        total_len: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> AlmostResult<()> {
        if self.sliding_window.is_some() {
            return Ok(());
        }
        let capacity = total_len.div_ceil(FULL_CACHE_RESERVE_GRAIN) * FULL_CACHE_RESERVE_GRAIN;
        self.full_grow(1, capacity, dtype, device)
    }

    /// Append rotated k/v into the capacity-stepped full-layer buffers and
    /// return the live (b, h, len, dim) views. Append-only cat semantics;
    /// only the allocation granularity is coarse (FULL_CACHE_STEP), minus
    /// whatever a known-length reserve already provided.
    fn full_append(
        &mut self,
        key_states: &candle_core::Tensor,
        value_states: &candle_core::Tensor,
    ) -> AlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
        // slice_set requires contiguous sources; k is post-rope contiguous
        // already, v arrives as a transposed view.
        let key_states = key_states.contiguous()?;
        let value_states = value_states.contiguous()?;
        let (b_size, _heads, seq_len, _head_dim) = key_states.dims4()?;
        let needed = self.full_cache_len + seq_len;
        let capacity = match &self.full_key_buffer {
            Some(buffer) => buffer.dims()[2],
            None => 0,
        };
        if needed > capacity {
            let new_capacity = needed.div_ceil(FULL_CACHE_STEP) * FULL_CACHE_STEP;
            self.full_grow(b_size, new_capacity, key_states.dtype(), key_states.device())?;
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

        // D3 visibility rides the cache structure: sliding layers hold a
        // fixed ring (decode attends exactly the ring = [p-w+1, p]); full
        // layers append into capacity-stepped buffers. Prefill masks or
        // window kernels handle multi-token visibility.
        let (key_states, value_states) = match self.sliding_window {
            Some(window) => {
                if q_len == 1 {
                    self.ring_decode(&key_states, &value_states, window)?
                } else {
                    self.ring_prefill(&key_states, &value_states, window, seqlen_offset)?
                }
            }
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

    /// Allocate the sliding ring on first use; kept across clears.
    fn ensure_ring(
        &mut self,
        b_size: usize,
        window: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> AlmostResult<()> {
        if self.ring_key.is_none() {
            let shape = (b_size, self.num_heads, window, self.head_dim);
            self.ring_key = Some(candle_core::Tensor::zeros(shape, dtype, device)?);
            self.ring_value = Some(candle_core::Tensor::zeros(shape, dtype, device)?);
        }
        Ok(())
    }

    /// Decode step on a sliding layer: write the new k/v into the next
    /// ring slot in place (overwriting the slot that just left the
    /// window once full), then attend the whole ring - exactly the
    /// visible set [p-w+1, p], rotation-invariant because rope bakes
    /// absolute positions into k. Zero allocation, zero copy beyond the
    /// one slot.
    fn ring_decode(
        &mut self,
        key_states: &candle_core::Tensor,
        value_states: &candle_core::Tensor,
        window: usize,
    ) -> AlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
        let key_states = key_states.contiguous()?;
        let value_states = value_states.contiguous()?;
        let (b_size, _heads, _one, _d) = key_states.dims4()?;
        self.ensure_ring(b_size, window, key_states.dtype(), key_states.device())?;
        let (Some(ring_key), Some(ring_value)) = (&self.ring_key, &self.ring_value) else {
            snafu::whatever!("sliding ring absent after ensure");
        };
        // General ring append (head may be nonzero with len < window
        // after a speculation rollback): write at the logical end,
        // advance head only once full.
        let slot = if self.ring_len < window {
            let slot = (self.ring_head + self.ring_len) % window;
            self.ring_len += 1;
            slot
        } else {
            let slot = self.ring_head;
            self.ring_head = (self.ring_head + 1) % window;
            slot
        };
        ring_key.slice_set(&key_states, 2, slot)?;
        ring_value.slice_set(&value_states, 2, slot)?;
        if self.ring_len == window {
            Ok((ring_key.clone(), ring_value.clone()))
        } else if self.ring_head + self.ring_len <= window {
            Ok((
                ring_key.narrow(2, self.ring_head, self.ring_len)?,
                ring_value.narrow(2, self.ring_head, self.ring_len)?,
            ))
        } else {
            Ok((
                ring_linear_tail(ring_key, self.ring_head, self.ring_len, self.ring_len)?,
                ring_linear_tail(ring_value, self.ring_head, self.ring_len, self.ring_len)?,
            ))
        }
    }

    /// Prefill chunk on a sliding layer: attention runs over the linear
    /// past (the last min(offset, w-1) logical ring entries) plus the
    /// chunk; afterwards the last min(total, w-1) positions are written
    /// back linearly (head 0), so prefill costs one bounded copy per
    /// CHUNK, never per token.
    fn ring_prefill(
        &mut self,
        key_states: &candle_core::Tensor,
        value_states: &candle_core::Tensor,
        window: usize,
        seqlen_offset: usize,
    ) -> AlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
        let (b_size, _heads, q_len, _d) = key_states.dims4()?;
        self.ensure_ring(b_size, window, key_states.dtype(), key_states.device())?;
        let need = seqlen_offset.min(window - 1);
        snafu::ensure_whatever!(
            self.ring_len >= need,
            "sliding ring holds {} entries but the chunk at offset {} needs {}",
            self.ring_len,
            seqlen_offset,
            need
        );
        let (key_states, value_states) = if need == 0 {
            (key_states.contiguous()?, value_states.contiguous()?)
        } else {
            let (Some(ring_key), Some(ring_value)) = (&self.ring_key, &self.ring_value) else {
                snafu::whatever!("sliding ring absent after ensure");
            };
            let past_key = ring_linear_tail(ring_key, self.ring_head, self.ring_len, need)?;
            let past_value = ring_linear_tail(ring_value, self.ring_head, self.ring_len, need)?;
            (
                candle_core::Tensor::cat(&[&past_key, key_states], 2)?,
                candle_core::Tensor::cat(&[&past_value, value_states], 2)?,
            )
        };
        // Persist the tail linearly for the next chunk or decode.
        let total = seqlen_offset + q_len;
        let kv_len = key_states.dims()[2];
        let (_, keep) = sliding_trim_bounds(total, window);
        let tail_key = key_states.narrow(2, kv_len - keep, keep)?.contiguous()?;
        let tail_value = value_states.narrow(2, kv_len - keep, keep)?.contiguous()?;
        let (Some(ring_key), Some(ring_value)) = (&self.ring_key, &self.ring_value) else {
            snafu::whatever!("sliding ring absent after ensure");
        };
        ring_key.slice_set(&tail_key, 2, 0)?;
        ring_value.slice_set(&tail_value, 2, 0)?;
        self.ring_head = 0;
        self.ring_len = keep;
        Ok((key_states, value_states))
    }

    /// Checkpoint before a speculative verification forward: lengths,
    /// plus the oldest rollback-reachable ring entries on sliding
    /// layers (the linear rewrite the verification's prefill path
    /// performs is the only destructive step a rollback must undo).
    /// Saved entries are anchored at position offset - min(offset, w-1),
    /// the same origin the rollback computes, so a saturated post-decode
    /// ring (which holds one entry older than that) skips its oldest
    /// slot.
    fn cache_mark(&self, offset: usize, span: usize) -> AlmostResult<AttentionMark> {
        let Some(window) = self.sliding_window else {
            return Ok(AttentionMark {
                full_len: self.full_cache_len,
                saved_key: None,
                saved_value: None,
            });
        };
        let past_len = offset.min(window - 1);
        snafu::ensure_whatever!(
            self.ring_len >= past_len,
            "sliding ring holds {} entries but offset {} implies {}",
            self.ring_len,
            offset,
            past_len
        );
        let skip = self.ring_len - past_len;
        let save = span.min(past_len);
        let (saved_key, saved_value) = if save == 0 {
            (None, None)
        } else {
            let (Some(ring_key), Some(ring_value)) = (&self.ring_key, &self.ring_value) else {
                snafu::whatever!("sliding ring absent with nonzero occupancy");
            };
            (
                Some(ring_linear_span(ring_key, self.ring_head, skip, save)?.contiguous()?),
                Some(ring_linear_span(ring_value, self.ring_head, skip, save)?.contiguous()?),
            )
        };
        Ok(AttentionMark {
            full_len: 0,
            saved_key,
            saved_value,
        })
    }

    /// Rewind to `accepted` of the `span` verified tokens. Full layers
    /// shrink their length; sliding rings re-point into the linear
    /// rewrite the verification left, restoring displaced oldest
    /// entries from the mark when the window saturated across the span.
    fn cache_rollback(
        &mut self,
        mark: &AttentionMark,
        offset_before: usize,
        accepted: usize,
        span: usize,
    ) -> AlmostResult<()> {
        let Some(window) = self.sliding_window else {
            self.full_cache_len = mark.full_len + accepted;
            return Ok(());
        };
        let total_after = offset_before + span;
        let total_target = offset_before + accepted;
        let keep_after = total_after.min(window - 1);
        let keep_target = total_target.min(window - 1);
        let after_start = total_after - keep_after;
        let target_start = total_target - keep_target;
        if target_start >= after_start {
            // Everything needed survived the rewrite; re-point into it.
            self.ring_head = target_start - after_start;
            self.ring_len = keep_target;
            return Ok(());
        }
        let missing = after_start - target_start;
        let (Some(saved_key), Some(saved_value)) = (&mark.saved_key, &mark.saved_value) else {
            snafu::whatever!("ring rollback needs {missing} displaced entries but none were saved");
        };
        let saved_len = saved_key.dims()[2];
        let old_start = offset_before - offset_before.min(window - 1);
        let need_from = target_start - old_start;
        snafu::ensure_whatever!(
            need_from + missing <= saved_len,
            "ring rollback needs saved entries [{need_from}, {}) but only {saved_len} were saved",
            need_from + missing
        );
        let (Some(ring_key), Some(ring_value)) = (&self.ring_key, &self.ring_value) else {
            snafu::whatever!("sliding ring absent during rollback");
        };
        let restore_key = saved_key.narrow(2, need_from, missing)?.contiguous()?;
        let restore_value = saved_value.narrow(2, need_from, missing)?.contiguous()?;
        ring_key.slice_set(&restore_key, 2, window - missing)?;
        ring_value.slice_set(&restore_value, 2, window - missing)?;
        self.ring_head = window - missing;
        self.ring_len = keep_target;
        Ok(())
    }

    /// The layer's live cache content as linear cpu tensors (k, v) for
    /// an E2 snapshot; None when the cache is empty.
    fn export_cache(&self) -> AlmostResult<Option<(candle_core::Tensor, candle_core::Tensor)>> {
        let cpu = candle_core::Device::Cpu;
        match self.sliding_window {
            None => {
                if self.full_cache_len == 0 {
                    return Ok(None);
                }
                let (Some(key_buffer), Some(value_buffer)) = (&self.full_key_buffer, &self.full_value_buffer)
                else {
                    snafu::whatever!("full-layer buffers absent with nonzero length");
                };
                Ok(Some((
                    key_buffer.narrow(2, 0, self.full_cache_len)?.to_device(&cpu)?,
                    value_buffer.narrow(2, 0, self.full_cache_len)?.to_device(&cpu)?,
                )))
            }
            Some(window) => {
                if self.ring_len == 0 {
                    return Ok(None);
                }
                let (Some(ring_key), Some(ring_value)) = (&self.ring_key, &self.ring_value) else {
                    snafu::whatever!("sliding ring absent with nonzero occupancy");
                };
                // Normalize to the between-forwards form: at most w-1
                // entries (a post-decode ring carries the extra oldest
                // slot that the next forward would drop anyway).
                let need = self.ring_len.min(window - 1);
                Ok(Some((
                    ring_linear_tail(ring_key, self.ring_head, self.ring_len, need)?.to_device(&cpu)?,
                    ring_linear_tail(ring_value, self.ring_head, self.ring_len, need)?.to_device(&cpu)?,
                )))
            }
        }
    }

    /// Load a snapshot's linear (k, v) back into this layer's cache
    /// (buffers re-reserved as needed; rings land linear, head 0).
    fn import_cache(
        &mut self,
        key: &candle_core::Tensor,
        value: &candle_core::Tensor,
    ) -> AlmostResult<()> {
        let (b_size, heads, len, head_dim) = key.dims4()?;
        snafu::ensure_whatever!(
            heads == self.num_heads && head_dim == self.head_dim,
            "snapshot layer shape ({heads}, {head_dim}) does not match the model ({}, {})",
            self.num_heads,
            self.head_dim
        );
        match self.sliding_window {
            None => {
                self.full_cache_len = 0;
                if len > 0 {
                    // Restored length is known: fine-grain reserve, so the
                    // append below does not pay FULL_CACHE_STEP rounding.
                    let capacity =
                        len.div_ceil(FULL_CACHE_RESERVE_GRAIN) * FULL_CACHE_RESERVE_GRAIN;
                    self.full_grow(b_size, capacity, key.dtype(), key.device())?;
                    let (key_device, value_device) = self.full_append(key, value)?;
                    let _ = (key_device, value_device);
                }
                Ok(())
            }
            Some(window) => {
                snafu::ensure_whatever!(
                    len < window,
                    "snapshot ring content {len} exceeds window-1 {}",
                    window - 1
                );
                self.ring_head = 0;
                self.ring_len = 0;
                if len > 0 {
                    self.ensure_ring(b_size, window, key.dtype(), key.device())?;
                    let (Some(ring_key), Some(ring_value)) = (&self.ring_key, &self.ring_value) else {
                        snafu::whatever!("sliding ring absent after ensure");
                    };
                    ring_key.slice_set(&key.contiguous()?, 2, 0)?;
                    ring_value.slice_set(&value.contiguous()?, 2, 0)?;
                    self.ring_len = len;
                }
                Ok(())
            }
        }
    }

    fn cached_len(&self) -> usize {
        match self.sliding_window {
            Some(_) => self.ring_len,
            None => self.full_cache_len,
        }
    }

    /// Lengths reset; ring and full-layer buffers keep their reserved
    /// capacity, so repeated generations do not re-pay the allocation.
    fn clear_kv_cache(&mut self) {
        self.ring_head = 0;
        self.ring_len = 0;
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
