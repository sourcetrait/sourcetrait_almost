//! The era-two hybrid model driver over the 32 layers (24 pre-norm
//! GDN + 8 post-norm NoPE attention): forward_all (stateless
//! whole-sequence, the parity instrument), the carried forward_chunk
//! path (chunked prefill + decode over the layers' internal caches),
//! the KvEviction compaction epochs, and the GraphDecode staged path
//! that CUDA graphs capture and replay.
//!
//! The recurrence and its gates ride f32 regardless of model dtype
//! (the pinned upstream discipline); attention softmax computes in
//! f32 and casts back.
use crate::*;

enum Layer {
    Gdn(GdnLayer),
    Attn(AttnLayer),
}

/// SpeculationPort: the pre-verification cache mark. Attention rows
/// are prefix-correct, so their rewind is a length reset recorded
/// here; the GDN caches are cumulative, so the mark shadows them per
/// layer and a partial accept restores the shadows while the caller
/// re-advances the accepted rows (forward_chunk_carry).
pub(crate) struct SpecMark {
    pub(crate) context_len: usize,
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
                LayerKind::LinearAttention => Layer::Gdn(GdnLayer::new(
                    config,
                    settings.fused_gdn,
                    settings.fused_prefill,
                    vb_layer,
                )?),
                LayerKind::FullAttention => Layer::Attn(AttnLayer::new(
                    config,
                    settings.use_flash_attn,
                    settings.eviction.as_ref(),
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

    /// The carried advance every chunk form shares: embed, the layer
    /// loop (offset causal mask for multi-token chunks; decode t == 1
    /// is mask-free), context-length advance, park collection. Returns
    /// the post-layer hidden [t, hidden] BEFORE the final norm.
    fn advance_chunk(
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
        Ok(hidden)
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
        let hidden = self.advance_chunk(input_ids)?;
        let hidden =
            norms::rms_norm_auto(&hidden, &self.norm, self.rms_eps, self.settings.fused_gdn)?;
        Ok(self.lm_head.forward(&hidden)?)
    }

    /// PrefillLogitsSkip: the final-prefill-chunk form - advance, then
    /// norm + head over the LAST row alone -> [1, vocab]. Per-row math
    /// (RMSNorm and the head are row-independent), so the row is
    /// value-identical to forward_chunk's last row at f32; bf16 kernel
    /// shapes differ (1-row vs t-row gemm) - kin-class deltas ride the
    /// gates.
    pub(crate) fn forward_chunk_last(
        &mut self,
        input_ids: &candle_core::Tensor,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let hidden = self.advance_chunk(input_ids)?;
        let rows = hidden.dim(0)?;
        let last = hidden.narrow(0, rows - 1, 1)?;
        let last =
            norms::rms_norm_auto(&last, &self.norm, self.rms_eps, self.settings.fused_gdn)?;
        Ok(self.lm_head.forward(&last)?)
    }

    /// PrefillLogitsSkip: the intermediate-prefill-chunk form - the
    /// caches advance, no logits are computed (no norm, no head) -
    /// where the skipped head traffic lives.
    pub(crate) fn forward_chunk_carry(
        &mut self,
        input_ids: &candle_core::Tensor,
    ) -> LibAlmostResult<()> {
        self.advance_chunk(input_ids)?;
        Ok(())
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
    /// prefill loop): one allocation at EXACTLY prompt + decode
    /// margin, instead of ~one regrow per reserve step of prefill -
    /// each regrow holds old+new buffers during its copy and frees an
    /// odd-sized block into the raw cudaMalloc heap (the
    /// fragmentation that inflates the 32K peak). ReserveGrainTrim:
    /// with no regrows to amortize, the KV_RESERVE_STEP rounding was
    /// pure padding (~170-235 MiB at 32K). A decode outrunning the
    /// margin regrows coarsely as before.
    pub fn reserve_for_generation(&mut self, prompt_tokens: usize) -> LibAlmostResult<()> {
        let device = self.device().clone();
        let dtype = self.embed_tokens.dtype();
        let capacity = prompt_tokens + RESERVE_DECODE_MARGIN;
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

    /// PrefixSnapshots: persist every carried cache (GDN state + conv
    /// tails, attention KV narrowed to the live length) plus the
    /// consumed-id trail as ONE safetensors file at the resolved
    /// token; returns the written path. Settings-agnostic - state is
    /// state (an eviction-compacted store saves as-is; the last-pass
    /// scores are never persisted). The trail must cover context_len
    /// (equal in the exact configuration, longer after compaction).
    pub fn snapshot_caches(
        &self,
        snapshots_dir: &Path,
        token: &str,
        model_id: &str,
        context_ids: &[u32],
    ) -> LibAlmostResult<PathBuf> {
        snafu::ensure_whatever!(
            self.context_len > 0,
            "nothing to snapshot: the carried context is empty"
        );
        let path = snapshot_path(snapshots_dir, token)?;
        let mut tensors = Vec::with_capacity(self.layers.len() * 2 + 1);
        for (index, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Gdn(gdn) => {
                    let (state, conv_tail) = gdn.cache_snapshot();
                    tensors.push((format!("gdn_state_{index}"), state));
                    tensors.push((format!("conv_tail_{index}"), conv_tail));
                }
                Layer::Attn(attn) => {
                    let Some(kv) = &attn.kv else {
                        snafu::whatever!(
                            "snapshot reached unallocated KV buffers (layer {index})"
                        );
                    };
                    snafu::ensure_whatever!(
                        kv.len == self.context_len,
                        "attention KV length {} does not match the context length {} (layer {index})",
                        kv.len,
                        self.context_len
                    );
                    tensors.push((
                        format!("attn_k_{index}"),
                        kv.k.narrow(1, 0, kv.len)?.contiguous()?,
                    ));
                    tensors.push((
                        format!("attn_v_{index}"),
                        kv.v.narrow(1, 0, kv.len)?.contiguous()?,
                    ));
                }
            }
        }
        snapshot::write_snapshot(&path, tensors, model_id, self.context_len, context_ids)?;
        Ok(path)
    }

    /// PrefixSnapshots: load a saved context into the carried caches
    /// at the resolved token. Format validation (version, model id,
    /// trail) rides the read; layer geometry and the model dtype
    /// validate here; writes land in place (GDN) or through the
    /// park-aware reserve (attention KV). An armed graph stage
    /// disarms (a restore is a clear-class epoch; the next arm
    /// revalidates).
    pub fn restore_caches(
        &mut self,
        snapshots_dir: &Path,
        token: &str,
        expected_model_id: &str,
    ) -> LibAlmostResult<RestoredContext> {
        let path = snapshot_path(snapshots_dir, token)?;
        let device = self.device().clone();
        let dtype = self.embed_tokens.dtype();
        let file = snapshot::read_snapshot(&path, expected_model_id, &device)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            match layer {
                Layer::Gdn(gdn) => {
                    let state = file.tensor(&format!("gdn_state_{index}"))?;
                    let conv_tail = file.tensor(&format!("conv_tail_{index}"))?;
                    gdn.restore_cache(state, conv_tail)?;
                }
                Layer::Attn(attn) => {
                    let k = file.tensor(&format!("attn_k_{index}"))?;
                    let v = file.tensor(&format!("attn_v_{index}"))?;
                    snafu::ensure_whatever!(
                        k.dtype() == dtype,
                        "snapshot KV dtype {:?} does not match the model dtype {dtype:?} (layer {index})",
                        k.dtype()
                    );
                    let len = attn.restore_kv(k, v)?;
                    snafu::ensure_whatever!(
                        len == file.context_len,
                        "snapshot KV length {len} does not match its context length {} (layer {index})",
                        file.context_len
                    );
                }
            }
        }
        self.context_len = file.context_len;
        #[cfg(feature = "cuda")]
        if let Some(stage) = &mut self.graph_stage {
            stage.armed = false;
        }
        self.contain_parked_grows();
        Ok(RestoredContext {
            context_len: file.context_len,
            context_ids: file.context_ids,
        })
    }

    /// SpeculationPort: shadow every GDN layer's carried caches and
    /// record the live lengths - the mark a verification forward can
    /// roll back to.
    pub(crate) fn spec_mark(&mut self) -> LibAlmostResult<SpecMark> {
        for layer in &mut self.layers {
            if let Layer::Gdn(gdn) = layer {
                gdn.shadow_save()?;
            }
        }
        Ok(SpecMark {
            context_len: self.context_len,
        })
    }

    /// SpeculationPort: roll every carried cache back to the mark -
    /// GDN shadows restore in place, attention KV lengths and the
    /// context length reset (rows past the mark become invisible; the
    /// caller re-advances the accepted rows through
    /// forward_chunk_carry). Classic-path only by construction
    /// (speculation never arms graphs and refuses eviction).
    pub(crate) fn spec_rollback(&mut self, mark: &SpecMark) -> LibAlmostResult<()> {
        for layer in &mut self.layers {
            match layer {
                Layer::Gdn(gdn) => gdn.shadow_restore()?,
                Layer::Attn(attn) => {
                    if let Some(kv) = &mut attn.kv {
                        kv.len = mark.context_len;
                    }
                }
            }
        }
        self.context_len = mark.context_len;
        Ok(())
    }

    /// KvEviction keep-sets from the live last-pass scores, one per
    /// attention layer (host-side ranking; per-head sets, all
    /// decode_cap long).
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

    /// KvEviction: the one-shot post-prefill compaction (generate
    /// calls this between prefill and graph arming): rank by the
    /// FINAL scoring pass (the question window), compact every
    /// attention layer to the cap, and - buffers never yet captured -
    /// SHRINK them to the decode-bounded capacity, cashing the
    /// KV-residency win. GDN layers are untouched (constant state).
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
        // ReserveGrainTrim: exact - the grain rounding was pure
        // padding here too.
        let capacity = evict.decode_cap + evict::OVERFLOW_SLACK + RESERVE_DECODE_MARGIN;
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

    /// Arm the DensityProfile observation pass on every attention
    /// layer; `window` is the recency width the middle-mass metric
    /// excludes. Armed attention runs the sliced eager path (flash
    /// never dispatches).
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

    /// GraphDecode: arm the staged graph-mode decode path (generate()
    /// calls this after prefill). Pre-grows every attention layer's KV
    /// once to the run's bucket ceiling - so later bucket crossings
    /// never reallocate under a captured graph - then builds or
    /// re-arms the staged buffers from the live cache state.
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
    #[allow(dead_code)]
    pub(crate) fn set_graph_capture(&mut self, _enabled: bool) -> LibAlmostResult<()> {
        snafu::whatever!("this build carries no cuda support (decode graphs need --features cuda)")
    }

    /// GraphDecode capacity epoch: when the next staged append would
    /// exceed the armed KV capacity, disarm and retire capture for
    /// this Model - the remaining steps take the classic path, whose
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
    #[allow(dead_code)]
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
