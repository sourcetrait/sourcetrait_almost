//! The hybrid model driver over its 24 GDN and 8 attention layers.
use crate::*;

enum Layer {
    Gdn(GdnLayer),
    Attn(AttnLayer),
}

/// The pre-verification cache mark a speculation round accepts against.
pub(crate) struct SpecMark {
    pub(crate) context_len: usize,
}

/// The public handle mark_context returns.
pub struct ContextMark(SpecMark);

/// The margin a graph arm pre-reserves past the live length.
const RESERVE_DECODE_MARGIN: usize = 256;

/// The era-two hybrid model and its three forward paths.
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
    ) -> LibQuestResult<Self> {
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
        #[cfg(feature = "cuda")]
        if settings.fused_prefill {
            let shared: fused_prefill::SharedPrefillScratch =
                std::rc::Rc::new(std::cell::RefCell::new(None));
            for layer in &mut layers {
                if let Layer::Gdn(gdn) = layer {
                    gdn.install_prefill_scratch(shared.clone());
                }
            }
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

    /// All-position logits for one sequence; STATELESS by construction.
    pub fn forward_all(
        &self,
        input_ids: &candle_core::Tensor,
    ) -> LibQuestResult<candle_core::Tensor> {
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

    /// The carried advance every chunk form shares, before the norm.
    fn advance_chunk(
        &mut self,
        input_ids: &candle_core::Tensor,
    ) -> LibQuestResult<candle_core::Tensor> {
        let seq_len = input_ids.dim(0)?;
        let mut hidden = self.embed_tokens.index_select(input_ids, 0)?;
        let mask = if seq_len > 1 && self.needs_prefill_mask() {
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

    /// Whether a multi-token carried chunk needs the mask tensor built.
    fn needs_prefill_mask(&self) -> bool {
        #[cfg(feature = "attn-profile")]
        {
            let profiled = self.layers.iter().any(|layer| {
                matches!(layer, Layer::Attn(attn) if attn.profile.borrow().is_some())
            });
            if profiled {
                return true;
            }
        }
        let flash_covers = cfg!(feature = "flash-attn")
            && self.settings.use_flash_attn
            && self.device().is_cuda()
            && matches!(
                self.embed_tokens.dtype(),
                candle_core::DType::BF16 | candle_core::DType::F16
            );
        !flash_covers
    }

    /// One CARRIED forward over the next chunk, advancing every cache.
    pub fn forward_chunk(
        &mut self,
        input_ids: &candle_core::Tensor,
    ) -> LibQuestResult<candle_core::Tensor> {
        let hidden = self.advance_chunk(input_ids)?;
        let hidden =
            norms::rms_norm_auto(&hidden, &self.norm, self.rms_eps, self.settings.fused_gdn)?;
        Ok(self.lm_head.forward(&hidden)?)
    }

    /// The final-chunk form: advance, then norm and head on one row.
    pub fn forward_chunk_last(
        &mut self,
        input_ids: &candle_core::Tensor,
    ) -> LibQuestResult<candle_core::Tensor> {
        let hidden = self.advance_chunk(input_ids)?;
        let rows = hidden.dim(0)?;
        let last = hidden.narrow(0, rows - 1, 1)?;
        let last =
            norms::rms_norm_auto(&last, &self.norm, self.rms_eps, self.settings.fused_gdn)?;
        Ok(self.lm_head.forward(&last)?)
    }

    /// The intermediate-chunk form: caches advance, no logits at all.
    pub fn forward_chunk_carry(
        &mut self,
        input_ids: &candle_core::Tensor,
    ) -> LibQuestResult<()> {
        self.advance_chunk(input_ids)?;
        Ok(())
    }

    /// Collect the layers' park latches, retiring capture if any fired.
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

    /// Reset every carried cache and the context length.
    pub fn clear_cache(&mut self) -> LibQuestResult<()> {
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

    /// The known-length KV pre-reserve: one allocation for the run.
    pub fn reserve_for_generation(&mut self, prompt_tokens: usize) -> LibQuestResult<()> {
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

    /// Persist every carried cache plus the id trail as one file.
    pub fn snapshot_caches(
        &self,
        snapshots_dir: &Path,
        token: &str,
        model_id: &str,
        context_ids: &[u32],
    ) -> LibQuestResult<PathBuf> {
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

    /// Load a saved context into the carried caches, validating geometry.
    pub fn restore_caches(
        &mut self,
        snapshots_dir: &Path,
        token: &str,
        expected_model_id: &str,
    ) -> LibQuestResult<RestoredContext> {
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

    /// A carried-context mark for shared-prefix scoring loops.
    pub fn mark_context(&mut self) -> LibQuestResult<ContextMark> {
        Ok(ContextMark(self.spec_mark()?))
    }

    /// Roll the carried caches back to a mark_context mark.
    pub fn rollback_context(&mut self, mark: &ContextMark) -> LibQuestResult<()> {
        self.spec_rollback(&mark.0)
    }

    /// Shadow every GDN cache, arm its capture, record the length.
    pub(crate) fn spec_mark(&mut self) -> LibQuestResult<SpecMark> {
        for layer in &mut self.layers {
            if let Layer::Gdn(gdn) = layer {
                gdn.shadow_save()?;
            }
        }
        Ok(SpecMark {
            context_len: self.context_len,
        })
    }

    /// The FULL rewind to a mark; partial accepts ride spec_accept.
    pub(crate) fn spec_rollback(&mut self, mark: &SpecMark) -> LibQuestResult<()> {
        for layer in &mut self.layers {
            match layer {
                Layer::Gdn(gdn) => {
                    gdn.shadow_restore()?;
                    gdn.spec_release();
                }
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

    /// The one-pass partial accept: keep the verify chunk's own work.
    pub(crate) fn spec_accept(
        &mut self,
        mark: &SpecMark,
        consumed: usize,
    ) -> LibQuestResult<()> {
        for layer in &mut self.layers {
            match layer {
                Layer::Gdn(gdn) => {
                    gdn.shadow_restore()?;
                    gdn.spec_readvance(consumed)?;
                }
                Layer::Attn(attn) => {
                    if let Some(kv) = &mut attn.kv {
                        kv.len = mark.context_len + consumed;
                    }
                }
            }
        }
        self.context_len = mark.context_len + consumed;
        Ok(())
    }

    /// A full accept keeps every cache row; the captures release.
    pub(crate) fn spec_release(&mut self) {
        for layer in &mut self.layers {
            if let Layer::Gdn(gdn) = layer {
                gdn.spec_release();
            }
        }
    }

    /// Keep-sets from the live last-pass scores, one per layer.
    fn evict_keep_sets(
        &self,
        evict: &EvictionSettings,
    ) -> LibQuestResult<Vec<Vec<Vec<u32>>>> {
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

    /// The one-shot post-prefill compaction, shrinking the buffers.
    pub(crate) fn evict_post_prefill(
        &mut self,
        evict: &EvictionSettings,
    ) -> LibQuestResult<()> {
        if self.context_len <= evict.decode_cap {
            return Ok(());
        }
        let keep_sets = self.evict_keep_sets(evict)?;
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

    /// A decode-overflow re-compaction epoch, packing in place.
    pub(crate) fn evict_overflow(&mut self, evict: &EvictionSettings) -> LibQuestResult<()> {
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

    /// Arm the density observation pass on every attention layer.
    #[cfg(feature = "attn-profile")]
    pub fn arm_attn_profile(&mut self, window: usize) {
        for layer in &mut self.layers {
            if let Layer::Attn(attn) = layer {
                attn.profile
                    .replace(Some(profile::ProfileAccum::new(attn.num_heads, window)));
            }
        }
    }

    /// Collect and disarm; reports carry absolute layer indices.
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

    /// The uniform attention KV length graph staging derives from.
    #[cfg(feature = "cuda")]
    fn graph_kv_state(&self) -> LibQuestResult<(usize, usize)> {
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

    /// Arm the staged graph decode path, pre-growing KV to its ceiling.
    #[cfg(feature = "cuda")]
    pub(crate) fn arm_graph_decode(&mut self, expected_total: usize) -> LibQuestResult<()> {
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

    /// Non-cuda builds carry no graph path; arming is a hard error.
    #[cfg(not(feature = "cuda"))]
    pub(crate) fn arm_graph_decode(&mut self, _expected_total: usize) -> LibQuestResult<()> {
        snafu::whatever!("this build carries no cuda support (decode graphs need --features cuda)")
    }

    /// Whether the staged decode path is armed for this cache state.
    #[cfg(feature = "cuda")]
    pub(crate) fn graph_armed(&self) -> bool {
        self.graph_stage.as_ref().is_some_and(|stage| stage.armed)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn graph_armed(&self) -> bool {
        false
    }

    /// The gates' switch between capture-replay and uncaptured stepping.
    #[cfg(feature = "cuda")]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_graph_capture(&mut self, enabled: bool) -> LibQuestResult<()> {
        let Some(stage) = &mut self.graph_stage else {
            snafu::whatever!("no graph stage to configure (arm first)");
        };
        stage.capture_enabled = enabled;
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    pub(crate) fn set_graph_capture(&mut self, _enabled: bool) -> LibQuestResult<()> {
        snafu::whatever!("this build carries no cuda support (decode graphs need --features cuda)")
    }

    /// The capacity epoch: disarm when the next append would overrun.
    #[cfg(feature = "cuda")]
    pub(crate) fn graph_disarm_when_full(&mut self) -> LibQuestResult<()> {
        if !self.graph_armed() {
            return Ok(());
        }
        let (kv_len, capacity) = self.graph_kv_state()?;
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
    pub(crate) fn graph_disarm_when_full(&mut self) -> LibQuestResult<()> {
        Ok(())
    }

    /// Drop every captured graph and trim the driver's cached pools.
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

    /// One staged decode step, returning the persistent logits buffer.
    #[cfg(feature = "cuda")]
    pub(crate) fn graph_decode_step(
        &mut self,
        token: u32,
    ) -> LibQuestResult<candle_core::Tensor> {
        let kv_len = self.context_len;
        let mut stage = match self.graph_stage.take() {
            Some(stage) => stage,
            None => snafu::whatever!("graph decode step without an armed stage"),
        };
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
    ) -> LibQuestResult<candle_core::Tensor> {
        snafu::whatever!("this build carries no cuda support (decode graphs need --features cuda)")
    }

    /// One staged step against a held-out stage: replay, or capture.
    #[cfg(feature = "cuda")]
    fn graph_step_with(
        &mut self,
        stage: &mut graph::DecodeStage,
        token: u32,
        kv_len: usize,
    ) -> LibQuestResult<()> {
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
                let captured = self.capture_decode_graph(stage)?;
                stage.graphs.insert(key, captured);
                for layer in self.layers.iter_mut() {
                    if let Layer::Attn(attn) = layer {
                        attn.buffers_captured = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// Capture this bucket's decode sequence into a CUDA graph.
    #[cfg(feature = "cuda")]
    fn capture_decode_graph(
        &mut self,
        stage: &graph::DecodeStage,
    ) -> LibQuestResult<cudarc::driver::CudaGraph> {
        let cuda_device = match self.device() {
            candle_core::Device::Cuda(cuda_device) => cuda_device.clone(),
            _ => snafu::whatever!("graph capture requires a cuda device"),
        };
        let stream = cuda_device.cuda_stream();
        let _htod_cache = cuda_device.enable_cuda_graph_htod_cache();
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

    /// The captured region, from embedding to the logits write.
    #[cfg(feature = "cuda")]
    fn graph_forward_sequence(
        &mut self,
        stage: &graph::DecodeStage,
    ) -> LibQuestResult<()> {
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
