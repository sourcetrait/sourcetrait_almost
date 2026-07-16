//! E4 decode-graph building blocks: the staged device buffers every
//! per-step dynamic rides (token id, the shared attention append
//! slot, the width-pooled pad mask, the logits output), the raw
//! slot-write scatter the KV append uses inside a capture, and the
//! bucket-keyed cache of instantiated graphs.
//!
//! The GDN layers need none of this: their carried state is
//! fixed-shape and updates in place on every path (model.rs
//! mixer_carried), so the classic t=1 step IS the captured form.
//! Only the 8 attention layers stage - one slot and one mask serve
//! all of them (every layer sees every token, so the lengths are
//! uniform by construction).
use crate::*;

/// The KV append inside a captured graph: a scatter of one
/// (heads, 1, dim) row into a (heads, capacity, dim) buffer at a
/// device-resident slot index - raw pointer args on the model's own
/// stream (safe-slice args wait per-slice events, which is
/// CAPTURE_ISOLATION inside an active capture when the events
/// predate it). Kernels are element-size generic, so bf16/f16 and
/// f32/u32 share two entry points.
const KERNEL_SRC: &str = r#"
extern "C" __global__ void slot_write_16(
    unsigned short* dst,
    const unsigned short* src,
    const unsigned int* slot,
    unsigned int heads,
    unsigned int capacity,
    unsigned int dim
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * dim) { return; }
    unsigned int h = i / dim;
    unsigned int e = i - h * dim;
    dst[((unsigned long long)h * capacity + slot[0]) * dim + e] = src[h * dim + e];
}

extern "C" __global__ void slot_write_32(
    unsigned int* dst,
    const unsigned int* src,
    const unsigned int* slot,
    unsigned int heads,
    unsigned int capacity,
    unsigned int dim
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * dim) { return; }
    unsigned int h = i / dim;
    unsigned int e = i - h * dim;
    dst[((unsigned long long)h * capacity + slot[0]) * dim + e] = src[h * dim + e];
}
"#;

/// Lazily nvrtc-compiled module, one per process (single-device box).
static MODULE: std::sync::OnceLock<std::sync::Arc<cudarc::driver::CudaModule>> =
    std::sync::OnceLock::new();

fn kernel(
    device: &candle_core::CudaDevice,
    name: &str,
) -> LibAlmostResult<cudarc::driver::CudaFunction> {
    let module = match MODULE.get() {
        Some(module) => module.clone(),
        None => {
            let ptx = match cudarc::nvrtc::compile_ptx(KERNEL_SRC) {
                Ok(ptx) => ptx,
                Err(error) => snafu::whatever!("nvrtc compile of slot_write failed: {error}"),
            };
            let module = match device.cuda_stream().context().load_module(ptx) {
                Ok(module) => module,
                Err(error) => snafu::whatever!("loading the slot_write module failed: {error}"),
            };
            MODULE.get_or_init(|| module).clone()
        }
    };
    match module.load_function(name) {
        Ok(function) => Ok(function),
        Err(error) => snafu::whatever!("loading kernel {name} failed: {error}"),
    }
}

fn launch_slot_write<T: candle_core::cuda::CudaDType>(
    function: &cudarc::driver::CudaFunction,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    (dst, dst_offset): (&candle_core::CudaStorage, usize),
    (src, src_offset): (&candle_core::CudaStorage, usize),
    (slot, slot_offset): (&candle_core::CudaStorage, usize),
    (heads, capacity, dim): (u32, u32, u32),
) -> candle_core::Result<()> {
    use cudarc::driver::{DevicePtr, PushKernelArg};

    // A candle view's start_offset is part of its address: a
    // "contiguous" tensor derived from a narrow (size-1 dims skip the
    // stride check) still points mid-storage, so every operand slices
    // at its layout offset - reading from 0 wrote the WRONG SPAN
    // (q-projection bytes as v) in the un-offset form.
    let dst_slice = T::as_cuda_slice(dst)?.slice(dst_offset..);
    let (dst_ptr, _dst_guard) = dst_slice.device_ptr(stream);
    let src_slice = T::as_cuda_slice(src)?.slice(src_offset..);
    let (src_ptr, _src_guard) = src_slice.device_ptr(stream);
    let slot_slice = slot.as_cuda_slice::<u32>()?.slice(slot_offset..);
    let (slot_ptr, _slot_guard) = slot_slice.device_ptr(stream);

    let total = heads * dim;
    let block = 256u32;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (total.div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder
        .arg(&dst_ptr)
        .arg(&src_ptr)
        .arg(&slot_ptr)
        .arg(&heads)
        .arg(&capacity)
        .arg(&dim);
    if let Err(error) = unsafe { builder.launch(config) } {
        candle_core::bail!("slot_write launch failed: {error}");
    }
    Ok(())
}

/// `buffer.inplace_op3(&row, &slot, &SlotWrite { dtype })`: dst is
/// (heads, capacity, dim), src (heads, 1, dim), slot a (1,) u32
/// device buffer; all contiguous (dst at offset 0).
pub(crate) struct SlotWrite {
    pub(crate) dtype: candle_core::DType,
}

impl candle_core::InplaceOp3 for SlotWrite {
    fn name(&self) -> &'static str {
        "almost_slot_write"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
        _s3: &candle_core::CpuStorage,
        _l3: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("slot_write is a cuda decode-graph building block")
    }

    fn cuda_fwd(
        &self,
        dst: &mut candle_core::CudaStorage,
        dst_layout: &candle_core::Layout,
        src: &candle_core::CudaStorage,
        src_layout: &candle_core::Layout,
        slot: &candle_core::CudaStorage,
        slot_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        let dst_dims = dst_layout.shape().dims3()?;
        let src_dims = src_layout.shape().dims3()?;
        let (heads, capacity, dim) = dst_dims;
        if src_dims != (heads, 1, dim) {
            candle_core::bail!(
                "slot_write wants dst (h, cap, d) and src (h, 1, d); got {dst_dims:?} and {src_dims:?}"
            );
        }
        if !dst_layout.is_contiguous()
            || !src_layout.is_contiguous()
            || !slot_layout.is_contiguous()
        {
            candle_core::bail!("slot_write wants contiguous operands");
        }

        let kernel_name = match self.dtype {
            candle_core::DType::BF16 | candle_core::DType::F16 => "slot_write_16",
            candle_core::DType::F32 | candle_core::DType::U32 => "slot_write_32",
            other => candle_core::bail!("slot_write has no {other:?} kernel"),
        };
        let device = dst.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, kernel_name) {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };

        let dims = (heads as u32, capacity as u32, dim as u32);
        let dst_at = (&*dst, dst_layout.start_offset());
        let src_at = (src, src_layout.start_offset());
        let slot_at = (slot, slot_layout.start_offset());
        match self.dtype {
            candle_core::DType::BF16 => {
                launch_slot_write::<half::bf16>(&function, &stream, dst_at, src_at, slot_at, dims)
            }
            candle_core::DType::F16 => {
                launch_slot_write::<half::f16>(&function, &stream, dst_at, src_at, slot_at, dims)
            }
            candle_core::DType::F32 => {
                launch_slot_write::<f32>(&function, &stream, dst_at, src_at, slot_at, dims)
            }
            candle_core::DType::U32 => {
                launch_slot_write::<u32>(&function, &stream, dst_at, src_at, slot_at, dims)
            }
            _ => unreachable!(),
        }
    }
}

/// Release the device's cached CUDA-graph memory pools (device 0 -
/// the single-device box). Destroyed graphs leave their in-graph
/// allocation pools driver-cached; the trim returns them. Safe
/// beside live graphs; failures are ignored (best-effort).
pub(crate) fn trim_graph_memory() {
    let _ = unsafe { cudarc::driver::sys::cuDeviceGraphMemTrim(0) };
}

/// Smallest grain multiple covering `len` valid entries (at least
/// one grain). The grain is a settings field
/// (LibSettings.graph_bucket_grain).
pub(crate) fn bucket_for(len: usize, grain: usize) -> usize {
    len.max(1).div_ceil(grain) * grain
}

/// Additive pad-mask values: 0.0 below `valid`, -inf across the
/// bucket's pad tail (post-softmax pad mass is exactly zero).
pub(crate) fn pad_mask_values(bucket: usize, valid: usize) -> Vec<f32> {
    (0..bucket)
        .map(|column| if column < valid { 0.0 } else { f32::NEG_INFINITY })
        .collect()
}

/// E4 staged decode state: the persistent device buffers every
/// per-step dynamic reads. The host stages a few bytes before each
/// step; the captured ops read values, never baked host constants.
/// Created once per Model at first arming and kept - masks are
/// width-pooled and value-reset IN PLACE, so every address a
/// captured graph bakes stays stable; clear epochs only drop the
/// armed flag.
#[derive(Debug, Clone)]
pub(crate) struct DecodeStage {
    /// (1,) u32: the token this step feeds (embedding index_select).
    pub(crate) ids: candle_core::Tensor,
    /// (1,) u32: the shared attention append slot (= context_len).
    pub(crate) kv_slot: candle_core::Tensor,
    /// (1, bucket) additive 0/-inf, model dtype (a handle into the
    /// width pool).
    pub(crate) kv_mask: candle_core::Tensor,
    /// (1, vocab) f32: the step's logits, written in-graph.
    pub(crate) logits_out: candle_core::Tensor,
    /// (1, 1) model-dtype 0.0: the mask-enable write source.
    mask_zero: candle_core::Tensor,
    /// Width-keyed mask pool: one tensor per bucket width, created
    /// once and value-reset in place - captured graphs bake these
    /// addresses, so a width's mask must never reallocate.
    masks: HashMap<usize, candle_core::Tensor>,
    /// Captured decode graphs keyed by bucket width.
    pub(crate) graphs: GraphCache,
    /// Replay switch (gates' uncaptured reference legs turn it off).
    pub(crate) capture_enabled: bool,
    /// Lifetime switch: once a regrow parks captured buffers,
    /// capture stays off for this Model (rearm restores
    /// capture_enabled from THIS, never from true).
    pub(crate) capture_permitted: bool,
    /// Whether stepping is armed for the current cache state (clear
    /// epochs unset it; arm_graph_decode re-arms).
    pub(crate) armed: bool,
    /// The KV capacity the cached graphs were captured against; a
    /// change means the buffers reallocated and every cached graph
    /// is stale.
    pub(crate) kv_capacity: usize,
    pub(crate) bucket: usize,
    grain: usize,
    dtype: candle_core::DType,
    device: candle_core::Device,
}

impl DecodeStage {
    /// Build from the live post-prefill state (`kv_len` valid rows in
    /// every attention layer).
    pub(crate) fn new(
        kv_len: usize,
        grain: usize,
        vocab: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> LibAlmostResult<Self> {
        let bucket = bucket_for(kv_len + 1, grain);
        let mut masks: HashMap<usize, candle_core::Tensor> = HashMap::new();
        let kv_mask = Self::pooled_mask(&mut masks, bucket, kv_len, dtype, device)?;
        Ok(Self {
            ids: candle_core::Tensor::zeros((1,), candle_core::DType::U32, device)?,
            kv_slot: candle_core::Tensor::zeros((1,), candle_core::DType::U32, device)?,
            kv_mask,
            logits_out: candle_core::Tensor::zeros(
                (1, vocab),
                candle_core::DType::F32,
                device,
            )?,
            mask_zero: candle_core::Tensor::zeros((1, 1), dtype, device)?,
            masks,
            graphs: GraphCache::default(),
            capture_enabled: true,
            capture_permitted: true,
            armed: true,
            // The creator records the true capacity right after
            // construction.
            kv_capacity: 0,
            bucket,
            grain,
            dtype,
            device: device.clone(),
        })
    }

    /// Get-or-create the width's pooled mask and reset its values to
    /// `valid` IN PLACE (stable address across the pool's lifetime).
    fn pooled_mask(
        pool: &mut HashMap<usize, candle_core::Tensor>,
        bucket: usize,
        valid: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> LibAlmostResult<candle_core::Tensor> {
        let mask = match pool.get(&bucket) {
            Some(mask) => mask.clone(),
            None => {
                let fresh = candle_core::Tensor::zeros((1, bucket), dtype, device)?;
                pool.insert(bucket, fresh.clone());
                fresh
            }
        };
        Self::write_mask_values(&mask, bucket, valid, dtype, device)?;
        Ok(mask)
    }

    /// Overwrite a mask's whole row in place from pad_mask_values.
    fn write_mask_values(
        mask: &candle_core::Tensor,
        bucket: usize,
        valid: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> LibAlmostResult<()> {
        let values = pad_mask_values(bucket, valid);
        let row =
            candle_core::Tensor::from_vec(values, (1, bucket), device)?.to_dtype(dtype)?;
        mask.slice_set(&row, 1, 0)?;
        Ok(())
    }

    /// Re-point the stage at a fresh post-prefill length (a new
    /// generation over the same buffers): bucket re-derived, pooled
    /// mask value-reset in place, stepping re-armed. Captured graphs
    /// survive - the model flushes them separately when the KV
    /// buffers reallocate.
    pub(crate) fn rearm(&mut self, kv_len: usize) -> LibAlmostResult<()> {
        self.bucket = bucket_for(kv_len + 1, self.grain);
        self.kv_mask = Self::pooled_mask(
            &mut self.masks,
            self.bucket,
            kv_len,
            self.dtype,
            &self.device,
        )?;
        self.capture_enabled = self.capture_permitted;
        self.armed = true;
        Ok(())
    }

    /// Re-derive the bucket for the width the NEXT append reaches; a
    /// crossing re-points to (or creates) that width's pooled mask
    /// with values reset from the live length - the graph switch
    /// happens on the same boundary via the cache key.
    pub(crate) fn ensure_bucket(&mut self, kv_len: usize) -> LibAlmostResult<()> {
        let bucket = bucket_for(kv_len + 1, self.grain);
        if bucket != self.bucket {
            self.bucket = bucket;
            self.kv_mask = Self::pooled_mask(
                &mut self.masks,
                bucket,
                kv_len,
                self.dtype,
                &self.device,
            )?;
        }
        Ok(())
    }

    /// Stage one step: the token, the shared append slot, and the
    /// newly-valid mask column. Tiny H2D writes outside the captured
    /// region; state-free (everything derives from the live length,
    /// and re-enabling an enabled column is a no-op).
    pub(crate) fn stage_step(&self, token: u32, kv_len: usize) -> LibAlmostResult<()> {
        snafu::ensure_whatever!(
            kv_len < self.bucket,
            "context length {} does not fit the {}-wide bucket (ensure_bucket not run?)",
            kv_len,
            self.bucket
        );
        self.ids
            .slice_set(&candle_core::Tensor::new(&[token], &self.device)?, 0, 0)?;
        self.kv_slot.slice_set(
            &candle_core::Tensor::new(&[kv_len as u32], &self.device)?,
            0,
            0,
        )?;
        self.kv_mask.slice_set(&self.mask_zero, 1, kv_len)?;
        Ok(())
    }
}

/// Captured, instantiated decode graphs keyed by bucket width.
/// Wrapped so the containing types keep their derives: Debug renders
/// a summary, Clone shares the instantiated execs.
#[derive(Clone, Default)]
pub(crate) struct GraphCache {
    // Rc, not Arc: CudaGraph is single-thread by cudarc's contract.
    graphs: HashMap<usize, std::rc::Rc<cudarc::driver::CudaGraph>>,
}

impl std::fmt::Debug for GraphCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "GraphCache({} captured)", self.graphs.len())
    }
}

impl GraphCache {
    pub(crate) fn get(&self, key: usize) -> Option<std::rc::Rc<cudarc::driver::CudaGraph>> {
        self.graphs.get(&key).cloned()
    }

    pub(crate) fn insert(&mut self, key: usize, graph: cudarc::driver::CudaGraph) {
        self.graphs.insert(key, std::rc::Rc::new(graph));
    }

    /// Drop every captured graph (the KV buffers reallocated).
    pub(crate) fn clear(&mut self) {
        self.graphs.clear();
    }
}
