use crate::*;

/// E4 device-side building blocks. The decode graph's per-step dynamics
/// ride STAGED DEVICE BUFFERS (token id, rope position, cache slots):
/// the host writes a few bytes before each replay and the captured ops
/// read values, never baked host constants. slot_write is the cache
/// append in that scheme - a scatter of one (heads, dim) row-block into
/// a (heads, capacity, dim) buffer at a device-resident slot index,
/// launched raw on candle's stream (raw pointer args, guards dropped at
/// launch - the capture-legal form the E4 probes pinned down).
///
/// Kernels are element-size generic (16/32-bit moves), so bf16/f16 and
/// f32/u32 share two entry points and nvrtc needs no dtype headers.
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

/// Lazily nvrtc-compiled module, one per process (single-device box;
/// per-ordinal caching can come with multi-device needs).
static MODULE: std::sync::OnceLock<std::sync::Arc<cudarc::driver::CudaModule>> =
    std::sync::OnceLock::new();

fn kernel(
    device: &candle_core::CudaDevice,
    name: &str,
) -> AlmostResult<cudarc::driver::CudaFunction> {
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

/// Scatter one (b=1, heads, 1, dim) row-block into a (b=1, heads,
/// capacity, dim) buffer at the slot index held in a device u32 buffer:
/// `buffer.inplace_op3(&block, &slot, &SlotWrite { dtype })`. The caller
/// states the operand dtype (as_cuda_slice validates it); all three
/// tensors must be contiguous; the write is capture-legal (raw pointer
/// args on the model's own stream).
pub(crate) struct SlotWrite {
    pub(crate) dtype: candle_core::DType,
}

#[cfg(feature = "cuda")]
fn launch_slot_write<T: candle_core::cuda::CudaDType>(
    function: &cudarc::driver::CudaFunction,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    dst: &candle_core::CudaStorage,
    src: &candle_core::CudaStorage,
    slot: &candle_core::CudaStorage,
    (heads, capacity, dim): (u32, u32, u32),
) -> candle_core::Result<()> {
    use cudarc::driver::{DevicePtr, PushKernelArg};

    let (dst_ptr, _dst_guard) = T::as_cuda_slice(dst)?.device_ptr(stream);
    let (src_ptr, _src_guard) = T::as_cuda_slice(src)?.device_ptr(stream);
    let (slot_ptr, _slot_guard) = slot.as_cuda_slice::<u32>()?.device_ptr(stream);

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

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        dst: &mut candle_core::CudaStorage,
        dst_layout: &candle_core::Layout,
        src: &candle_core::CudaStorage,
        src_layout: &candle_core::Layout,
        slot: &candle_core::CudaStorage,
        slot_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        let dst_dims = dst_layout.shape().dims4()?;
        let src_dims = src_layout.shape().dims4()?;
        let (batch, heads, capacity, dim) = dst_dims;
        if batch != 1 || src_dims != (1, heads, 1, dim) {
            candle_core::bail!(
                "slot_write wants dst (1, h, cap, d) and src (1, h, 1, d); got {dst_dims:?} and {src_dims:?}"
            );
        }
        if !dst_layout.is_contiguous()
            || !src_layout.is_contiguous()
            || !slot_layout.is_contiguous()
            || dst_layout.start_offset() != 0
        {
            candle_core::bail!("slot_write wants contiguous operands (dst at offset 0)");
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
        // Raw pointer args: the capture-legal launch form (safe-slice
        // args wait per-slice events, which is CAPTURE_ISOLATION inside
        // a graph capture when the events predate it).
        match self.dtype {
            candle_core::DType::BF16 => {
                launch_slot_write::<half::bf16>(&function, &stream, dst, src, slot, dims)
            }
            candle_core::DType::F16 => {
                launch_slot_write::<half::f16>(&function, &stream, dst, src, slot, dims)
            }
            candle_core::DType::F32 => {
                launch_slot_write::<f32>(&function, &stream, dst, src, slot, dims)
            }
            candle_core::DType::U32 => {
                launch_slot_write::<u32>(&function, &stream, dst, src, slot, dims)
            }
            _ => unreachable!(),
        }
    }
}

/// Bucket grain for graph-mode attention widths: the kv narrows are
/// static per graph, so valid widths round up to this and the pad mask
/// hides the tail - at most one grain of padded compute, and a small
/// graph-cache population for phase C.
pub(crate) const BUCKET_GRAIN: usize = 2048;

/// Smallest bucket covering `len` valid entries (at least one grain).
pub(crate) fn bucket_for(len: usize) -> usize {
    len.max(1).div_ceil(BUCKET_GRAIN) * BUCKET_GRAIN
}

/// Additive pad-mask values: 0.0 below `valid`, -inf across the
/// bucket's pad tail (post-softmax pad mass is exactly zero).
pub(crate) fn pad_mask_values(bucket: usize, valid: usize) -> Vec<f32> {
    (0..bucket)
        .map(|column| if column < valid { 0.0 } else { f32::NEG_INFINITY })
        .collect()
}

/// E4 staged decode state: the persistent device buffers every
/// per-step dynamic of the graph-mode decode reads (token id, rope
/// position, append slots, pad masks, the logits output). The host
/// stages a few bytes before each step; the op sequence reads values,
/// never baked host constants - the same sequence phase C captures.
/// Armed by Model::arm_graph_decode after prefill; dropped on
/// clear/restore epochs.
#[derive(Debug, Clone)]
pub(crate) struct DecodeStage {
    /// (1, 1) u32: the token this step feeds (embedding index_select).
    pub(crate) ids: candle_core::Tensor,
    /// (1,) u32: the token's absolute rope position.
    pub(crate) position: candle_core::Tensor,
    /// (1,) u32: full-layer append slot (= the live full length).
    pub(crate) full_slot: candle_core::Tensor,
    /// (1,) u32: sliding-layer ring append slot.
    pub(crate) ring_slot: candle_core::Tensor,
    /// (1, 1, 1, full_bucket) additive 0/-inf, model dtype.
    pub(crate) full_mask: candle_core::Tensor,
    /// (1, 1, 1, ring_bucket) additive 0/-inf, model dtype.
    pub(crate) ring_mask: candle_core::Tensor,
    /// (1, vocab) f32: the step's logits, written in-graph.
    pub(crate) logits_out: candle_core::Tensor,
    /// (1, 1, 1, 1) model-dtype 0.0: the mask-enable write source.
    mask_zero: candle_core::Tensor,
    /// (1, heads, 1, 1) f32 zeros: the score-slot reset source.
    pub(crate) score_zero: candle_core::Tensor,
    pub(crate) full_bucket: usize,
    pub(crate) ring_bucket: usize,
    window: usize,
    dtype: candle_core::DType,
    device: candle_core::Device,
}

impl DecodeStage {
    /// Build from the live post-prefill cache state (`full_len` /
    /// `ring_len` valid entries per layer type, rings head-0 linear
    /// unless saturated).
    pub(crate) fn new(
        full_len: usize,
        ring_len: usize,
        window: usize,
        heads: usize,
        vocab: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> AlmostResult<Self> {
        let full_bucket = bucket_for(full_len + 1);
        let ring_bucket = bucket_for((ring_len + 1).min(window)).min(window);
        let full_mask = Self::mask_tensor(full_bucket, full_len, dtype, device)?;
        let ring_mask = Self::mask_tensor(ring_bucket, ring_len, dtype, device)?;
        Ok(Self {
            ids: candle_core::Tensor::zeros((1, 1), candle_core::DType::U32, device)?,
            position: candle_core::Tensor::zeros((1,), candle_core::DType::U32, device)?,
            full_slot: candle_core::Tensor::zeros((1,), candle_core::DType::U32, device)?,
            ring_slot: candle_core::Tensor::zeros((1,), candle_core::DType::U32, device)?,
            full_mask,
            ring_mask,
            logits_out: candle_core::Tensor::zeros(
                (1, vocab),
                candle_core::DType::F32,
                device,
            )?,
            mask_zero: candle_core::Tensor::zeros((1, 1, 1, 1), dtype, device)?,
            score_zero: candle_core::Tensor::zeros(
                (1, heads, 1, 1),
                candle_core::DType::F32,
                device,
            )?,
            full_bucket,
            ring_bucket,
            window,
            dtype,
            device: device.clone(),
        })
    }

    fn mask_tensor(
        bucket: usize,
        valid: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> AlmostResult<candle_core::Tensor> {
        let values = pad_mask_values(bucket, valid);
        Ok(candle_core::Tensor::from_vec(values, (1, 1, 1, bucket), device)?.to_dtype(dtype)?)
    }

    /// Re-derive both buckets for the widths the NEXT append reaches
    /// and rebuild a mask when its bucket changes (a crossing, or a
    /// store shrunk by an eviction epoch). Uncaptured host work; phase
    /// C also switches graphs here.
    pub(crate) fn ensure_buckets(&mut self, full_len: usize, ring_len: usize) -> AlmostResult<()> {
        let full_bucket = bucket_for(full_len + 1);
        if full_bucket != self.full_bucket {
            self.full_bucket = full_bucket;
            self.full_mask = Self::mask_tensor(full_bucket, full_len, self.dtype, &self.device)?;
        }
        let ring_bucket = bucket_for((ring_len + 1).min(self.window)).min(self.window);
        if ring_bucket != self.ring_bucket {
            self.ring_bucket = ring_bucket;
            self.ring_mask = Self::mask_tensor(
                ring_bucket,
                ring_len.min(self.window),
                self.dtype,
                &self.device,
            )?;
        }
        Ok(())
    }

    /// Stage one step: the token, its position, both append slots, and
    /// the newly-valid mask columns. Tiny H2D writes outside the
    /// (future) captured region; state-free - everything derives from
    /// the live lengths, and re-enabling an enabled column is a no-op.
    pub(crate) fn stage_step(
        &self,
        token: u32,
        offset: usize,
        full_len: usize,
        ring_len: usize,
        ring_head: usize,
    ) -> AlmostResult<()> {
        self.ids
            .slice_set(&candle_core::Tensor::new(&[[token]], &self.device)?, 0, 0)?;
        self.position.slice_set(
            &candle_core::Tensor::new(&[offset as u32], &self.device)?,
            0,
            0,
        )?;
        snafu::ensure_whatever!(
            full_len < self.full_bucket,
            "full store length {} does not fit the {}-wide bucket (ensure_buckets not run?)",
            full_len,
            self.full_bucket
        );
        self.full_slot.slice_set(
            &candle_core::Tensor::new(&[full_len as u32], &self.device)?,
            0,
            0,
        )?;
        self.full_mask.slice_set(&self.mask_zero, 3, full_len)?;
        let ring_slot = if ring_len < self.window {
            let slot = (ring_head + ring_len) % self.window;
            snafu::ensure_whatever!(
                slot < self.ring_bucket,
                "ring slot {} does not fit the {}-wide bucket (ensure_buckets not run?)",
                slot,
                self.ring_bucket
            );
            self.ring_mask.slice_set(&self.mask_zero, 3, slot)?;
            slot
        } else {
            // Saturated ring: every slot is valid (static mask); the
            // write rotates through the oldest slot.
            ring_head
        };
        self.ring_slot.slice_set(
            &candle_core::Tensor::new(&[ring_slot as u32], &self.device)?,
            0,
            0,
        )?;
        Ok(())
    }
}
