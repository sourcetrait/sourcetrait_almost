// Wired into the decode path by E4 phase A (next slice); until then the
// building blocks are exercised by their gpu tests only.
#![allow(dead_code)]

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
