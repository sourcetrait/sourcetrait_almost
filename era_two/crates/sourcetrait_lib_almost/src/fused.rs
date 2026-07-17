//! GdnChainFusion building blocks: the fused gated-delta decode step -
//! the per-token recurrence (decay, kv memory, delta, state update,
//! y readout) as ONE raw nvrtc kernel over the f32 [heads, dk, dv]
//! state IN PLACE, replacing the ~7-kernel candle chain per GDN layer
//! per step. Raw pointer args on the model's own stream, operands
//! addressed at their layout offsets (the offset-leak lock) -
//! capture-legal exactly like slot_write.
//!
//! Numerics: the same per-element math as gdn::recurrent_step with two
//! deliberate reassociations - the decay fold (decay * sum(state * k)
//! for sum((state * decay) * k)) and serial in-thread reductions over
//! dk (candle sums tree-wise) - so cuda readings move within the bf16
//! envelope class; the cpu path never dispatches here and stays
//! bit-stable by construction.
use crate::*;

/// One block per head, one thread per value column: pass 1 reduces the
/// kv memory for column j, pass 2 updates the state column in place
/// and reduces the y readout. packed rows are q | k | v | decay | beta
/// (2*dk + dv + 2 columns).
const KERNEL_SRC: &str = r#"
extern "C" __global__ void gdn_fused_step_f32(
    float* state,
    const float* packed,
    float* y,
    unsigned int heads,
    unsigned int dk,
    unsigned int dv
) {
    unsigned int h = blockIdx.x;
    unsigned int j = threadIdx.x;
    if (h >= heads || j >= dv) { return; }
    const float* q = packed + (unsigned long long)h * (2 * dk + dv + 2);
    const float* k = q + dk;
    const float* v = k + dk;
    float decay = v[dv];
    float beta = v[dv + 1];
    float* s = state + ((unsigned long long)h * dk) * dv + j;
    float kv = 0.f;
    for (unsigned int i = 0; i < dk; ++i) {
        kv += s[(unsigned long long)i * dv] * k[i];
    }
    kv *= decay;
    float delta = (v[j] - kv) * beta;
    float acc = 0.f;
    for (unsigned int i = 0; i < dk; ++i) {
        float value = s[(unsigned long long)i * dv] * decay + k[i] * delta;
        s[(unsigned long long)i * dv] = value;
        acc += value * q[i];
    }
    y[(unsigned long long)h * dv + j] = acc;
}

__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    unsigned int wide = ((unsigned int)bits) << 16;
    return __uint_as_float(wide);
}

__device__ __forceinline__ unsigned short f32_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return (unsigned short)(rounded >> 16);
}

// One block per row: strided f32 sum-of-squares reduction over C,
// then normalize + weight, all in f32, one rounding to bf16 at the
// store (the classic chain rounds normed to bf16 BEFORE the weight
// mul - a one-rounding reassociation the envelope gates arbitrate).
extern "C" __global__ void rms_norm_fused_bf16(
    unsigned short* out,
    const unsigned short* x,
    const unsigned short* weight,
    float eps,
    unsigned int columns
) {
    extern __shared__ float partial[];
    unsigned long long row = blockIdx.x;
    const unsigned short* x_row = x + row * columns;
    unsigned short* out_row = out + row * columns;
    float sumsq = 0.f;
    for (unsigned int i = threadIdx.x; i < columns; i += blockDim.x) {
        float value = bf16_to_f32(x_row[i]);
        sumsq += value * value;
    }
    partial[threadIdx.x] = sumsq;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv = rsqrtf(partial[0] / (float)columns + eps);
    for (unsigned int i = threadIdx.x; i < columns; i += blockDim.x) {
        float normed = bf16_to_f32(x_row[i]) * inv;
        out_row[i] = f32_to_bf16(normed * bf16_to_f32(weight[i]));
    }
}

// The gated form over per-head rows: y (f32) and gate (f32,
// pre-upcast) ride one packed [heads, 2 * dv] tensor; norm over dv,
// weight [dv], gate = silu(gate) in f32, one rounding to bf16.
extern "C" __global__ void rms_norm_gated_fused_bf16(
    unsigned short* out,
    const float* packed,
    const unsigned short* weight,
    float eps,
    unsigned int dv
) {
    extern __shared__ float partial[];
    unsigned long long head = blockIdx.x;
    const float* y_row = packed + head * (2ull * dv);
    const float* gate_row = y_row + dv;
    unsigned short* out_row = out + head * dv;
    float sumsq = 0.f;
    for (unsigned int i = threadIdx.x; i < dv; i += blockDim.x) {
        sumsq += y_row[i] * y_row[i];
    }
    partial[threadIdx.x] = sumsq;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv = rsqrtf(partial[0] / (float)dv + eps);
    for (unsigned int i = threadIdx.x; i < dv; i += blockDim.x) {
        float weighted = y_row[i] * inv * bf16_to_f32(weight[i]);
        float gate = gate_row[i];
        float silu = gate / (1.f + expf(-gate));
        out_row[i] = f32_to_bf16(weighted * silu);
    }
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
                Err(error) => {
                    snafu::whatever!("nvrtc compile of the fused kernels failed: {error}")
                }
            };
            let module = match device.cuda_stream().context().load_module(ptx) {
                Ok(module) => module,
                Err(error) => {
                    snafu::whatever!("loading the fused-kernel module failed: {error}")
                }
            };
            MODULE.get_or_init(|| module).clone()
        }
    };
    match module.load_function(name) {
        Ok(function) => Ok(function),
        Err(error) => snafu::whatever!("loading kernel {name} failed: {error}"),
    }
}

/// The norm kernels' block width (power of two - the tree reduction
/// requires it; shared memory is one f32 per thread).
const NORM_BLOCK: u32 = 256;

/// `state.inplace_op3(&packed, &y, &GdnFusedStep)`: state
/// (heads, dk, dv) f32 updated IN PLACE; packed
/// (heads, 2*dk + dv + 2) f32 rows q | k | v | decay | beta; y
/// (heads, dv) f32 - WRITTEN by the kernel (the third operand is an
/// output; candle tensors mutate through shared storage by design,
/// same as slice_set). All f32, all contiguous, all addressed at
/// their layout offsets.
pub(crate) struct GdnFusedStep;

impl candle_core::InplaceOp3 for GdnFusedStep {
    fn name(&self) -> &'static str {
        "almost_gdn_fused_step"
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
        candle_core::bail!("gdn_fused_step is a cuda decode building block")
    }

    fn cuda_fwd(
        &self,
        state: &mut candle_core::CudaStorage,
        state_layout: &candle_core::Layout,
        packed: &candle_core::CudaStorage,
        packed_layout: &candle_core::Layout,
        y: &candle_core::CudaStorage,
        y_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, dk, dv) = state_layout.shape().dims3()?;
        let packed_dims = packed_layout.shape().dims2()?;
        let y_dims = y_layout.shape().dims2()?;
        if packed_dims != (heads, 2 * dk + dv + 2) || y_dims != (heads, dv) {
            candle_core::bail!(
                "gdn_fused_step wants state (h, dk, dv), packed (h, 2dk+dv+2), y (h, dv); \
                 got {:?}, {packed_dims:?}, {y_dims:?}",
                (heads, dk, dv)
            );
        }
        if !state_layout.is_contiguous()
            || !packed_layout.is_contiguous()
            || !y_layout.is_contiguous()
        {
            candle_core::bail!("gdn_fused_step wants contiguous operands");
        }
        if dv > 1024 {
            candle_core::bail!("gdn_fused_step blocks one thread per value column (dv <= 1024)");
        }

        let device = state.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "gdn_fused_step_f32") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };

        let state_slice = state.as_cuda_slice::<f32>()?.slice(state_layout.start_offset()..);
        let (state_ptr, _state_guard) = state_slice.device_ptr(&stream);
        let packed_slice = packed
            .as_cuda_slice::<f32>()?
            .slice(packed_layout.start_offset()..);
        let (packed_ptr, _packed_guard) = packed_slice.device_ptr(&stream);
        let y_slice = y.as_cuda_slice::<f32>()?.slice(y_layout.start_offset()..);
        let (y_ptr, _y_guard) = y_slice.device_ptr(&stream);

        let (heads, dk, dv) = (heads as u32, dk as u32, dv as u32);
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (heads, 1, 1),
            block_dim: (dv, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&state_ptr)
            .arg(&packed_ptr)
            .arg(&y_ptr)
            .arg(&heads)
            .arg(&dk)
            .arg(&dv);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("gdn_fused_step launch failed: {error}");
        }
        Ok(())
    }
}

/// `out.inplace_op3(&x, &weight, &RmsNormFused { eps })`: out and x
/// (rows, columns) bf16, weight (columns,) bf16; the whole
/// cast/sqr/mean/rsqrt/normalize/weight chain in one launch per call,
/// one rounding to bf16 at the store (the classic chain rounds normed
/// before the weight mul - an envelope-class reassociation).
pub(crate) struct RmsNormFused {
    pub(crate) eps: f32,
}

impl candle_core::InplaceOp3 for RmsNormFused {
    fn name(&self) -> &'static str {
        "almost_rms_norm_fused"
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
        candle_core::bail!("rms_norm_fused is a cuda decode building block")
    }

    fn cuda_fwd(
        &self,
        out: &mut candle_core::CudaStorage,
        out_layout: &candle_core::Layout,
        x: &candle_core::CudaStorage,
        x_layout: &candle_core::Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (rows, columns) = out_layout.shape().dims2()?;
        if x_layout.shape().dims2()? != (rows, columns)
            || weight_layout.shape().dims1()? != columns
        {
            candle_core::bail!(
                "rms_norm_fused wants out/x (rows, columns) and weight (columns,)"
            );
        }
        if !out_layout.is_contiguous()
            || !x_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!("rms_norm_fused wants contiguous operands");
        }

        let device = out.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "rms_norm_fused_bf16") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let out_slice = out
            .as_cuda_slice::<half::bf16>()?
            .slice(out_layout.start_offset()..);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        let x_slice = x
            .as_cuda_slice::<half::bf16>()?
            .slice(x_layout.start_offset()..);
        let (x_ptr, _x_guard) = x_slice.device_ptr(&stream);
        let weight_slice = weight
            .as_cuda_slice::<half::bf16>()?
            .slice(weight_layout.start_offset()..);
        let (weight_ptr, _weight_guard) = weight_slice.device_ptr(&stream);

        let columns = columns as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (NORM_BLOCK, 1, 1),
            shared_mem_bytes: NORM_BLOCK * 4,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&out_ptr)
            .arg(&x_ptr)
            .arg(&weight_ptr)
            .arg(&self.eps)
            .arg(&columns);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("rms_norm_fused launch failed: {error}");
        }
        Ok(())
    }
}

/// `out.inplace_op3(&packed, &weight, &RmsNormGatedFused { eps })`:
/// out (heads, dv) bf16, packed (heads, 2 * dv) f32 rows y | gate
/// (gate pre-upcast; y raw f32 from the fused step - the classic
/// path's y->bf16 round-trip before the variance disappears), weight
/// (dv,) bf16. Norm-then-gate with silu in f32, one rounding out.
pub(crate) struct RmsNormGatedFused {
    pub(crate) eps: f32,
}

impl candle_core::InplaceOp3 for RmsNormGatedFused {
    fn name(&self) -> &'static str {
        "almost_rms_norm_gated_fused"
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
        candle_core::bail!("rms_norm_gated_fused is a cuda decode building block")
    }

    fn cuda_fwd(
        &self,
        out: &mut candle_core::CudaStorage,
        out_layout: &candle_core::Layout,
        packed: &candle_core::CudaStorage,
        packed_layout: &candle_core::Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, dv) = out_layout.shape().dims2()?;
        if packed_layout.shape().dims2()? != (heads, 2 * dv)
            || weight_layout.shape().dims1()? != dv
        {
            candle_core::bail!(
                "rms_norm_gated_fused wants out (h, dv), packed (h, 2dv), weight (dv,)"
            );
        }
        if !out_layout.is_contiguous()
            || !packed_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!("rms_norm_gated_fused wants contiguous operands");
        }

        let device = out.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "rms_norm_gated_fused_bf16") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let out_slice = out
            .as_cuda_slice::<half::bf16>()?
            .slice(out_layout.start_offset()..);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        let packed_slice = packed
            .as_cuda_slice::<f32>()?
            .slice(packed_layout.start_offset()..);
        let (packed_ptr, _packed_guard) = packed_slice.device_ptr(&stream);
        let weight_slice = weight
            .as_cuda_slice::<half::bf16>()?
            .slice(weight_layout.start_offset()..);
        let (weight_ptr, _weight_guard) = weight_slice.device_ptr(&stream);

        let dv = dv as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (heads as u32, 1, 1),
            block_dim: (NORM_BLOCK, 1, 1),
            shared_mem_bytes: NORM_BLOCK * 4,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&out_ptr)
            .arg(&packed_ptr)
            .arg(&weight_ptr)
            .arg(&self.eps)
            .arg(&dv);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("rms_norm_gated_fused launch failed: {error}");
        }
        Ok(())
    }
}
